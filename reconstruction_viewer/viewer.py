import argparse
import atexit
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import viser
from PIL import Image
from viser import transforms

from backend import Backend
from media import start_media
from registry import load_registry

ROOT = Path(__file__).resolve().parent


class ReconstructionViewer:
    def __init__(self, registry, port, media_port):
        self.registry = registry
        self.entries = load_registry(registry)
        self.by_label = {entry["label"]: entry for entry in self.entries}
        self.backends, self.clients = {}, {}
        self.state_lock, self.switch_lock = threading.RLock(), threading.Lock()
        self.entry, self.metadata, self.revision = None, None, 0
        self.closed = threading.Event()
        self.media_port = media_port
        self.media = start_media(self.entries, media_port, port, ROOT)
        self.server = viser.ViserServer(host="127.0.0.1", port=port, label="动态重建 · 统一查看器", verbose=False)
        self.server.gui.configure_theme(dark_mode=True, brand_color=(99, 166, 224))
        self.server.gui.main_panel.dock_right()
        self.server.scene.world_axes.visible = False
        self.server.gui.add_markdown("# 动态重建\n拖动：旋转 · 右键拖动：平移 · 滚轮：缩放\n\n画面来自动态模型实时渲染，不是视频回放。")
        self.scene = self.server.gui.add_dropdown("方法与场景", options=list(self.by_label))
        self.status = self.server.gui.add_markdown("正在加载模型…")
        with self.server.gui.add_folder("时间"):
            self.frame = self.server.gui.add_slider("时间帧", min=0, max=1, step=1, initial_value=0, disabled=True)
            self.timeline = self.server.gui.add_markdown("")
            self.playing = self.server.gui.add_checkbox("播放动态", initial_value=False)
            self.fps = self.server.gui.add_slider("播放速度", min=1, max=30, step=1, initial_value=15)
        with self.server.gui.add_folder("视角与画面"):
            self.reset = self.server.gui.add_button("恢复参考视角")
            self.resolution = self.server.gui.add_slider("预览分辨率", min=256, max=1280, step=64, initial_value=768)
            self.save = self.server.gui.add_button("保存当前画面")
        self.details = self.server.gui.add_markdown("")
        self.links = self.server.gui.add_markdown("")
        self.export = self.server.gui.add_markdown("")

        self.scene.on_update(lambda _: self.select_scene())
        self.frame.on_update(lambda _: self.request_render())
        self.resolution.on_update(lambda _: self.request_render())
        self.reset.on_click(lambda _: self.reset_cameras())
        self.save.on_click(self.save_image)
        self.server.on_client_connect(self.connect)
        self.server.on_client_disconnect(self.disconnect)
        self.select_scene()
        threading.Thread(target=self.playback, daemon=True).start()
        atexit.register(self.close)

    def select_scene(self):
        requested = self.scene.value
        with self.switch_lock:
            if requested != self.scene.value or self.closed.is_set():
                return
            entry = self.by_label[requested]
            self.playing.value = False
            self.fps.value = entry.get("fps", 15)
            self.resolution.value = entry.get("preview_resolution", 768)
            self.frame.disabled = True
            self.status.content = f'正在加载 {entry["label"]}…'
            self.details.content = ""
            self.links.content = ""
            self.export.content = ""
            with self.state_lock:
                self.entry, self.metadata = None, None
                self.revision += 1
            for client in self.server.get_clients().values():
                client.scene.set_background_image(None)
            try:
                for scene_id, backend in list(self.backends.items()):
                    if scene_id != entry["id"]:
                        backend.close()
                        del self.backends[scene_id]
                if entry["id"] not in self.backends:
                    self.backends[entry["id"]] = Backend(entry, self.registry)
                metadata = self.backends[entry["id"]].metadata(entry["id"])
                with self.state_lock:
                    self.frame.value = 0
                    self.frame.max = len(metadata["times"]) - 1
                    self.frame.value = metadata["initial_frame"]
                    self.entry, self.metadata = entry, metadata
                    self.revision += 1
                self.frame.disabled = False
                metrics = json.loads((Path(entry["root"]) / entry["results"] / "metrics.json").read_text())
                values = metrics.get("metrics", {}).get("official_covisible", metrics)
                self.details.content = (f'**{entry["label"]}**\n\n{metadata["gaussians"]:,} Gaussians · {len(metadata["times"])} 时间帧'
                                        f'\n\nPSNR {values["psnr"]:.2f} dB · SSIM {values["ssim"]:.3f}\n\n指标范围：{entry["metric_scope"]}。仅展示已测试场景，非全屋导航或物理仿真。')
                self.links.content = f'[真实／重建对照视频](http://127.0.0.1:{self.media_port}/scene/{entry["id"]}) · [全部复现](http://127.0.0.1:{self.media_port}/)'
                self.status.content = "就绪 · 可自由改变视角与时间"
                self.reset_cameras()
                self.request_render()
            except Exception as error:
                logging.exception("Scene load failed")
                self.status.content = f"加载失败：{error}。检查 logs/。"

    def reset_camera(self, client):
        with self.state_lock:
            metadata = self.metadata
        if metadata is None:
            return
        c2w = np.asarray(metadata["c2w"])
        with self.server.atomic():
            client.camera.position = c2w[:3, 3]
            client.camera.up_direction = -c2w[:3, 1]
            client.camera.look_at = np.asarray(metadata["look_at"])
            client.camera.wxyz = transforms.SO3.from_matrix(c2w[:3, :3]).wxyz
            client.camera.fov = metadata["fov"]

    def reset_cameras(self):
        for client in self.server.get_clients().values():
            self.reset_camera(client)
        self.request_render()

    def connect(self, client):
        event = threading.Event()
        self.clients[client.client_id] = event
        client.camera.on_update(lambda _: event.set())
        self.reset_camera(client)
        threading.Thread(target=self.render_client, args=(client, event), daemon=True).start()
        event.set()

    def disconnect(self, client):
        event = self.clients.pop(client.client_id, None)
        if event:
            event.set()

    def request_render(self):
        with self.state_lock:
            metadata = self.metadata
            frame = int(self.frame.value)
        if metadata:
            progress = frame / max(1, len(metadata["times"]) - 1)
            self.timeline.content = f'帧 {frame}/{len(metadata["times"]) - 1} · 进度 {progress:.1%}'
        for event in list(self.clients.values()):
            event.set()

    def query(self, client):
        with self.state_lock:
            if self.entry is None or self.metadata is None:
                return None
            entry, metadata, revision = self.entry, self.metadata, self.revision
            frame = int(self.frame.value)
            time_value = metadata["times"][frame]
        c2w = np.eye(4)
        c2w[:3, :3] = transforms.SO3(client.camera.wxyz).as_matrix()
        c2w[:3, 3] = client.camera.position
        aspect = float(client.camera.aspect)
        size = int(self.resolution.value)
        width, height = (size, max(1, round(size / aspect))) if aspect >= 1 else (max(1, round(size * aspect)), size)
        return entry, revision, frame, dict(size=[width, height], w2c=np.linalg.inv(c2w).tolist(), fov=float(client.camera.fov), time=time_value)

    def render_client(self, client, event):
        while not self.closed.is_set() and client.client_id in self.clients:
            event.wait(timeout=1)
            if not event.is_set():
                continue
            event.clear()
            request = self.query(client)
            if request is None:
                continue
            entry, revision, frame, query = request
            try:
                image = self.backends[entry["id"]].render(entry["id"], **query)
                if revision == self.revision and client.client_id in self.clients:
                    client.scene.set_background_image(image, format="jpeg", jpeg_quality=90)
            except Exception as error:
                logging.exception("Render failed")
                self.status.content = f"渲染失败：{error}"
                self.playing.value = False

    def playback(self):
        while not self.closed.wait(1 / float(self.fps.value)):
            with self.state_lock:
                if self.playing.value and self.metadata is not None:
                    self.frame.value = (int(self.frame.value) + 1) % len(self.metadata["times"])

    def save_image(self, event):
        if event.client is None:
            return
        request = self.query(event.client)
        if request is None:
            return
        entry, revision, frame, query = request
        self.export.content = "正在保存当前画面…"
        image = self.backends[entry["id"]].render(entry["id"], **query)
        destination = ROOT / "screenshots"
        destination.mkdir(exist_ok=True)
        name = f'{entry["id"]}_frame{frame:04d}_{datetime.now():%Y%m%d_%H%M%S_%f}.png'
        Image.fromarray(image).save(destination / name)
        (destination / name).with_suffix(".json").write_text(json.dumps({"scene": entry["id"], "frame": frame, **query}, indent=2))
        if revision == self.revision:
            self.export.content = f'[下载当前模型渲染](http://127.0.0.1:{self.media_port}/screenshots/{name})'

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.media.shutdown()
        self.media.server_close()
        for backend in self.backends.values():
            backend.close()
        self.server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=ROOT / "scenes.json")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--media-port", type=int, default=8900)
    args = parser.parse_args()
    viewer = ReconstructionViewer(args.registry, args.port, args.media_port)
    print(f"Unified viewer: http://127.0.0.1:{args.port}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.close()


if __name__ == "__main__":
    main()
