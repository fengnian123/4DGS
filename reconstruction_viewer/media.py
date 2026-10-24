import html
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


def start_media(entries, port, viewer_port, root):
    by_id = {entry["id"]: entry for entry in entries}

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/" or path.startswith("/scene/"):
                if path != "/" and path.split("/")[-1] not in by_id:
                    self.send_error(404)
                    return
                selected = entries if path == "/" else [by_id[path.split("/")[-1]]]
                sections = []
                for entry in selected:
                    directory = Path(entry["root"]) / entry["results"]
                    order = {'real_before_after': 0, 'comparison': 1, 'fixed_reference_before_after': 2, 'camera_sweep': 3}
                    videos = sorted(directory.glob("*.mp4"), key=lambda video: (order.get(video.stem, 4), video.name))
                    titles = {
                        'comparison': '真实画面 / 当前重建（采集视角）',
                        'real_before_after': f'真实画面 / {entry.get("baseline_label", "原版")} / 当前模型（各自采集视角）',
                        'fixed_reference_before_after': f'{entry.get("baseline_label", "原版")} / 当前模型（相同固定视角；无真实画面对照）',
                        'depth_before_after': '实测稀疏深度 / V2 / 实测深度版（米；实测图黑色为未观测）',
                        'camera_sweep': '固定时间的模型视角漫游（无真实画面对照）',
                        'fixed_time_view': '固定时间的模型视角漫游（无真实画面对照）',
                        'fixed_time_camera_sweep': '固定时间的模型视角漫游（无真实画面对照）',
                        'fixed_view_time': '固定相机的模型动态渲染（无逐帧真实画面对照）',
                    }
                    clips = "".join(f'<h3>{html.escape(titles.get(video.stem, video.stem))}</h3><video src="/media/{entry["id"]}/{video.name}" controls playsinline preload="metadata"></video>' for video in videos)
                    sensor_link = (f' · <a href="/media/{entry["id"]}/sensor_metrics.json">留出雷达点深度检查</a>'
                                   if (directory / 'sensor_metrics.json').exists() else '')
                    sections.append(f'<section><h2>{html.escape(entry["label"])}</h2><p>指标范围：{html.escape(entry["metric_scope"])}</p>{clips}<p><a href="/media/{entry["id"]}/metrics.json">逐帧指标</a>{sensor_link}</p></section>')
                body = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                        '<title>动态重建 · 视频对照</title><style>body{max-width:1100px;margin:30px auto;padding:0 20px;background:#11161f;color:#edf0f4;font:16px/1.6 system-ui}video{width:100%;background:#000}section{margin:40px 0}a{color:#8ecbff}</style>'
                        f'<h1>动态重建 · 视频对照</h1><a href="http://127.0.0.1:{viewer_port}">返回统一交互查看器</a>'
                        '<p>请按各视频标题辨认真实画面与模型渲染。固定视角和视角漫游没有逐帧真实画面对照。不同数据集与遮罩范围的指标不直接横向比较。</p>'
                        + "".join(sections)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/media/") or path.startswith("/screenshots/"):
                return super().do_GET()
            self.send_error(404)

        def translate_path(self, path):
            pieces = unquote(urlsplit(path).path).split("/")
            if len(pieces) >= 4 and pieces[1] == "media" and pieces[2] in by_id:
                entry = by_id[pieces[2]]
                directory = (Path(entry["root"]) / entry["results"]).resolve()
                target = (directory / "/".join(pieces[3:])).resolve()
            elif len(pieces) >= 3 and pieces[1] == "screenshots":
                directory = (root / "screenshots").resolve()
                target = (directory / "/".join(pieces[2:])).resolve()
            else:
                return str(root / ".not-found")
            return str(target) if target.is_relative_to(directory) and target.is_file() else str(root / ".not-found")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
