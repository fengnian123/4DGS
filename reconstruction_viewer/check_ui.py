import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import expect, sync_playwright
from registry import load_registry

ROOT = Path(__file__).resolve().parent
destination = ROOT / "tests"
destination.mkdir(exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--first-scene')
parser.add_argument('--registry', type=Path, default=ROOT / 'scenes.json')
parser.add_argument('--url', default='http://127.0.0.1:8890')
parser.add_argument('--browser-executable')
args = parser.parse_args()
entries = load_registry(args.registry)
default_label = entries[0]['label']
if args.first_scene:
    assert any(entry['id'] == args.first_scene for entry in entries)
    entries.sort(key=lambda entry: entry['id'] != args.first_scene)
labels = [entry["label"] for entry in entries]
errors = []
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path=args.browser_executable, headless=True, args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(args.url)
    def select(label):
        if page.get_by_role("combobox").first.input_value() != label:
            page.get_by_role("combobox").first.click()
            page.get_by_role("option", name=label, exact=True).click()
        page.locator("strong").filter(has_text=label).wait_for(timeout=60000)
        page.get_by_text("就绪 · 可自由改变视角与时间", exact=True).wait_for(timeout=60000)

    select(labels[0])
    page.get_by_text("就绪 · 可自由改变视角与时间", exact=True).wait_for(timeout=60000)
    page.wait_for_timeout(5000)
    first_image = destination / f'{entries[0]["id"]}_ui.png'
    page.screenshot(path=str(first_image))
    variation = float(np.asarray(Image.open(first_image).crop((100, 200, 850, 700))).std())
    assert variation > 8, variation
    marker = page.get_by_text(re.compile(r"^帧 \d+/\d+ · 进度"))
    before_playback = marker.inner_text()
    page.get_by_text("播放动态", exact=True).click()
    page.wait_for_timeout(2000)
    page.get_by_text("播放动态", exact=True).click()
    after_playback = marker.inner_text()
    assert before_playback != after_playback
    def snapshot():
        link = page.get_by_text("下载当前模型渲染", exact=True)
        previous = link.get_attribute("href") if link.count() else None
        page.get_by_role("button", name="保存当前画面", exact=True).click()
        link.wait_for(timeout=15000)
        if previous is not None:
            expect(link).not_to_have_attribute("href", previous, timeout=15000)
        url = link.get_attribute("href")
        return url, page.request.get(url.removesuffix(".png") + ".json").json()

    _, before_camera = snapshot()
    page.mouse.move(400, 450)
    page.mouse.down()
    page.mouse.move(470, 420, steps=12)
    page.mouse.up()
    page.wait_for_timeout(2000)
    page.screenshot(path=str(destination / f'{entries[0]["id"]}_orbit_ui.png'))
    export_url, after_camera = snapshot()
    assert after_camera["scene"] == entries[0]["id"]
    camera_change = float(np.abs(np.asarray(after_camera["w2c"]) - np.asarray(before_camera["w2c"])).max())
    assert camera_change > .001, camera_change
    assert after_camera["time"] == before_camera["time"]
    page.get_by_role("button", name="恢复参考视角", exact=True).click()
    for index in range(1, len(entries)):
        select(labels[index])
        page.wait_for_timeout(3000)
        target = destination / f'{entries[index]["id"]}_ui.png'
        page.screenshot(path=str(target))
        variation = float(np.asarray(Image.open(target).crop((100, 200, 850, 700))).std())
        assert variation > 8, (labels[index], variation)
    select(default_label)
    page.wait_for_timeout(1500)
    report = {"page_errors": errors, "scenes_selected": labels, "before_playback": before_playback, "after_playback": after_playback, "camera_transform_max_change": camera_change, "snapshot_saved": export_url}
    print(json.dumps(report, indent=2), flush=True)
    (destination / "ui.json").write_text(json.dumps(report, indent=2))
    browser.close()
assert not errors, errors
