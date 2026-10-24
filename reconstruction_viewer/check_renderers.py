import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend import Backend
from registry import load_registry

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--registry", type=Path, default=ROOT / "scenes.json")
registry = parser.parse_args().registry.resolve()
entries = load_registry(registry)
destination = ROOT / "tests"
destination.mkdir(exist_ok=True)
reports = []
for entry in entries:
    backend = Backend(entry, registry)
    try:
        metadata = backend.metadata(entry["id"])
        frame = metadata["initial_frame"]
        query = dict(w2c=np.linalg.inv(np.asarray(metadata["c2w"])).tolist(), fov=metadata["fov"], size=metadata["resolution"], time=metadata["times"][frame])
        image = backend.render(entry["id"], **query)
        reference_path = Path(entry["root"]) / entry["results"]
        default_pattern = "rgb/1_{frame:05d}.png" if entry["id"].startswith("som-") else "renders/{frame:05d}.png"
        reference_path /= entry.get("reference_pattern", default_pattern).format(frame=frame)
        reference = np.asarray(Image.open(reference_path).convert("RGB"))
        difference = float(np.abs(image.astype(float) - reference.astype(float)).mean())
        Image.fromarray(image).save(destination / f'{entry["id"]}_reference.png')
        later = dict(query, time=metadata["times"][min(frame + 20, len(metadata["times"]) - 1)])
        other = backend.render(entry["id"], **later)
        change = float(np.abs(image.astype(float) - other.astype(float)).mean())
        report = {"scene": entry["id"], "reference_frame": frame, "reference_mae_uint8": difference, "time_change_mae_uint8": change, "resolution": metadata["resolution"]}
        reports.append(report)
        print(json.dumps(report), flush=True)
        assert difference < 1.5, report
        assert change > .02, report
    finally:
        backend.close()
(destination / "renderers.json").write_text(json.dumps(reports, indent=2))
