import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
import yaml
from plyfile import PlyData
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
run = ROOT / "runs/balloon1"
repository = ROOT / "shape-of-motion"


def command(*args):
    return subprocess.check_output(args, cwd=repository, text=True).strip()


archive = ROOT / "data/Balloon1.zip"
with archive.open("rb") as stream:
    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
versions = {}
for name in ("torch", "torchvision", "numpy", "gsplat", "cuml-cu12", "cupy-cuda12x", "viser", "nerfview", "torchmetrics", "pytorch-msssim", "roma", "scipy", "plyfile"):
    versions[name] = importlib.metadata.version(name)
history = {}
for path in run.rglob("events.out.tfevents.*"):
    events = EventAccumulator(str(path), size_guidance={"scalars": 0})
    events.Reload()
    for tag in events.Tags().get("scalars", []):
        if tag.startswith("val/"):
            history.setdefault(tag, []).extend({"step": item.step, "value": item.value, "wall_time": item.wall_time} for item in events.Scalars(tag))
record = {
    "repository": "https://github.com/vye16/shape-of-motion",
    "commit": command("git", "rev-parse", "HEAD"),
    "local_diff": command("git", "diff"),
    "python": sys.version,
    "versions": versions,
    "gpu": torch.cuda.get_device_name(),
    "torch_cuda": torch.version.cuda,
    "dataset": "NVIDIA Balloon1, author-provided preprocessed archive",
    "dataset_folder_url": "https://drive.google.com/drive/folders/1xzn-Mu_jyr-JTsrERRU-Mh2hQ-NWdfv8",
    "dataset_file_id": "1atIXDSiNRCh7kOaoB4xU2UlerDZ3CouX",
    "archive_sha256": checksum,
    "archive_bytes": archive.stat().st_size,
    "train_images": 192,
    "validation_images": 576,
    "raw_preprocessing_tested": False,
    "used_private_robot_data": False,
    "requested_motion_bases": 10,
    "actual_motion_bases_note": "The upstream initializer keeps only clusters with more than 100 tracks; 9 of the requested 10 bases survive for this sequence.",
    "config": yaml.safe_load((run / "cfg.yaml").read_text()),
    "validation_history": history,
    "final_evaluation": json.loads((run / "evaluation/metrics.json").read_text()),
}
evaluation = run / "evaluation"
results = record["final_evaluation"]
assert results["validation_images"] == 576 and results["checkpoint_step"] == 12000
assert all(np.isfinite(list(values.values())).all() for values in results["metrics"].values())
video_checks = {}
for name, expected_frames in (("comparison_view1.mp4", 192), ("comparison_view2.mp4", 192), ("comparison_view3.mp4", 192), ("fixed_time_camera_sweep.mp4", 144)):
    with iio.get_reader(evaluation / name) as reader:
        count = reader.count_frames()
        metadata = reader.get_meta_data()
        assert count == expected_frames, (name, count)
        video_checks[name] = {"frames": count, "fps": metadata["fps"], "size": metadata["size"]}
vertices = PlyData.read(str(evaluation / "gaussians_t096.ply"))["vertex"].data
assert len(vertices) == results["num_foreground_gaussians"] + results["num_background_gaussians"]
assert all(np.isfinite(vertices[name]).all() for name in vertices.dtype.names)
with np.load(evaluation / "tracks_3d.npz", allow_pickle=False) as tracks:
    xyz = tracks["xyz_normalized"]
    assert xyz.shape == (512, 192, 3) and np.isfinite(xyz).all()
    motion = float(np.linalg.norm(xyz[:, 1:] - xyz[:, :1], axis=-1).max(axis=1).mean())
    assert motion > 0
record["artifact_checks"] = {"videos": video_checks, "ply_vertices": len(vertices), "trajectory_shape": list(xyz.shape), "mean_max_trajectory_displacement_normalized_units": motion}
record["viewer_check"] = json.loads((run / "interactive_preview.json").read_text())
(run / "experiment.json").write_text(json.dumps(record, indent=2))
print(run / "experiment.json")
