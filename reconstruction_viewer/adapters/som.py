import json
import sys
from pathlib import Path

import numpy as np
import torch


class ShapeOfMotion:
    def __init__(self, entry):
        root = Path(entry["root"])
        sys.path.insert(0, str(entry.get("code_root", root / "shape-of-motion")))
        from flow3d.scene_model import SceneModel

        # Only load the trusted checkpoint produced by the local reproduction.
        checkpoint = torch.load(root / entry["model"] / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
        self.model = SceneModel.init_from_state_dict(checkpoint["model"]).cuda().eval()
        source = root / entry["source"]
        frame = self.model.num_frames // 2
        camera = json.loads((source / "camera" / f"1_{frame:05d}.json").read_text())
        norm = torch.load(source / "flow3d_preprocessed/cache/scene_norm_dict.pth", weights_only=True)
        orientation, position = np.asarray(camera["orientation"]), np.asarray(camera["position"])
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3], w2c[:3, 3] = orientation, -orientation @ position
        w2c = torch.from_numpy(w2c) @ torch.linalg.inv(norm["transfm"])
        w2c[:3, 3] /= norm["scale"]
        c2w = torch.linalg.inv(w2c).numpy()
        width, height = camera["image_size"]
        self.metadata = {
            "c2w": c2w.tolist(),
            "fov": float(2 * np.arctan(camera["principal_point"][1] / camera["focal_length"])),
            "resolution": [width // 2, height // 2],
            "times": list(range(self.model.num_frames)),
            "initial_frame": frame,
            "look_at": (c2w[:3, 3] + c2w[:3, 2] * 2).tolist(),
            "gaussians": self.model.num_gaussians,
        }

    @torch.no_grad()
    def render(self, request):
        width, height = request["size"]
        focal = .5 * height / np.tan(float(request["fov"]) / 2)
        K = torch.tensor([[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]], device="cuda", dtype=torch.float32)
        w2c = torch.tensor(request["w2c"], device="cuda", dtype=torch.float32)
        output = self.model.render(int(request["time"]), w2c[None], K[None], (width, height))["img"][0]
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite Shape of Motion render")
        return (output.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
