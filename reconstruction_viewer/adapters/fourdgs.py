import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


class FourDGS:
    def __init__(self, entry):
        root = Path(entry["root"])
        sys.path.insert(0, str(entry.get("code_root", root / "4DGaussians")))
        import mmcv
        from arguments import ModelHiddenParams, ModelParams, PipelineParams
        from gaussian_renderer import render
        from scene import GaussianModel, Scene
        from scene.cameras import MiniCam
        from utils.graphics_utils import getProjectionMatrix
        from utils.params_utils import merge_hparams

        parser = argparse.ArgumentParser()
        model, hidden, pipeline = ModelParams(parser), ModelHiddenParams(parser), PipelineParams(parser)
        args = parser.parse_args([])
        args.source_path = str(root / entry["source"])
        args.model_path = str(root / entry["model"])
        args = merge_hparams(args, mmcv.Config.fromfile(root / entry["config"]))
        self.gaussians = GaussianModel(args.sh_degree, hidden.extract(args))
        scene = Scene(model.extract(args), self.gaussians, load_iteration=entry["iteration"], shuffle=False)
        self.gaussians._deformation.eval()
        self.pipe = pipeline.extract(args)
        self.background = torch.ones(3, device="cuda")
        self.render_model, self.camera_class, self.projection = render, MiniCam, getProjectionMatrix
        self.dataset_type = scene.dataset_type
        metrics = json.loads((root / entry["results"] / "metrics.json").read_text())
        reference_index = metrics["test_frames"] // 2
        reference = scene.getTestCameras()[reference_index]
        c2w = np.linalg.inv(reference.world_view_transform.numpy().T)
        xyz = self.gaussians.get_xyz.detach().cpu().numpy()[::max(1, self.gaussians.get_xyz.shape[0] // 2000)]
        distances = (np.linalg.inv(c2w)[:3, :3] @ xyz.T).T[:, 2] + np.linalg.inv(c2w)[2, 3]
        positive = distances[distances > 0]
        distance = float(np.median(positive)) if len(positive) else 1.0
        self.metadata = {
            "c2w": c2w.tolist(),
            "fov": reference.FoVy,
            "resolution": [reference.image_width, reference.image_height],
            "times": [row["normalized_time"] for row in metrics["frames"]],
            "initial_frame": reference_index,
            "look_at": (c2w[:3, 3] + c2w[:3, 2] * distance).tolist(),
            "gaussians": int(self.gaussians.get_xyz.shape[0]),
        }

    @torch.no_grad()
    def render(self, request):
        width, height = request["size"]
        w2c = np.asarray(request["w2c"], dtype=np.float32)
        fovy = float(request["fov"])
        fovx = float(2 * np.arctan(np.tan(fovy / 2) * width / height))
        view = torch.from_numpy(w2c).T
        projection = self.projection(znear=.01, zfar=100., fovX=fovx, fovY=fovy).T
        camera = self.camera_class(width, height, fovy, fovx, .01, 100., view, view @ projection, float(request["time"]))
        output = self.render_model(camera, self.gaussians, self.pipe, self.background, cam_type=self.dataset_type)["render"]
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite 4DGS render")
        return (output.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
