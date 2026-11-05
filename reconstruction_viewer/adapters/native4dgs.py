"""Yang et al. ICLR 2024 native 4D Gaussians for the shared viewer contract."""
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class Native4DGS:
    def __init__(self, entry):
        package = Path(__file__).resolve().parents[2] / 'native4dgs_local'
        os.environ['NATIVE4DGS_ROOT'] = str(entry['root'])
        os.environ['NATIVE4DGS_CODE_ROOT'] = str(entry.get('code_root', Path(entry['root']) / 'code/4d-gaussian-splatting'))
        sys.path.insert(0, str(package))
        # Avoid colliding with reconstruction_viewer/viewer.py in the worker.
        spec = importlib.util.spec_from_file_location('native4dgs_render_service', package / 'viewer.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from omegaconf import OmegaConf
        self.native = module
        self.state = module.load(entry['experiment'], Path(entry['root']) / entry['model'])
        self.pc = self.state['pc']
        self.pipe = SimpleNamespace(**OmegaConf.to_container(self.state['cfg'].PipelineParams))
        self.background = torch.tensor([1., 1., 1.] if self.state['cfg'].ModelParams.white_background else [0., 0., 0.], device='cuda')
        frames = self.state['transforms']['frames']
        index = 0
        camera, _, _ = module.make_camera(self.state, {'camera': index, 'width': 0, 'time': frames[index].get('time', 0)})
        self.reference = camera
        c2w = np.linalg.inv(camera.world_view_transform.cpu().numpy().T)
        xyz = self.pc.get_xyz.detach().cpu().numpy()[::max(1, self.pc.get_xyz.shape[0] // 2000)]
        w2c = np.linalg.inv(c2w)
        depths = (w2c[:3, :3] @ xyz.T).T[:, 2] + w2c[2, 3]
        positive = depths[depths > 0]
        distance = float(np.median(positive)) if len(positive) else 1.0
        # Dense display timeline: test-view sampling is not the animation rate.
        lo, hi = map(float, self.state['cfg'].time_duration)
        timeline = np.linspace(lo, min(hi, max(f.get('time', 0) for f in frames)), 300).tolist()
        self.metadata = {
            'c2w': c2w.tolist(), 'fov': float(camera.FoVy),
            'resolution': [camera.image_width, camera.image_height],
            'times': timeline, 'initial_frame': 0,
            'look_at': (c2w[:3, 3] + c2w[:3, 2] * distance).tolist(),
            'gaussians': int(self.pc.get_xyz.shape[0]),
            'iteration': int(self.state['iteration']),
        }

    @torch.no_grad()
    def render(self, request):
        width, height = map(int, request['size'])
        w2c = np.asarray(request['w2c'], dtype=np.float32)
        fovy = float(request['fov'])
        fovx = 2 * math.atan(math.tan(fovy / 2) * width / height)
        fx = width / (2 * math.tan(fovx / 2))
        fy = height / (2 * math.tan(fovy / 2))
        cx = self.reference.cx * width / self.reference.image_width if self.reference.cx > 0 else width / 2
        cy = self.reference.cy * height / self.reference.image_height if self.reference.cy > 0 else height / 2
        cam = self.native.Camera(0, w2c[:3, :3].T, w2c[:3, 3], fovx, fovy,
                                None, None, 'shared_viewer', 0, timestamp=float(request['time']),
                                cx=cx, cy=cy, fl_x=fx, fl_y=fy,
                                resolution=(width, height), meta_only=True).cuda()
        result = self.native.render(cam, self.pc, self.pipe, self.background)['render']
        if not torch.isfinite(result).all():
            raise FloatingPointError('Non-finite Native 4DGS render')
        return (result.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
