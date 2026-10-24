import json
import sys
from pathlib import Path

import numpy as np
import torch


class AgentOS:
    def __init__(self, entry):
        root = Path(entry['root'])
        sys.path.insert(0, entry['code_root'])
        from flow3d.scene_model import SceneModel

        checkpoint = torch.load(root / entry['model'] / 'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
        self.model = SceneModel.init_from_state_dict(checkpoint['model']).cuda().eval()
        frame = self.model.num_frames // 2
        c2w = torch.linalg.inv(self.model.w2cs[frame]).cpu().numpy()
        width, height = json.loads((root / entry['results'] / 'metrics.json').read_text())['resolution']
        focal = float(self.model.Ks[frame, 1, 1])
        self.metadata = {
            'c2w': c2w.tolist(), 'fov': float(2 * np.arctan(height / (2 * focal))),
            'resolution': [width, height], 'times': list(range(self.model.num_frames)),
            'initial_frame': frame, 'look_at': (c2w[:3, 3] + c2w[:3, 2] * 2).tolist(),
            'gaussians': self.model.num_gaussians,
        }

    @torch.no_grad()
    def render(self, request):
        width, height = request['size']
        focal = .5 * height / np.tan(float(request['fov']) / 2)
        K = torch.tensor([[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]], device='cuda', dtype=torch.float32)
        w2c = torch.tensor(request['w2c'], device='cuda', dtype=torch.float32)
        output = self.model.render(int(request['time']), w2c[None], K[None], (width, height))['img'][0]
        if not torch.isfinite(output).all():
            raise FloatingPointError('Non-finite AgentOS reconstruction')
        return (output.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
