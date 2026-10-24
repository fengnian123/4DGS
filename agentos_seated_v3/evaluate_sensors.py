import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw
from checkpoint_io import load_checkpoint

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'shape-of-motion'))
from flow3d.scene_model import SceneModel


def summarize(values):
    if not values:
        return None
    values = np.concatenate(values)
    return {'returns': len(values), 'median_error_m': float(np.median(values)),
            'mean_error_m': float(np.mean(values)), 'p90_error_m': float(np.quantile(values, .9)),
            'fraction_under_5cm': float((values < .05).mean())}


@torch.no_grad()
def main():
    output = ROOT / 'results/office'
    output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'withheld LiDAR returns from the same sequence, not independent sensor calibration or a navigation evaluation',
              'split': 'original return IDs held out consistently across all cameras; no held-out return used in input completion or direct loss',
              'timing': 'returns within +/-25 ms of camera capture; robot motion compensated; human motion and physical exposure delay not independently known',
              'models': {}}
    sensor = np.load(ROOT / 'agentos_office/sensor_constraints.npz')
    norm = torch.load(ROOT / 'agentos_office/flow3d_preprocessed/scene_norm_dict.pth', weights_only=True)
    scale = float(norm['scale'])
    depth_maps = {}
    right_w2cs = torch.from_numpy(np.linalg.inv(sensor['right_c2w'])).float() @ torch.linalg.inv(norm['transfm']).float()
    right_w2cs[:, :3, 3] /= scale
    for name, directory in (('v2', ROOT.parent / 'agentos_seated_v2'), ('v3', ROOT)):
        checkpoint = load_checkpoint(directory / 'runs/office/checkpoints/last.ckpt')
        model = SceneModel.init_from_state_dict(checkpoint['model']).cuda().eval()
        collected = {region: [] for region in ('all', 'foreground', 'prop', 'right_auxiliary')}
        records = []
        depth_maps[name] = []
        for frame in range(80):
            rendered = model.render(frame, model.w2cs[frame:frame+1], model.Ks[frame:frame+1], (768, 512), return_depth=True)
            depth = rendered['depth'][0, :, :, 0].cpu().numpy() * scale
            depth_maps[name].append(depth)
            heldout = np.load(ROOT / f'sensors/front/{frame:05d}.npz')['heldout_depth']
            mask = cv2.imread(str(ROOT / f'agentos_office/masks/{frame:05d}.png'), 0) > 0
            prop = cv2.imread(str(ROOT / f'prop_masks/{frame:05d}.png'), 0) > 0
            row = {'frame': frame}
            for region, selected in (('all', heldout > 0), ('foreground', (heldout > 0) & mask), ('prop', (heldout > 0) & prop)):
                if selected.any() and (region != 'prop' or prop.mean() >= .01):
                    errors = np.abs(depth[selected] - heldout[selected])
                    collected[region].append(errors)
                    row[region] = {'returns': len(errors), 'median_error_m': float(np.median(errors))}
            auxiliary = model.render(frame, right_w2cs[frame:frame+1].cuda(), model.Ks[frame:frame+1], (768, 512), return_depth=True)
            aux_depth = auxiliary['depth'][0, :, :, 0].cpu().numpy() * scale
            aux_heldout = np.load(ROOT / f'sensors/right/{frame:05d}.npz')['heldout_depth']
            valid = aux_heldout > .3
            if valid.any():
                errors = np.abs(aux_depth[valid] - aux_heldout[valid])
                collected['right_auxiliary'].append(errors)
                row['right_auxiliary'] = {'returns': len(errors), 'median_error_m': float(np.median(errors))}
            records.append(row)
        report['models'][name] = {'checkpoint_step': int(checkpoint['global_step']),
                                  **{region: summarize(values) for region, values in collected.items()},
                                  'per_frame': records}
        del model
        torch.cuda.empty_cache()
    (output / 'sensor_metrics.json').write_text(json.dumps(report, indent=2))
    def panel(depth, label, sparse=False):
        color = cv2.cvtColor(cv2.applyColorMap((np.clip((depth - .12) / (3 - .12), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        if sparse:
            color[depth <= 0] = 0
        canvas = Image.new('RGB', (768, 542), '#17202a')
        canvas.paste(Image.fromarray(color), (0, 30))
        ImageDraw.Draw(canvas).text((8, 8), label, fill='white')
        return np.asarray(canvas)
    with iio.get_writer(output / 'depth_before_after.mp4', fps=10, codec='libx264', quality=8, macro_block_size=2) as writer:
        for frame in range(80):
            measurements = np.load(ROOT / f'sensors/front/{frame:05d}.npz')
            measured = np.maximum(measurements['train_depth'], measurements['heldout_depth'])
            row = np.concatenate((panel(measured, f'MEASURED sparse LiDAR | t={frame/10:.1f}s | BLACK=UNOBSERVED', True),
                                  panel(depth_maps['v2'][frame], 'V2 MODEL camera-Z depth | color range 0.12-3.0 m'),
                                  panel(depth_maps['v3'][frame], 'V3 MODEL camera-Z depth | color range 0.12-3.0 m')), axis=1)
            writer.append_data(row)
            if frame in (0, 30, 40, 48, 79):
                iio.imwrite(output / f'depth_before_after_{frame:05d}.png', row)
    print(json.dumps({name: {key: value for key, value in result.items() if key != 'per_frame'} for name, result in report['models'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
