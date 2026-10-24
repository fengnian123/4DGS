import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as iio
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from checkpoint_io import load_checkpoint

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'shape-of-motion'))
from flow3d.metrics import compute_psnr, mSSIM
from flow3d.scene_model import SceneModel


def panel(array, label):
    canvas = Image.new('RGB', (array.shape[1], array.shape[0] + 30), '#17202a')
    canvas.paste(Image.fromarray(array), (0, 30))
    ImageDraw.Draw(canvas).text((8, 8), label, fill='white')
    return np.asarray(canvas)


def to_rgb(value):
    if not torch.isfinite(value).all():
        raise FloatingPointError('Non-finite rendering')
    return (value.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)


def load_model(root):
    checkpoint = load_checkpoint(root / 'runs/office/checkpoints/last.ckpt')
    return SceneModel.init_from_state_dict(checkpoint['model']).cuda().eval(), int(checkpoint['global_step'])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    old_root = ROOT.parent / 'agentos_seated_v2'
    old, old_step = load_model(old_root)
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        new = SceneModel.init_from_state_dict(checkpoint['model']).cuda().eval()
        new_step = int(checkpoint['global_step'])
    else:
        new, new_step = load_model(ROOT)
    width, height = Image.open(ROOT / 'clip/front/00000.png').size
    out = args.output or ROOT / ('preview_comparison' if args.preview else 'results/office')
    out.mkdir(parents=True, exist_ok=True)
    frames = (0, 10, 30, 40, 48, 60, 79) if args.preview else range(new.num_frames)
    # Both camera refinements are anchored to the original physical camera at frame 40.
    # Their normalized coordinate systems differ, so use each model's own normalized pose.
    summary = {'scope': 'same input RGB and full image; training-view fit, not held-out generalization',
               'baseline_directory': str(old_root), 'baseline_step': old_step,
               'optimized_step': new_step, 'optimized_seed_step': 0, 'per_frame': []}
    ssims = [mSSIM().cuda(), mSSIM().cuda()]
    fixed_video = iio.get_writer(out / 'fixed_reference_before_after.mp4', fps=10, codec='libx264', quality=8, macro_block_size=2)
    moving_video = iio.get_writer(out / 'real_before_after.mp4', fps=10, codec='libx264', quality=8, macro_block_size=2)
    try:
        for t in frames:
            gt = torch.from_numpy(np.array(Image.open(ROOT / f'clip/front/{t:05d}.png'))).float().cuda()[None] / 255
            foreground = cv2.imread(str(ROOT / f'agentos_office/masks/{t:05d}.png'), 0) > 0
            background = ~cv2.dilate(foreground.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            background = torch.from_numpy(background).cuda()
            capture, fixed, record = [], [], {'frame': int(t)}
            for name, model, ssim in zip(('baseline', 'optimized'), (old, new), ssims):
                rendered = model.render(t, model.w2cs[t:t+1], model.Ks[t:t+1], (width, height), return_mask=True)
                prediction = rendered['img']
                record[name + '_psnr'] = compute_psnr(prediction, gt)
                foreground_alpha = rendered['mask'][0, :, :, 0][background]
                record[name + '_background_foreground_alpha'] = foreground_alpha.mean().item()
                record[name + '_background_false_positive_fraction'] = (foreground_alpha > .2).float().mean().item()
                ssim.update(prediction, gt)
                capture.append(panel(to_rgb(prediction[0]), f'{name.upper()} | capture camera t={t/10:.1f}s'))
                frozen = model.render(t, model.w2cs[40:41], model.Ks[40:41], (width, height))['img'][0]
                fixed.append(panel(to_rgb(frozen), f'{name.upper()} | same fixed physical camera | t={t/10:.1f}s | no GT'))
            summary['per_frame'].append(record)
            pair = np.concatenate(fixed, axis=1)
            triplet = np.concatenate([panel(to_rgb(gt[0]), f'REAL INPUT t={t/10:.1f}s'), *capture], axis=1)
            fixed_video.append_data(pair)
            moving_video.append_data(triplet)
            if t in (0, 10, 30, 40, 48, 60, 79):
                iio.imwrite(out / f'fixed_before_after_{t:05d}.png', pair)
                iio.imwrite(out / f'real_before_after_{t:05d}.png', triplet)
    finally:
        fixed_video.close()
        moving_video.close()
    for index, name in enumerate(('baseline', 'optimized')):
        summary[name + '_ssim'] = ssims[index].compute().item()
        summary[name + '_psnr'] = float(np.mean([r[name + '_psnr'] for r in summary['per_frame']]))
        for metric in ('background_foreground_alpha', 'background_false_positive_fraction'):
            summary[name + '_' + metric] = float(np.mean([r[name + '_' + metric] for r in summary['per_frame']]))
        for phase, lower, upper in (('early', 0, 16), ('action', 16, 56), ('late', 56, 80)):
            values = [r[name + '_psnr'] for r in summary['per_frame'] if lower <= r['frame'] < upper]
            summary[name + '_' + phase + '_psnr'] = float(np.mean(values))
    (out / 'before_after_metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_frame'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
