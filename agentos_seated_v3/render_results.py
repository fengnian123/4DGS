import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw
from checkpoint_io import load_checkpoint

ROOT = Path(__file__).resolve().parent
SOM = ROOT.parent / 'som_local'
sys.path.insert(0, str(SOM))
sys.path.insert(0, str(ROOT / 'shape-of-motion'))
from flow3d.metrics import compute_psnr, mSSIM
from flow3d.scene_model import SceneModel
from evaluate import export_ply


def rgb(value):
    if not torch.isfinite(value).all():
        raise FloatingPointError('Non-finite rendered image')
    return (value.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)


def caption(array, text):
    canvas = Image.new('RGB', (array.shape[1], array.shape[0] + 28), '#17202a')
    canvas.paste(Image.fromarray(array), (0, 28))
    ImageDraw.Draw(canvas).text((8, 8), text, fill='white')
    return np.asarray(canvas)


@torch.no_grad()
def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    out = args.output or ROOT / 'results/office'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'rgb').mkdir(exist_ok=True)
    checkpoint = load_checkpoint(ROOT / 'runs/office/checkpoints/last.ckpt')
    model = SceneModel.init_from_state_dict(checkpoint['model']).cuda().eval()
    width, height = Image.open(ROOT / 'clip/front/00000.png').size
    if args.preview:
        frame = model.num_frames // 2
        gt = np.asarray(Image.open(ROOT / f'clip/front/{frame:05d}.png'))
        prediction = model.render(frame, model.w2cs[frame:frame+1], model.Ks[frame:frame+1], (width, height))['img'][0]
        pair = caption(np.concatenate((gt, rgb(prediction)), axis=1), f'Intermediate checkpoint step={checkpoint["global_step"]} | real / model')
        iio.imwrite(ROOT / 'training_preview.png', pair)
        print(f'Preview checkpoint step={checkpoint["global_step"]}', flush=True)
        return
    norm = torch.load(ROOT / 'agentos_office/flow3d_preprocessed/scene_norm_dict.pth', weights_only=True)
    scale = float(norm['scale'])
    ssims = mSSIM().cuda()
    records, rows = [], []
    with iio.get_writer(out / 'comparison.mp4', fps=10, codec='libx264', quality=8, macro_block_size=2) as writer:
        for t in range(model.num_frames):
            gt = torch.from_numpy(np.asarray(Image.open(ROOT / f'clip/front/{t:05d}.png')).copy()).float().cuda()[None] / 255
            prediction = model.render(t, model.w2cs[t:t+1], model.Ks[t:t+1], (width, height))['img']
            score = compute_psnr(prediction, gt)
            ssims.update(prediction, gt)
            mask = torch.from_numpy(np.asarray(Image.open(ROOT / f'agentos_office/masks/{t:05d}.png')).copy() > 0).cuda()[None]
            record = {'frame': t, 'psnr': score, 'foreground_psnr': compute_psnr(prediction, gt, mask), 'background_psnr': compute_psnr(prediction, gt, ~mask)}
            prop_path = ROOT / f'prop_masks/{t:05d}.png'
            if prop_path.exists():
                prop = torch.from_numpy(np.asarray(Image.open(prop_path)).copy() > 0).cuda()[None]
                record['prop_fraction'] = prop.float().mean().item()
                record['prop_psnr'] = compute_psnr(prediction, gt, prop) if record['prop_fraction'] >= .01 else None
            records.append(record)
            predicted_rgb = rgb(prediction[0])
            iio.imwrite(out / f'rgb/{t:05d}.png', predicted_rgb)
            pair = caption(np.concatenate((rgb(gt[0]), predicted_rgb), axis=1), f't={t/10:.1f}s | LEFT: real / RIGHT: Shape of Motion | TRAINING VIEW, PSNR {score:.2f}')
            writer.append_data(pair)
            if t in (0, model.num_frames // 2, model.num_frames - 1):
                rows.append(pair)
                iio.imwrite(out / f'comparison_{t:05d}.png', pair)
            elif prop_path.exists() and t == 30:
                iio.imwrite(out / f'comparison_{t:05d}.png', pair)
    iio.imwrite(out / 'comparison.png', np.concatenate(rows, axis=0))
    frame = model.num_frames // 2
    reference = torch.linalg.inv(model.w2cs[frame]).clone()
    views = []
    with iio.get_writer(out / 'camera_sweep.mp4', fps=20, codec='libx264', quality=8, macro_block_size=2) as writer:
        for offset in np.concatenate((np.linspace(0, -.1, 30), np.linspace(-.1, .1, 60), np.linspace(.1, 0, 30))):
            pose = reference.clone()
            pose[:3, 3] += reference[:3, 0] * float(offset / scale)
            predicted = model.render(frame, torch.linalg.inv(pose)[None], model.Ks[frame:frame+1], (width, height))['img'][0]
            writer.append_data(caption(rgb(predicted), f'Fixed t={frame/10:.1f}s, virtual lateral offset {offset:+.3f}m | NO GROUND TRUTH'))
        for offset in (-.1, 0., .1):
            pose = reference.clone()
            pose[:3, 3] += reference[:3, 0] * float(offset / scale)
            predicted = model.render(frame, torch.linalg.inv(pose)[None], model.Ks[frame:frame+1], (width, height))['img'][0]
            views.append(caption(rgb(predicted), f'Virtual lateral offset {offset:+.2f}m | no ground truth'))
    iio.imwrite(out / 'camera_sweep.png', np.concatenate(views, axis=1))
    independent = []
    # Independent physical camera: retain its invalid FOV as black rather than fabricating pixels.
    poses = np.load(ROOT / 'clip/left/poses.npy')
    refinement_path = ROOT / 'camera_refinement.npz'
    if refinement_path.exists():
        refinement = np.load(refinement_path)
        poses = refinement['refined_c2w'] @ np.linalg.inv(refinement['initial_c2w']) @ poses
    valid = torch.from_numpy(np.asarray(Image.open(ROOT / 'clip/left/valid_mask.png')).copy() > 0).cuda()[None]
    independent_rows = []
    for t in (0, frame, model.num_frames - 1):
        w2c = torch.from_numpy(np.linalg.inv(poses[t])).float()
        w2c = w2c @ torch.linalg.inv(norm['transfm']).float()
        w2c[:3, 3] /= scale
        prediction = model.render(t, w2c.cuda()[None], model.Ks[t:t+1], (width, height))['img']
        gt = torch.from_numpy(np.asarray(Image.open(ROOT / f'clip/left/{t:05d}.png')).copy()).float().cuda()[None] / 255
        score = compute_psnr(prediction, gt, valid)
        independent.append({'frame': t, 'valid_fraction': valid.float().mean().item(), 'valid_fov_psnr': score})
        pred_rgb = rgb(prediction[0])
        pred_rgb[~valid[0].cpu().numpy()] = 0
        independent_rows.append(caption(np.concatenate((rgb(gt[0]), pred_rgb), axis=1), f'LEFT camera NOT USED IN TRAINING | real / model | VALID FOV ONLY PSNR {score:.2f}'))
    iio.imwrite(out / 'independent_left_camera.png', np.concatenate(independent_rows, axis=0))
    export_ply(model, frame, out / f'gaussians_t{frame:03d}.ply')
    candidates = torch.where(model.fg.get_opacities().reshape(-1) > .5)[0]
    ids = candidates[::max(1, len(candidates) // 512)][:512]
    times = torch.arange(model.num_frames, device='cuda')
    tracks, _ = model.compute_poses_fg(times, inds=ids)
    np.savez_compressed(out / 'tracks_3d.npz', gaussian_ids=ids.cpu().numpy(), frame_ids=times.cpu().numpy(), xyz_normalized=tracks.cpu().numpy(), colors=model.fg.get_colors()[ids].cpu().numpy(), scene_scale=np.asarray(scale), transform_before_scale=np.asarray(norm['transfm']))
    metrics = {
        'scope': 'training_view_reconstruction; not a held-out navigation evaluation',
        'psnr': float(np.mean([r['psnr'] for r in records])), 'ssim': ssims.compute().item(),
        'foreground_psnr': float(np.mean([r['foreground_psnr'] for r in records])),
        'background_psnr': float(np.mean([r['background_psnr'] for r in records])),
        'frames': model.num_frames, 'fps': 10, 'resolution': [width, height],
        'checkpoint_step': int(checkpoint['global_step']), 'checkpoint_epoch': int(checkpoint['epoch']),
        'foreground_gaussians': model.num_fg_gaussians, 'background_gaussians': model.num_bg_gaussians,
        'motion_bases': model.num_motion_bases, 'independent_left_camera_diagnostic': independent,
        'geometry_status': 'body-aligned pointcloud convention supported by static geometry/LIO/IMU checks; camera-LiDAR track reprojection has a large outlier tail; physical exposure delay and upstream deskew remain independently unvalidated',
        'preprocessing': 'fresh model initialization; per-return LiDAR timestamps within +/-25 ms, LIO rigid deskew; measured metric depth with local sensor-supported completion; component-wise metric fit of inherited monocular shape where supported; low-confidence prior in remaining holes; direct LiDAR loss and right-camera constraints on measured pixels; SAM2 masks and Bootstapir tracks reused',
        'initialization_checkpoint_step': 0,
        'per_frame': records,
    }
    prop_scores = [record['prop_psnr'] for record in records if record.get('prop_psnr') is not None]
    if prop_scores:
        metrics['prop_psnr'] = float(np.mean(prop_scores))
        metrics['prop_scored_frames'] = len(prop_scores)
        metrics['prop_scope'] = 'SAM2 prop-mask pixels in training frames with mask area >= 1%; not independent geometry accuracy'
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != 'per_frame'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
