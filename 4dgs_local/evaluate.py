"""Render full official held-out splits and independently control view/time."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '4DGaussians'))

import imageio.v2 as imageio
import mmcv
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

from arguments import ModelParams, ModelHiddenParams, PipelineParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.loss_utils import ssim
from utils.params_utils import merge_hparams


def rgb(tensor):
    return (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')


def caption(array, text):
    image = Image.fromarray(array)
    result = Image.new('RGB', (image.width, image.height + 40), '#111827')
    result.paste(image, (0, 40))
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    draw.text((10, 10), text, fill='white', font=font)
    return np.array(result)


def writer(path):
    return imageio.get_writer(path, fps=30, codec='libx264', quality=8,
                             macro_block_size=1, ffmpeg_params=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])


def camera_pose(camera):
    return np.linalg.inv(camera.world_view_transform.numpy().T)


def virtual_camera(template, c2w, time):
    w2c = np.linalg.inv(c2w)
    return Camera(0, w2c[:3, :3].T, w2c[:3, 3], template.FoVx, template.FoVy,
                  template.original_image, None, 'virtual', 0, time=float(time))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser)
    hidden = ModelHiddenParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('--scene', choices=['printer', 'printer_calibrated', 'coffee'], required=True)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--virtual_frames', type=int, default=180)
    args = parser.parse_args()
    name = args.scene
    sources = dict(printer=ROOT / 'data/raw/vrig-3dprinter',
                   printer_calibrated=ROOT / 'data/derived/printer_calibrated',
                   coffee=ROOT / 'data/raw/coffee_martini')
    args.source_path = str(sources[name])
    args.model_path = str(ROOT / 'runs' / ('printer_calibrated_stable' if name == 'printer_calibrated' else name))
    args = merge_hparams(args, mmcv.Config.fromfile(ROOT / 'configs' / f'{name}.py'))
    gaussians = GaussianModel(args.sh_degree, hidden.extract(args))
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    args.iteration = scene.loaded_iter
    pipe = pipeline.extract(args)
    background = torch.tensor([1., 1., 1.], device='cuda')
    destination = ROOT / 'results' / f'{name}_{args.iteration}'
    for folder in ['renders', 'ground_truth']:
        (destination / folder).mkdir(parents=True, exist_ok=True)
    test = scene.getTestCameras()
    chosen = set(np.linspace(0, len(test) - 1, 3).round().astype(int))
    rows, frame_metrics = [], []
    previous_time = -1
    with writer(destination / 'comparison.mp4') as movie:
        for index in range(len(test)):
            camera = test[index]
            if camera.time < previous_time:
                raise ValueError('test frames not in temporal order')
            previous_time = camera.time
            output = render(camera, gaussians, pipe, background, cam_type=scene.dataset_type)['render'].clamp(0, 1)
            gt = camera.original_image.to('cuda').clamp(0, 1)
            mse = torch.mean((output - gt) ** 2)
            psnr = float(-10 * torch.log10(mse.clamp_min(1e-12)))
            score = float(ssim(output.unsqueeze(0), gt.unsqueeze(0)))
            if scene.dataset_type == 'dynerf':
                source_frame = str(Path(test.dataset.image_paths[index]).relative_to(args.source_path))
            else:
                source_frame = test.dataset.all_img_origin[test.dataset.i_test[index]]
            frame_metrics.append(dict(frame=index, source_frame=source_frame,
                                      normalized_time=float(camera.time), psnr=psnr, ssim=score))
            output_rgb, gt_rgb = rgb(output), rgb(gt)
            Image.fromarray(output_rgb).save(destination / 'renders' / f'{index:05d}.png')
            Image.fromarray(gt_rgb).save(destination / 'ground_truth' / f'{index:05d}.png')
            pair = np.concatenate([caption(gt_rgb, f'REAL held-out | t={camera.time:.3f}'),
                                   caption(output_rgb, f'LOCAL 4DGS | PSNR {psnr:.2f} dB')], axis=1)
            movie.append_data(pair)
            if index == len(test) // 2:
                Image.fromarray(pair).save(destination / 'preview.jpg', quality=95)
            if index in chosen:
                rows.append(Image.fromarray(pair))
            if index % 50 == 0:
                print(f'{name}: held-out frame {index}/{len(test)}, PSNR={psnr:.2f}', flush=True)
    width = min(rows[0].width, 1200)
    resized = [im.resize((width, round(im.height * width / im.width))) for im in rows]
    contact = Image.new('RGB', (width, sum(im.height for im in resized)))
    y = 0
    for im in resized:
        contact.paste(im, (0, y))
        y += im.height
    contact.save(destination / 'comparison.jpg', quality=95)

    template = test[len(test) // 2]
    c2w = camera_pose(template)
    start, end = float(test[0].time), float(test[len(test) - 1].time)
    with writer(destination / 'fixed_view_time.mp4') as movie:
        for time in np.linspace(start, end, args.virtual_frames):
            camera = virtual_camera(template, c2w, time)
            image = render(camera, gaussians, pipe, background, cam_type=scene.dataset_type)['render']
            movie.append_data(caption(rgb(image), f'LOCAL 4DGS | fixed view | t={time:.3f}'))

    # Stay close to observed views: interpolate the two nearest training poses.
    train = scene.getTrainCameras()
    if scene.dataset_type == 'dynerf':
        indices = range(0, len(train), 300)
    else:
        indices = range(len(train))
    nearest = sorted(indices, key=lambda i: np.linalg.norm(camera_pose(train[i])[:3, 3] - c2w[:3, 3]))[:2]
    poses = [camera_pose(train[i]) for i in nearest]
    rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([p[:3, :3] for p in poses])))
    time = (start + end) / 2
    with writer(destination / 'fixed_time_view.mp4') as movie:
        for phase in np.linspace(0, 2 * np.pi, args.virtual_frames):
            alpha = (1 - np.cos(phase)) / 2
            pose = np.eye(4)
            pose[:3, :3] = rotations(alpha).as_matrix()
            pose[:3, 3] = (1 - alpha) * poses[0][:3, 3] + alpha * poses[1][:3, 3]
            camera = virtual_camera(template, pose, time)
            image = render(camera, gaussians, pipe, background, cam_type=scene.dataset_type)['render']
            movie.append_data(caption(rgb(image), f'LOCAL 4DGS | changing view | fixed t={time:.3f}'))
    metrics = dict(scene=name, iteration=args.iteration, coarse_iterations=3000,
                   test_frames=len(test), resolution=[template.image_width, template.image_height],
                   psnr=float(np.mean([m['psnr'] for m in frame_metrics])),
                   ssim=float(np.mean([m['ssim'] for m in frame_metrics])),
                   metric_scope='full RGB frame, float [0,1], unmasked, mean per-frame PSNR/SSIM',
                   held_out='cam00.mp4 (excluded from training and initialization)' if name == 'coffee' else 'official val_ids',
                   preprocessing={'printer_calibrated': 'undistorted centered pinhole crop, 536x720',
                                  'printer': 'official loader, original release RGB/2x',
                                  'coffee': 'official loader, resized video frames, 676x507'}[name],
                   gaussians=int(gaussians.get_xyz.shape[0]), frames=frame_metrics,
                   virtual_view_train_indices=nearest,
                   virtual_videos='synthesized view/time queries, no ground truth at interpolated poses')
    temporary = destination / 'metrics.json.tmp'
    temporary.write_text(json.dumps(metrics, indent=2))
    temporary.replace(destination / 'metrics.json')
    from build_gallery import build
    build()
    print(json.dumps({k: v for k, v in metrics.items() if k != 'frames'}), flush=True)


if __name__ == '__main__':
    main()
