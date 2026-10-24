import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "shape-of-motion"))
from flow3d.data.nvidia_dataset import NvidiaDataset
from flow3d.metrics import compute_psnr, mLPIPS, mPSNR, mSSIM
from flow3d.scene_model import SceneModel


def caption(frame, text):
    canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 28), "#17202a")
    canvas.paste(Image.fromarray(frame), (0, 28))
    ImageDraw.Draw(canvas).text((8, 8), text, fill="white")
    return np.asarray(canvas)


def uint8(tensor):
    return (tensor.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "runs/balloon1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/extracted/Balloon1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out = args.output or args.run / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rgb").mkdir(exist_ok=True)
    # This checkpoint is generated locally by run_training.py and contains NumPy optimizer scalars.
    checkpoint = torch.load(args.run / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
    model = SceneModel.init_from_state_dict(checkpoint["model"]).cuda().eval()
    dataset = NvidiaDataset(str(args.data), split="val", load_from_cache=True)
    metrics = {
        group: {"psnr": mPSNR().cuda(), "ssim": mSSIM().cuda(), "lpips": mLPIPS().cuda()}
        for group in ("official_covisible", "foreground", "background")
    }
    writers, full_scores, records, contact_rows, templates = {}, {}, [], [], {}
    motion_rows = []
    try:
        for index in range(min(len(dataset), args.limit or len(dataset))):
            batch = dataset[index]
            name = batch["frame_names"]
            view = name.split("_")[0]
            t = int(batch["ts"])
            gt = batch["imgs"].cuda()[None]
            w2c, K = batch["w2cs"].cuda()[None], batch["Ks"].cuda()[None]
            H, W = gt.shape[1:3]
            prediction = model.render(t, w2c, K, (W, H))["img"]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"Non-finite render: {name}")
            valid = (batch["valid_masks"] * batch["covisible_masks"]).cuda()[None]
            fg = batch["masks"].cuda()[None]
            for group, mask in (("official_covisible", valid), ("foreground", valid * fg), ("background", valid * (1 - fg))):
                for metric in metrics[group].values():
                    metric.update(prediction, gt, mask)
            score = compute_psnr(prediction, gt)
            full_scores.setdefault(view, []).append(score)
            records.append({"frame": name, "time": t, "full_rgb_psnr": score, "covisible_fraction": valid.mean().item()})
            pred_rgb, gt_rgb = uint8(prediction[0]), uint8(gt[0])
            iio.imwrite(out / "rgb" / f"{name}.png", pred_rgb)
            pair = np.concatenate((gt_rgb, pred_rgb), axis=1)
            pair = caption(pair, f"Held-out view {view}, t={t:03d} | LEFT: real / RIGHT: Shape of Motion | full PSNR {score:.2f}")
            if view not in writers:
                writers[view] = iio.get_writer(out / f"comparison_view{view}.mp4", fps=15, codec="libx264", quality=8, macro_block_size=2)
            writers[view].append_data(pair)
            if view == "1" and t in (0, model.num_frames // 2, model.num_frames - 1):
                motion_rows.append(pair)
            if t == model.num_frames // 2:
                contact_rows.append(pair)
                templates[view] = (w2c, K)
                iio.imwrite(out / f"comparison_view{view}_t{t}.png", pair)
            if (index + 1) % 48 == 0:
                print(f"Evaluated {index + 1}/{len(dataset)} held-out frames", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    results = {
        "checkpoint_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "validation_images": len(records),
        "resolution": [W, H],
        "num_foreground_gaussians": model.num_fg_gaussians,
        "num_background_gaussians": model.num_bg_gaussians,
        "num_motion_bases": model.num_motion_bases,
        "metrics": {group: {key: metric.compute().item() for key, metric in values.items()} for group, values in metrics.items()},
        "full_rgb_mean_psnr": float(np.mean([row["full_rgb_psnr"] for row in records])),
        "full_rgb_psnr_by_view": {view: float(np.mean(values)) for view, values in full_scores.items()},
        "per_frame": records,
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "per_frame"}, indent=2), flush=True)
    if contact_rows:
        iio.imwrite(out / "comparison.png", np.concatenate(contact_rows, axis=0))
    if motion_rows:
        iio.imwrite(out / "motion_comparison.png", np.concatenate(motion_rows, axis=0))
    if args.limit:
        return
    views = sorted(templates)
    t = model.num_frames // 2
    w2c_a, K = templates[views[0]]
    w2c_b, _ = templates[views[-1]]
    c2w_a, c2w_b = (torch.linalg.inv(w).cpu().numpy()[0] for w in (w2c_a, w2c_b))
    rotations = Slerp([0, 1], Rotation.from_matrix(np.stack((c2w_a[:3, :3], c2w_b[:3, :3]))))
    with iio.get_writer(out / "fixed_time_camera_sweep.mp4", fps=24, codec="libx264", quality=8, macro_block_size=2) as writer:
        for index, alpha in enumerate(np.concatenate((np.linspace(0, 1, 72), np.linspace(1, 0, 72)))):
            c2w = np.eye(4)
            c2w[:3, :3] = rotations(alpha).as_matrix()
            c2w[:3, 3] = (1 - alpha) * c2w_a[:3, 3] + alpha * c2w_b[:3, 3]
            w2c = torch.from_numpy(np.linalg.inv(c2w)).float().cuda()[None]
            prediction = model.render(t, w2c, K, (W, H))["img"][0]
            frame = caption(uint8(prediction), f"Fixed time t={t} | interpolated camera | no ground truth")
            writer.append_data(frame)
            if index == 0:
                iio.imwrite(out / "fixed_time_camera_sweep.png", frame)
    export_ply(model, t, out / f"gaussians_t{t:03d}.ply")
    candidates = torch.where(model.fg.get_opacities().reshape(-1) > .5)[0]
    indices = candidates[::max(1, len(candidates) // 512)][:512]
    times = torch.arange(model.num_frames, device="cuda")
    tracks, _ = model.compute_poses_fg(times, inds=indices)
    np.savez_compressed(
        out / "tracks_3d.npz",
        gaussian_ids=indices.cpu().numpy(),
        frame_ids=times.cpu().numpy(),
        xyz_normalized=tracks.cpu().numpy(),
        colors=model.fg.get_colors()[indices].cpu().numpy(),
        scene_scale=np.asarray(dataset.scene_norm_dict["scale"]),
        transform_before_scale=np.asarray(dataset.scene_norm_dict["transfm"]),
    )


def export_ply(model, t, path):
    from plyfile import PlyData, PlyElement
    means, quats = model.compute_poses_all(torch.tensor([t], device="cuda"))
    xyz = means[:, 0].cpu().numpy()
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    values = np.concatenate((xyz, np.zeros_like(xyz), (model.get_colors_all().cpu().numpy() - .5) / .28209479177387814, torch.logit(model.get_opacities_all().clamp(1e-6, 1 - 1e-6)).cpu().numpy().reshape(-1, 1), model.get_scales_all().log().cpu().numpy(), quats[:, 0].cpu().numpy()), axis=1)
    vertices = np.empty(len(xyz), dtype=[(name, "f4") for name in fields])
    for index, name in enumerate(fields):
        vertices[name] = values[:, index]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))
    print(f"Exported Gaussian snapshot: {path}", flush=True)


if __name__ == "__main__":
    main()
