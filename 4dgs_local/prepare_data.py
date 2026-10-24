"""Prepare official real captures and a calibrated, training-camera-only initializer."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parent


def extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        if not all((destination / name).resolve().is_relative_to(destination)
                   for name in handle.namelist()):
            raise ValueError('unsafe archive path')
        handle.extractall(destination)


def ply(path, xyz, rgb):
    dtype = [(name, 'f4') for name in ('x', 'y', 'z', 'nx', 'ny', 'nz')]
    dtype += [(name, 'u1') for name in ('red', 'green', 'blue')]
    points = np.zeros(len(xyz), dtype=dtype)
    for index, name in enumerate(('x', 'y', 'z')):
        points[name] = xyz[:, index]
    for index, name in enumerate(('red', 'green', 'blue')):
        points[name] = rgb[:, index]
    PlyData([PlyElement.describe(points, 'vertex')], text=False).write(path)


def coffee_cloud(path):
    videos = sorted(path.glob('cam*.mp4'))
    poses = np.load(path / 'poses_bounds.npy')[:, :-2].reshape(-1, 3, 5)
    assert len(videos) == len(poses)
    images, matrices, centers, features = {}, {}, {}, {}
    sift = cv2.SIFT_create(nfeatures=10000)
    for index in range(1, len(videos)):
        capture = cv2.VideoCapture(str(videos[index]))
        ok, image = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(videos[index])
        image = cv2.resize(image, (676, 507), interpolation=cv2.INTER_AREA)
        images[index] = image
        pose = poses[index]
        c2w = np.stack((pose[:, 1], pose[:, 0], -pose[:, 2]), axis=1)
        center = pose[:, 3]
        extrinsic = np.column_stack((c2w.T, -c2w.T @ center))
        focal = pose[2, 4] * 676 / pose[1, 4]
        intrinsics = np.array([[focal, 0, 338.], [0, focal, 253.5], [0, 0, 1.]])
        matrices[index] = intrinsics @ extrinsic
        centers[index] = center
        features[index] = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    pairs = set()
    for index in images:
        nearest = sorted((other for other in images if other != index),
                         key=lambda other: np.linalg.norm(centers[index] - centers[other]))[:4]
        pairs.update(tuple(sorted((index, other))) for other in nearest)
    matcher = cv2.BFMatcher()
    xyz_all, rgb_all = [], []
    for first, second in sorted(pairs):
        a, da = features[first]
        b, db = features[second]
        matches = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < .75 * n.distance]
        if not matches:
            continue
        ua = np.array([a[m.queryIdx].pt for m in matches], dtype=np.float64)
        ub = np.array([b[m.trainIdx].pt for m in matches], dtype=np.float64)
        h = cv2.triangulatePoints(matrices[first], matrices[second], ua.T, ub.T)
        xyz = (h[:3] / h[3:]).T
        errors, positive = [], []
        for camera, pixels in ((first, ua), (second, ub)):
            projected = np.column_stack((xyz, np.ones(len(xyz)))) @ matrices[camera].T
            errors.append(np.linalg.norm(projected[:, :2] / projected[:, 2:3] - pixels, axis=1))
            positive.append(projected[:, 2] > 0)
        rays_a, rays_b = xyz - centers[first], xyz - centers[second]
        cosine = np.sum(rays_a * rays_b, axis=1) / np.maximum(
            np.linalg.norm(rays_a, axis=1) * np.linalg.norm(rays_b, axis=1), 1e-9)
        keep = (np.isfinite(xyz).all(1) & positive[0] & positive[1]
                & (errors[0] < 2.) & (errors[1] < 2.) & (cosine < np.cos(np.deg2rad(.5))))
        pixels = ua[keep].round().astype(int)
        xyz_all.append(xyz[keep])
        rgb_all.append(images[first][pixels[:, 1], pixels[:, 0], ::-1])
    xyz, rgb = np.concatenate(xyz_all), np.concatenate(rgb_all)
    if len(xyz) < 1000:
        raise RuntimeError(f'only {len(xyz)} calibrated matches')
    lower, upper = np.quantile(xyz, [.005, .995], axis=0)
    keep = ((xyz >= lower) & (xyz <= upper)).all(1)
    xyz, rgb = xyz[keep], rgb[keep]
    _, unique = np.unique(np.round(xyz, 3), axis=0, return_index=True)
    unique = np.random.default_rng(0).permutation(unique)[:40000]
    xyz, rgb = xyz[unique], rgb[unique]
    ply(path / 'points3D_downsample2.ply', xyz, rgb)
    metadata = dict(initializer='calibrated_SIFT_triangulation_at_frame_0',
                    cameras_used=[p.name for p in videos[1:]], held_out_camera=videos[0].name,
                    points=len(xyz), matching_resolution=[676, 507],
                    max_reprojection_error_pixels=2., bounds=[xyz.min(0).tolist(), xyz.max(0).tolist()])
    (path / 'initialization.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', choices=('printer', 'coffee'))
    args = parser.parse_args()
    if args.scene == 'printer':
        extract(ROOT / 'data/vrig_3dprinter.zip', ROOT / 'data/raw')
        sources = list((ROOT / 'data/raw').glob('*3dprinter*/dataset.json'))
        if len(sources) != 1:
            raise ValueError(sources)
        path = sources[0].parent
        source = ROOT / 'data/author_clouds/hypernerf/virg/virg-3dprinter/points3D_downsample2.ply'
        shutil.copy2(source, path / 'points3D_downsample2.ply')
        print(path, flush=True)
    else:
        extract(ROOT / 'data/coffee_martini.zip', ROOT / 'data/raw')
        path = ROOT / 'data/raw/coffee_martini'
        coffee_cloud(path)


if __name__ == '__main__':
    main()
