import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'agentos_seated'
V2 = ROOT.parent / 'agentos_seated_v2'


def calibrate_component_shape(sparse, region, output, confidence):
    measured = (sparse > 0) & region
    y, x = np.where(measured)
    if len(x) < 50 or region.sum() < 1000:
        return
    # Keep a monocular shape only after a local metric fit to measured returns.
    ids = np.arange(len(x))
    fitting, validation = ids[ids % 5 != 0], ids[ids % 5 == 0]
    inverse = 1 / output[y, x]
    actual = sparse[y, x]
    def predict(parameters, values):
        return 1 / np.maximum(parameters[0] * values + parameters[1], 1 / 15)
    result = least_squares(lambda p: np.log(predict(p, inverse[fitting]) / actual[fitting]),
                           [1., 0.], bounds=([.25, -4], [4., 4]), loss='soft_l1', f_scale=.1, max_nfev=60)
    before = np.median(np.abs(np.log(output[y[validation], x[validation]] / actual[validation])))
    after = np.median(np.abs(np.log(predict(result.x, inverse[validation]) / actual[validation])))
    candidate = predict(result.x, 1 / output[region])
    if after < before * .8 and ((candidate > .12) & (candidate < 12)).mean() > .98:
        output[region] = np.clip(candidate, .12, 12)
        confidence[region] = .25


def fit_plane(depth, region, K, rng):
    y, x = np.where((depth > 0) & region)
    if len(x) < 60:
        return None, None
    rays = np.c_[x, y, np.ones(len(x))] @ np.linalg.inv(K).T
    xyz = rays * depth[y, x, None]
    best = np.zeros(len(x), bool)
    for _ in range(100):
        a, b, c = xyz[rng.choice(len(xyz), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal) + 1e-8
        accepted = np.abs((xyz - a) @ normal) < .008
        if accepted.sum() > best.sum():
            best = accepted
    if best.mean() < .65:
        return None, {'samples': len(x), 'inlier_fraction': float(best.mean()), 'accepted': False}
    center = xyz[best].mean(0)
    _, _, vh = np.linalg.svd(xyz[best] - center, full_matrices=False)
    normal = vh[-1]
    yy, xx = np.indices(depth.shape)
    all_rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ np.linalg.inv(K).T
    denominator = all_rays @ normal
    plane = (center @ normal) / np.where(np.abs(denominator) > 1e-5, denominator, np.nan)
    # Do not extrapolate an observed planar patch to an unobserved whole object.
    hull = np.zeros(depth.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.c_[x[best], y[best]].astype(np.int32)), 1)
    hull = cv2.dilate(hull, np.ones((11, 11), np.uint8)) > 0
    valid = hull & region & np.isfinite(plane) & (plane > .12) & (plane < 2)
    output = np.where(valid, plane, 0).astype(np.float32)
    return output, {'samples': len(x), 'inlier_fraction': float(best.mean()), 'accepted': True,
                    'completed_pixels': int(valid.sum()), 'median_plane_residual_m': float(np.median(np.abs((xyz[best] - center) @ normal)))}


def complete_region(sparse, region, image, output, confidence):
    y, x = np.where((sparse > 0) & region)
    qy, qx = np.where(region)
    if len(x) < 8 or len(qx) == 0:
        return
    feature = np.c_[x, y, image[y, x] * 40]
    query = np.c_[qx, qy, image[qy, qx] * 40]
    distance, idx = cKDTree(feature).query(query, k=8, workers=4)
    spatial = np.linalg.norm(np.stack((x[idx] - qx[:, None], y[idx] - qy[:, None]), axis=-1), axis=-1)
    weights = np.exp(-distance ** 2 / (2 * 9 ** 2)) * (spatial < 20)
    inverse = 1 / sparse[y[idx], x[idx]]
    # Mixed near/far surfaces are left to the nearest return, not averaged across an edge.
    median = sparse[y[idx[:, 0]], x[idx[:, 0]]]
    consistent = np.abs(sparse[y[idx], x[idx]] - median[:, None]) < (.03 + median[:, None] * .05)
    weights *= consistent
    predicted = 1 / ((weights * inverse).sum(1) / weights.sum(1).clip(1e-8)).clip(1 / 15)
    good = (spatial[:, 0] < 16) & (weights.sum(1) > .8) & np.isfinite(predicted)
    output[qy[good], qx[good]] = predicted[good]
    confidence[qy[good], qx[good]] = .7


def main():
    cv2.setNumThreads(4)
    dataset = ROOT / 'agentos_office'
    dataset.mkdir(exist_ok=True)
    for name in ('images', 'masks', 'bootstapir'):
        link = dataset / name
        if not link.exists():
            link.symlink_to(V2 / 'agentos_office' / name, target_is_directory=True)
    for name, target in (('clip', BASE / 'clip'), ('prop_masks', V2 / 'prop_masks')):
        if not (ROOT / name).exists():
            (ROOT / name).symlink_to(target, target_is_directory=True)
    shutil.copyfile(V2 / 'prop_quality.json', ROOT / 'prop_quality.json')
    shutil.copyfile(V2 / 'camera_refinement.npz', ROOT / 'camera_refinement.npz')
    source = np.load(V2 / 'agentos_office/agentos_office.npz')
    depth = source['depths'].copy()
    confidence = np.full(depth.shape, .08, np.float32)
    measured, right, records = [], [], []
    K, rng = source['intrinsic'], np.random.default_rng(42)
    for frame in range(80):
        sparse = np.load(ROOT / f'sensors/front/{frame:05d}.npz')['train_depth'].copy()
        mask = cv2.imread(str(dataset / f'masks/{frame:05d}.png'), 0) > 0
        prop = cv2.imread(str(ROOT / f'prop_masks/{frame:05d}.png'), 0) > 0
        image = cv2.imread(str(dataset / f'images/{frame:05d}.png')).astype(np.float32) / 255
        # Exclude ambiguous foreground/background boundaries from geometry supervision.
        inside = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        outside = cv2.erode((~mask).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sparse[~(inside | outside)] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats((inside & ~prop).astype(np.uint8), 8)
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] >= 1000:
                calibrate_component_shape(sparse, labels == component, depth[frame], confidence[frame])
        calibrate_component_shape(sparse, prop, depth[frame], confidence[frame])
        complete_region(sparse, inside & ~prop, image, depth[frame], confidence[frame])
        complete_region(sparse, outside, image, depth[frame], confidence[frame])
        prop_inside = cv2.erode(prop.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        plane, plane_info = fit_plane(sparse, prop_inside, K, rng)
        if plane is not None:
            use = plane > 0
            depth[frame, use], confidence[frame, use] = plane[use], .85
        else:
            complete_region(sparse, prop_inside, image, depth[frame], confidence[frame])
        valid = sparse > 0
        depth[frame, valid], confidence[frame, valid] = sparse[valid], 1
        measured.append(sparse)
        aux = np.load(ROOT / f'sensors/right/{frame:05d}.npz')['train_depth'].copy()
        aux[aux < .3] = 0
        right.append(aux)
        heldout = np.load(ROOT / f'sensors/front/{frame:05d}.npz')['heldout_depth']
        row = {'frame': frame, 'plane': plane_info,
               'measured_pixels': int(valid.sum()), 'completed_fraction': float((confidence[frame] > .5).mean()),
               'auxiliary_right_pixels': int((aux > 0).sum())}
        for name, region in (('all', np.ones(mask.shape, bool)), ('foreground', mask), ('prop', prop)):
            selected = (heldout > 0) & region
            if selected.any():
                row[name + '_heldout_median_before_m'] = float(np.median(np.abs(source['depths'][frame][selected] - heldout[selected])))
                row[name + '_heldout_median_after_m'] = float(np.median(np.abs(depth[frame][selected] - heldout[selected])))
        records.append(row)
        if frame % 10 == 0:
            print('LiDAR input', row, flush=True)
    np.savez_compressed(dataset / 'agentos_office.npz', cam_c2w=source['cam_c2w'], intrinsic=K, depths=depth)
    refinement = np.load(ROOT / 'camera_refinement.npz')
    right_poses = np.load(BASE / 'clip/right/poses.npy')
    right_poses = refinement['refined_c2w'] @ np.linalg.inv(refinement['initial_c2w']) @ right_poses
    np.savez_compressed(dataset / 'sensor_constraints.npz', lidar_depths=np.array(measured), depth_confidence=confidence,
                        right_depths=np.array(right), right_c2w=right_poses.astype(np.float32))
    from filter_auxiliary import main as filter_auxiliary
    filter_auxiliary()
    cache = dataset / 'flow3d_preprocessed'
    cache.mkdir(exist_ok=True)
    shutil.copyfile(V2 / 'agentos_office/flow3d_preprocessed/scene_norm_dict.pth', cache / 'scene_norm_dict.pth')
    (ROOT / 'depth_completion.json').write_text(json.dumps(records, indent=2))
    print('Prepared measured-depth initialization; no model weights or new monocular inference used.', flush=True)


if __name__ == '__main__':
    main()
