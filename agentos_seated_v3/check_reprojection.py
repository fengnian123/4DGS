import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / 'agentos_seated_v2'


def main():
    bundle = np.load(V2 / 'camera_refinement.npz')
    frames, ids, observations = bundle['frame_ids'], bundle['point_ids'], bundle['observations']
    K = np.load(V2 / 'agentos_office/agentos_office.npz')['intrinsic']
    camera_sets = {'original_lio': bundle['initial_c2w'], 'v2_refined': bundle['refined_c2w']}
    # Entire held-out image feature tracks; no camera fitting is performed here.
    selected = bundle['heldout_points'][ids]
    frames, ids, observations = frames[selected], ids[selected], observations[selected]
    trees, depths = {}, {}
    for frame in range(80):
        sparse = np.load(ROOT / f'sensors/front/{frame:05d}.npz')
        depth = np.maximum(sparse['train_depth'], sparse['heldout_depth'])
        y, x = np.where(depth > 0)
        trees[frame] = (cKDTree(np.c_[x, y]), x, y)
        depths[frame] = depth
    records = []
    errors = {name: [] for name in camera_sets}
    for identity in np.unique(ids):
        indexes = np.flatnonzero(ids == identity)
        anchors = []
        for index in indexes:
            frame, uv = int(frames[index]), observations[index]
            tree, x, y = trees[frame]
            distance, nearest = tree.query(uv)
            if distance <= 1.5:
                anchors.append((distance + abs(frame - 40) * .002, index, depths[frame][y[nearest], x[nearest]]))
        if not anchors:
            continue
        _, anchor, z = min(anchors)
        ray = np.linalg.inv(K) @ np.r_[observations[anchor], 1]
        row = {'track': int(identity), 'anchor_frame': int(frames[anchor]), 'measured_depth_m': float(z)}
        for name, poses in camera_sets.items():
            world = poses[frames[anchor], :3, :3] @ (ray * z) + poses[frames[anchor], :3, 3]
            target = np.linalg.inv(poses[frames[indexes]])
            xyz = np.einsum('nij,j->ni', target[:, :3, :3], world) + target[:, :3, 3]
            projected = xyz @ K.T
            pixels = projected[:, :2] / projected[:, 2:]
            valid = (xyz[:, 2] > 0) & (frames[indexes] != frames[anchor])
            error = np.linalg.norm(pixels - observations[indexes], axis=1)[valid]
            if len(error):
                errors[name].extend(error.tolist())
                row[name + '_median_px'] = float(np.median(error))
        records.append(row)
    report = {'scope': 'internal static-image-track reprojection using a nearby measured LiDAR ray as depth anchor; not an independent calibration target',
              'anchored_tracks': len(records), 'tracks': records,
              'summary': {name: {'observations': len(values), 'median_px': float(np.median(values)), 'p90_px': float(np.quantile(values, .9))} for name, values in errors.items()}}
    (ROOT / 'reprojection_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'tracks'}, indent=2))


if __name__ == '__main__':
    main()
