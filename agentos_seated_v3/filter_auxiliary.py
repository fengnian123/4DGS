import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent


def supported_view(depth, front_depth, front_image, right_image, front_c2w, right_c2w, K):
    y, x = np.where(depth > .3)
    keep = ~((x < 500) & (y > 290))
    valid_fov = cv2.imread(str(ROOT / 'clip/right/valid_mask.png'), 0)
    valid_fov = cv2.erode(valid_fov, np.ones((9, 9), np.uint8)) > 0
    keep &= valid_fov[y, x]
    rays = np.c_[x, y, np.ones(len(x))] @ np.linalg.inv(K).T
    xyz = rays * depth[y, x, None]
    relative = np.linalg.inv(front_c2w) @ right_c2w
    xyz_front = xyz @ relative[:3, :3].T + relative[:3, 3]
    projected = xyz_front @ K.T
    uv = np.rint(projected[:, :2] / projected[:, 2:].clip(1e-5)).astype(int)
    keep &= (xyz_front[:, 2] > .12) & (uv[:, 0] >= 0) & (uv[:, 0] < 768) & (uv[:, 1] >= 0) & (uv[:, 1] < 512)
    fy, fx = np.where(front_depth > 0)
    distance, indices = cKDTree(np.c_[fx, fy]).query(uv, workers=4)
    reference = front_depth[fy[indices], fx[indices]]
    keep &= (distance < 2.5) & (np.abs(reference - xyz_front[:, 2]) < .03 + reference * .05)
    colors_front = front_image[uv[:, 1].clip(0, 511), uv[:, 0].clip(0, 767)].astype(float) / 255
    colors_right = right_image[y, x].astype(float) / 255
    keep &= np.linalg.norm(colors_front - colors_right, axis=1) < .3
    output = np.zeros_like(depth)
    output[y[keep], x[keep]] = depth[y[keep], x[keep]]
    return output


def main():
    path = ROOT / 'agentos_office/sensor_constraints.npz'
    sensor = dict(np.load(path))
    cameras = np.load(ROOT / 'agentos_office/agentos_office.npz')
    output, records = [], []
    for frame in range(80):
        raw = np.load(ROOT / f'sensors/right/{frame:05d}.npz')['train_depth']
        front = np.load(ROOT / f'sensors/front/{frame:05d}.npz')['train_depth']
        rgb_front = cv2.imread(str(ROOT / f'clip/front/{frame:05d}.png'))
        rgb_right = cv2.imread(str(ROOT / f'clip/right/{frame:05d}.png'))
        filtered = supported_view(raw, front, rgb_front, rgb_right, cameras['cam_c2w'][frame], sensor['right_c2w'][frame], cameras['intrinsic'])
        output.append(filtered)
        records.append({'frame': frame, 'original_rays': int((raw > .3).sum()), 'joint_visible_rays': int((filtered > 0).sum())})
    sensor['right_depths'] = np.array(output)
    np.savez_compressed(path, **sensor)
    (ROOT / 'auxiliary_visibility.json').write_text(json.dumps(records, indent=2))
    print('Median original / filtered right rays', np.median([r['original_rays'] for r in records]), np.median([r['joint_visible_rays'] for r in records]), flush=True)


if __name__ == '__main__':
    main()
