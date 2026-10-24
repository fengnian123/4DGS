"""Make a derived pinhole dataset compatible with the official centered camera."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
source = ROOT / 'data/raw/vrig-3dprinter'
destination = ROOT / 'data/derived/printer_calibrated'
destination.mkdir(parents=True, exist_ok=True)
for name in ['scene.json', 'metadata.json', 'dataset.json', 'points3D_downsample2.ply']:
    shutil.copy2(source / name, destination / name)
(destination / 'camera').mkdir(exist_ok=True)
(destination / 'rgb/2x').mkdir(parents=True, exist_ok=True)
ids = json.loads((source / 'dataset.json').read_text())['ids']
for index, frame in enumerate(ids):
    parameters = json.loads((source / 'camera' / f'{frame}.json').read_text())
    image = cv2.imread(str(source / 'rgb/2x' / f'{frame}.png'))
    width, height = 536, 720
    scale = image.shape[1] / parameters['image_size'][0]
    focal = parameters['focal_length'] * scale
    cx, cy = np.array(parameters['principal_point']) * scale
    intrinsics = np.array([[focal, parameters['skew'] * scale, cx],
                           [0, focal * parameters['pixel_aspect_ratio'], cy], [0, 0, 1.]])
    k1, k2, k3 = parameters['radial_distortion']
    p1, p2 = parameters['tangential_distortion']
    distortion = np.array([k1, k2, p1, p2, k3])
    target = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])
    maps = cv2.initUndistortRectifyMap(intrinsics, distortion, None, target,
                                      (width, height), cv2.CV_32FC1)
    corrected = cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(destination / 'rgb/2x' / f'{frame}.png'), corrected,
                [cv2.IMWRITE_PNG_COMPRESSION, 1])
    parameters.update(image_size=[width * 2, height * 2], principal_point=[width, height],
                      skew=0., pixel_aspect_ratio=1., radial_distortion=[0., 0., 0.],
                      tangential_distortion=[0., 0.])
    (destination / 'camera' / f'{frame}.json').write_text(json.dumps(parameters, indent=2))
    if index % 100 == 0:
        print(f'calibrated frame {index}/{len(ids)}', flush=True)
metadata = dict(source=str(source), resolution=[536, 720],
                operation='undistort and reproject to centered square-pixel pinhole',
                preserves=['world poses', 'focal length', 'timestamps', 'train/val split', 'author point cloud'],
                field_of_view='upper centered 720 rows; not the original full 960-row image',
                interpolation='OpenCV bilinear remap', note='derived data, original is untouched')
(destination / 'preprocessing.json').write_text(json.dumps(metadata, indent=2))
print(destination, flush=True)
