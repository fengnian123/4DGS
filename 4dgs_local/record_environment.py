"""Record reproducibility metadata and local compatibility patches."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import torch

ROOT = Path(__file__).resolve().parent
repository = ROOT / '4DGaussians'
environment = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0),
                   code_repository='https://github.com/hustvl/4DGaussians',
                   code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
                   submodules=subprocess.check_output(['git', 'submodule', 'status'], cwd=repository, text=True).splitlines(),
                   packages={name: version(name) for name in ['mmcv', 'numpy', 'plyfile', 'yapf', 'opencv-python', 'scipy', 'imageio', 'lpips']},
                   data_sources={'printer': 'https://github.com/google/hypernerf/releases/download/v0.1/vrig_3dprinter.zip',
                                 'coffee': 'https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip',
                                 'printer_cloud': 'https://drive.google.com/file/d/1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5/view'},
                   evaluations='full official held-out splits; float RGB PSNR and SSIM, unmasked',
                   initialization={'coffee': 'training cameras only, calibrated SIFT triangulation at frame 0',
                                   'printer': 'author provided cloud; generation provenance not independently audited'},
                   variants={'printer': 'official loader, original release RGB/2x',
                             'coffee': '676x507 resized video frames',
                             'printer_calibrated': 'derived undistorted centered pinhole 536x720, original unchanged'})
(ROOT / 'results/provenance.json').write_text(json.dumps(environment, indent=2))
for name, cwd in [('upstream', repository), ('simple_knn', repository / 'submodules/simple-knn')]:
    patch = subprocess.check_output(['git', 'diff', '--binary'], cwd=cwd)
    (ROOT / 'results' / f'{name}.patch').write_bytes(patch)
print(json.dumps(environment, indent=2))
