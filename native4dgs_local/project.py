"""Portable paths for the isolated Native 4DGS experiment."""
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('NATIVE4DGS_ROOT', PACKAGE)).expanduser().resolve()
CODE = Path(os.environ.get('NATIVE4DGS_CODE_ROOT', ROOT / 'code/4d-gaussian-splatting')).resolve()
SCENES = ('jumpingjacks_auditfix_full', 'bouncingballs_auditfix_full', 'coffee_martini_auditfix_full')


def config(name):
    from omegaconf import OmegaConf
    if name not in SCENES:
        raise ValueError(f'Unknown experiment: {name}')
    cfg = OmegaConf.load(PACKAGE / 'configs' / f'{name}.yaml')
    for key in ('source_path', 'model_path'):
        path = Path(cfg.ModelParams[key])
        if path.is_absolute():
            raise ValueError('Packaged configurations must contain relative paths')
        cfg.ModelParams[key] = str(ROOT / path)
    return cfg


def prepare_environment():
    for name in ('cache', 'tmp', 'logs', 'runs', 'results'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    for name, path in {
        'TORCH_EXTENSIONS_DIR': ROOT / 'cache/torch_extensions_auditfix',
        'TORCH_HOME': ROOT / 'cache/torch',
        'XDG_CACHE_HOME': ROOT / 'cache/xdg',
        'PIP_CACHE_DIR': ROOT / 'cache/pip',
        'CUDA_CACHE_PATH': ROOT / 'cache/cuda',
        'MPLCONFIGDIR': ROOT / 'cache/matplotlib',
        'TMPDIR': ROOT / 'tmp',
    }.items():
        os.environ.setdefault(name, str(path))
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    for key in ('MAX_JOBS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(key, '4')
    os.environ.setdefault('PYTHONNOUSERSITE', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def definitions():
    return json.loads((PACKAGE / 'experiments.json').read_text(encoding='utf-8'))
