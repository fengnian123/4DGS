"""Fetch pinned upstream sources and apply the exact local reproduction patches."""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOM = ('https://github.com/vye16/shape-of-motion.git', '579753e1c7ba96f60cd7690e5b835627bd1935e9')
FOURDGS = ('https://github.com/hustvl/4DGaussians.git', '843d5ac636c37e4b611242287754f3d4ed150144')
TARGETS = {
    'som': ('som_local/shape-of-motion', SOM, 'som.patch'),
    '4dgs': ('4dgs_local/4DGaussians', FOURDGS, 'fourdgs.patch'),
    'agentos-v2': ('agentos_seated_v2/shape-of-motion', SOM, 'som-agentos-v2.patch'),
    'agentos-v3': ('agentos_seated_v3/shape-of-motion', SOM, 'som-agentos-v3.patch'),
}


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def apply_patch(repository, name):
    patch = str(ROOT / 'patches' / name)
    applied = subprocess.run(['git', 'apply', '--reverse', '--check', patch], cwd=repository, capture_output=True)
    if applied.returncode == 0:
        return
    run('git', 'apply', '--check', patch, cwd=repository)
    run('git', 'apply', patch, cwd=repository)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', nargs='+', choices=(*TARGETS, 'all'))
    args = parser.parse_args()
    for name in (TARGETS if 'all' in args.targets else dict.fromkeys(args.targets)):
        relative, (url, commit), patch = TARGETS[name]
        path = ROOT / relative
        if not path.exists():
            run('git', 'clone', url, str(path))
            run('git', 'checkout', '--detach', commit, cwd=path)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
        if head != commit:
            raise RuntimeError(f'{relative}: unexpected commit {head}; refusing to overwrite checkout')
        apply_patch(path, patch)
        if name == '4dgs':
            run('git', 'submodule', 'update', '--init', 'submodules/simple-knn', 'submodules/depth-diff-gaussian-rasterization', cwd=path)
            apply_patch(path / 'submodules/simple-knn', 'simple-knn.patch')
        print(f'Ready: {relative}', flush=True)


if __name__ == '__main__':
    main()
