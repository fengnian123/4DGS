"""Launch one full published configuration; never silently reduce its budget."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from project import ROOT, CODE, SCENES, config, prepare_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', choices=SCENES)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--best-psnr', type=float)
    args = parser.parse_args()
    from omegaconf import OmegaConf
    cfg = config(args.scene)
    source, output = Path(cfg.ModelParams.source_path), Path(cfg.ModelParams.model_path)
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        return
    prepare_environment()
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, wanted), hard))
    if not (CODE / 'train.py').is_file():
        raise FileNotFoundError('Run bootstrap.py first.')
    for split in ('train', 'test'):
        if not (source / f'transforms_{split}.json').is_file():
            raise FileNotFoundError(f'Missing prepared data: {source}')
    if not args.resume and list(output.glob('chkpnt*.pth')):
        raise FileExistsError('Existing checkpoints found; choose a new experiment root or explicitly resume.')
    resolved = ROOT / 'logs' / f'{args.scene}.resolved.yaml'
    OmegaConf.save(cfg, resolved)
    command = [sys.executable, '-u', str(CODE / 'train.py'), '--config', str(resolved)]
    if args.scene.startswith('coffee'):
        os.environ['NATIVE4DGS_OFFLOAD_ADAM'] = '1'
    if args.resume:
        checkpoint = args.resume.resolve()
        if not checkpoint.is_relative_to(output.resolve()) or not checkpoint.is_file():
            raise ValueError('Resume checkpoint must belong to this experiment.')
        if args.best_psnr is None:
            parser.error('--resume requires the previous best held-out --best-psnr')
        os.environ['NATIVE4DGS_RESUME_BEST_PSNR'] = str(args.best_psnr)
        command.extend(['--start_checkpoint', str(checkpoint)])
    status = {'scene': args.scene, 'budget': int(cfg.OptimizationParams.iterations),
              'resume': bool(args.resume), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'stage': 'training'}
    status_path = ROOT / 'logs' / f'{args.scene}.status.json'
    status_path.write_text(json.dumps(status, indent=2))
    with (ROOT / 'logs' / f'{args.scene}.log').open('a' if args.resume else 'w') as log:
        completed = subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
    status['stage'] = 'trained' if completed.returncode == 0 else 'failed'
    status['returncode'] = completed.returncode
    status_path.write_text(json.dumps(status, indent=2))
    if completed.returncode:
        raise RuntimeError(f'Training failed; see {log.name}')
    print(f'Training complete. Evaluate with evaluate.py {args.scene}')


if __name__ == '__main__':
    main()
