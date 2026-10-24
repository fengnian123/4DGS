"""Train with the scene's official settings, then render held-out and virtual views."""
import argparse
import os
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('scene', choices=['printer', 'printer_calibrated', 'coffee'])
parser.add_argument('--eval-only', action='store_true')
parser.add_argument('--iteration', type=int, default=-1)
args = parser.parse_args()
sources = dict(printer=ROOT / 'data/raw/vrig-3dprinter',
               printer_calibrated=ROOT / 'data/derived/printer_calibrated',
               coffee=ROOT / 'data/raw/coffee_martini')
source = sources[args.scene]
model = ROOT / 'runs' / ('printer_calibrated_stable' if args.scene == 'printer_calibrated' else args.scene)
environment = dict(os.environ)
environment.setdefault('OMP_NUM_THREADS', '4')
if not args.eval_only:
    command = [sys.executable, '-u', 'train.py', '-s', str(source), '-m', str(model),
               '--configs', str(ROOT / 'configs' / f'{args.scene}.py'),
               '--port', {'printer': '6027', 'coffee': '6028', 'printer_calibrated': '6029'}[args.scene],
               '--test_iterations', '3000', '7000', '14000',
               '--save_iterations', '3000', '7000', '14000']
    subprocess.run(command, cwd=ROOT / '4DGaussians', env=environment, check=True)
subprocess.run([sys.executable, '-u', str(ROOT / 'evaluate.py'), '--scene', args.scene,
                '--iteration', str(args.iteration)], cwd=ROOT, env=environment, check=True)
