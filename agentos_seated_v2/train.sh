#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MAX_JOBS="${MAX_JOBS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$experiment_dir/shape-of-motion"
exec "$experiment_dir/../som_local/.venv/bin/python" -u run_training.py \
  --work-dir "$experiment_dir/runs/office" \
  --num-fg 30000 --num-bg 80000 --num-motion-bases 16 --num-dl-workers 0 \
  --num-epochs 800 --optim.max-steps 8000 \
  --optim.stop-control-steps 6000 --optim.stop-control-by-screen-steps 6000 \
  data:custom --data.data-dir "$experiment_dir/agentos_office" --data.mask-erosion-radius 3
