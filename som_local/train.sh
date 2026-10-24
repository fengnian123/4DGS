#!/usr/bin/env bash
set -euo pipefail
SOM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MAX_JOBS="${MAX_JOBS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$SOM_ROOT/shape-of-motion"
"$SOM_ROOT/.venv/bin/python" -u run_training.py \
  --work-dir "$SOM_ROOT/runs/balloon1" \
  --num-dl-workers 0 \
  data:nvidia --data.data-dir "$SOM_ROOT/data/extracted/Balloon1"
