#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOM_ROOT="$EXPERIMENT_ROOT/../som_local"
SOM_PYTHON="$SOM_ROOT/.venv/bin/python"
export MAX_JOBS="${MAX_JOBS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$EXPERIMENT_ROOT"
"$SOM_PYTHON" inventory.py --source "${AGENTOS_DATA_ROOT:?Set AGENTOS_DATA_ROOT to the local MCAP directory}"
"$SOM_PYTHON" extract_clip.py
"$SOM_PYTHON" export_mcap.py
"$SOM_PYTHON" preprocess.py all
cd "$SOM_ROOT/shape-of-motion"
"$SOM_PYTHON" run_training.py --work-dir "$EXPERIMENT_ROOT/runs/office" \
  --num-fg 20000 --num-bg 40000 --num-dl-workers 0 \
  data:custom --data.data-dir "$EXPERIMENT_ROOT/agentos_office" --data.mask-erosion-radius 3
cd "$EXPERIMENT_ROOT"
"$SOM_PYTHON" render_results.py
