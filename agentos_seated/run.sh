#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$EXPERIMENT_ROOT/../agentos_dynamic"
SOM_ROOT="$EXPERIMENT_ROOT/../som_local"
SOM_PYTHON="$SOM_ROOT/.venv/bin/python"
export MAX_JOBS="${MAX_JOBS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
"$SOM_PYTHON" "$PIPELINE_ROOT/inventory.py" --source "${AGENTOS_DATA_ROOT:?Set AGENTOS_DATA_ROOT to the local MCAP directory}"
"$SOM_PYTHON" "$PIPELINE_ROOT/extract_clip.py" --root "$EXPERIMENT_ROOT" --source-name unified_20260817_160810_0002 --start 15.95 --duration 8 --pitch 0 --fov 95 --width 768 --height 512
"$SOM_PYTHON" "$EXPERIMENT_ROOT/show_correction.py"
"$SOM_PYTHON" "$PIPELINE_ROOT/export_mcap.py" --root "$EXPERIMENT_ROOT"
"$SOM_PYTHON" "$PIPELINE_ROOT/preprocess.py" masks --root "$EXPERIMENT_ROOT" --include-seated-chairs
"$SOM_PYTHON" "$EXPERIMENT_ROOT/track_prop.py"
"$SOM_PYTHON" "$PIPELINE_ROOT/preprocess.py" depth --root "$EXPERIMENT_ROOT"
"$SOM_PYTHON" "$PIPELINE_ROOT/preprocess.py" tracks --root "$EXPERIMENT_ROOT"
cd "$SOM_ROOT/shape-of-motion"
"$SOM_PYTHON" run_training.py --work-dir "$EXPERIMENT_ROOT/runs/office" --num-fg 25000 --num-bg 50000 --num-dl-workers 0 data:custom --data.data-dir "$EXPERIMENT_ROOT/agentos_office" --data.mask-erosion-radius 3
"$SOM_PYTHON" "$PIPELINE_ROOT/render_results.py" --root "$EXPERIMENT_ROOT"
