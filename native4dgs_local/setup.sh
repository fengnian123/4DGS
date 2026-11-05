#!/usr/bin/env bash
set -euo pipefail
native4dgs_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NATIVE4DGS_ROOT="${NATIVE4DGS_ROOT:-$native4dgs_package}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export PIP_CACHE_DIR="$NATIVE4DGS_ROOT/cache/pip"
export TORCH_EXTENSIONS_DIR="$NATIVE4DGS_ROOT/cache/torch_extensions_auditfix"
export MAX_JOBS="${MAX_JOBS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONNOUSERSITE=1
: "${TORCH_CUDA_ARCH_LIST:?Set the CUDA architecture of the selected GPU; the recorded run used 8.6}"
native4dgs_python="${PYTHON_BIN:-python3.10}"
mkdir -p "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$NATIVE4DGS_ROOT/envs"
"$native4dgs_python" "$native4dgs_package/bootstrap.py"
"$native4dgs_python" -m venv "$NATIVE4DGS_ROOT/envs/train"
native4dgs_python="$NATIVE4DGS_ROOT/envs/train/bin/python"
"$native4dgs_python" -m pip install --upgrade pip setuptools wheel
"$native4dgs_python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$native4dgs_python" -m pip install -r "$native4dgs_package/requirements.txt"
native4dgs_code="${NATIVE4DGS_CODE_ROOT:-$NATIVE4DGS_ROOT/code/4d-gaussian-splatting}"
"$native4dgs_python" -m pip install --no-build-isolation "$native4dgs_code/simple-knn" "$native4dgs_code/pointops2"
printf 'Environment ready: %s\n' "$NATIVE4DGS_ROOT/envs/train"
