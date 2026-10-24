#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
method="${1:?Usage: bash scripts/setup.sh som|4dgs}"
case "$method" in
  som) environment_root="$release_root/som_local" ;;
  4dgs) environment_root="$release_root/4dgs_local" ;;
  *) echo "Choose som or 4dgs" >&2; exit 2 ;;
esac
"${BOOTSTRAP_PYTHON:-python3.11}" -m venv "$environment_root/.venv"
runtime="$environment_root/.venv/bin/python"
"$runtime" -m pip install --upgrade pip setuptools wheel
"$runtime" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$runtime" -m pip install -r "$environment_root/requirements-local.txt"
if [[ "$method" == 4dgs ]]; then
  "$runtime" -m pip install ninja
  "$runtime" -m pip install --no-build-isolation "$environment_root/4DGaussians/submodules/simple-knn" "$environment_root/4DGaussians/submodules/depth-diff-gaussian-rasterization"
fi
