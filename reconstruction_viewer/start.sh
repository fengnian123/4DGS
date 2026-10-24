#!/usr/bin/env bash
set -euo pipefail
VIEWER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${VIEWER_PYTHON:-$VIEWER_ROOT/../som_local/.venv/bin/python}" -u "$VIEWER_ROOT/viewer.py" "$@"
