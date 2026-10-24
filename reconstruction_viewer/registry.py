"""Load scene paths relative to the registry, with optional environment expansion."""
import json
import os
from pathlib import Path


def load_registry(path):
    path = Path(path).expanduser().resolve()
    entries = json.loads(path.read_text())
    if not entries:
        raise ValueError("Scene registry is empty")
    ids, labels = set(), set()
    for entry in entries:
        if entry["id"] in ids or entry["label"] in labels:
            raise ValueError("Scene IDs and labels must be unique")
        ids.add(entry["id"])
        labels.add(entry["label"])
        for key in ("root", "python", "code_root"):
            if key in entry:
                value = Path(os.path.expandvars(entry[key])).expanduser()
                # Resolving a venv's python symlink would bypass its pyvenv.cfg.
                entry[key] = os.path.abspath(path.parent / value)
    return entries
