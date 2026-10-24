"""Safely extract the official preprocessed sequence and report available splits."""
from pathlib import Path
import zipfile
import json

ROOT = Path(__file__).resolve().parent
destination = (ROOT / 'data/extracted').resolve()
destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ROOT / 'data/Balloon1.zip') as archive:
    for name in archive.namelist():
        if not (destination / name).resolve().is_relative_to(destination):
            raise ValueError(name)
    archive.extractall(destination)
for path in destination.rglob('splits/train.json'):
    scene = path.parent.parent
    train = json.loads(path.read_text())
    val = json.loads((scene / 'splits/val.json').read_text())
    print(scene, 'train', len(train['frame_names']), 'val', len(val['frame_names']), flush=True)
    print('first train frames', train['frame_names'][:3], flush=True)
