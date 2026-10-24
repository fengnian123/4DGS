"""Publish the latest fully rendered checkpoint of each scene."""
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent / 'results'


def build():
    selected = {}
    for path in ROOT.glob('*/metrics.json'):
        metrics = json.loads(path.read_text())
        scene = metrics['scene']
        if scene in selected and selected[scene]['iteration'] >= metrics['iteration']:
            continue
        if not all((path.parent / file).is_file() for file in
                   ['comparison.mp4', 'fixed_view_time.mp4', 'fixed_time_view.mp4', 'comparison.jpg']):
            continue
        selected[scene] = dict(directory=path.parent.name, iteration=metrics['iteration'])
    with tempfile.NamedTemporaryFile(mode='w', prefix='.manifest-', suffix='.tmp', dir=ROOT, delete=False) as handle:
        json.dump(selected, handle, indent=2)
        temporary = Path(handle.name)
    temporary.replace(ROOT / 'manifest.json')
    return selected


if __name__ == '__main__':
    print(json.dumps(build()))
