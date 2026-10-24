"""Register portable reproduction directories or existing local experiment outputs."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projects-root', type=Path, default=ROOT)
    parser.add_argument('--experiments-root', type=Path, default=ROOT)
    parser.add_argument('--include-private', action='store_true')
    parser.add_argument('--available-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'reconstruction_viewer/scenes.local.json')
    args = parser.parse_args()
    projects, experiments = args.projects_root.resolve(), args.experiments_root.resolve()
    entries = json.loads((ROOT / 'reconstruction_viewer/scenes.all.example.json').read_text())
    selected = []
    for entry in entries:
        private = 'agentos' in entry['id']
        if private and not args.include_private:
            continue
        for key in ('root', 'python', 'code_root'):
            if key not in entry:
                continue
            relative = Path(entry[key]).relative_to('..')
            base = experiments if relative.parts[0].startswith('agentos_') else projects
            entry[key] = str(base / relative)
        if entry['id'] == 'som-balloon1' and projects != ROOT:
            entry['results'] = 'runs/balloon1/evaluation'
        if args.available_only:
            model = Path(entry['root']) / entry['model']
            checkpoint = (model / 'checkpoints/last.ckpt' if entry['adapter'] != 'adapters.fourdgs:FourDGS'
                          else model / f'point_cloud/iteration_{entry["iteration"]}/point_cloud.ply')
            if not Path(entry['python']).is_file() or not checkpoint.is_file():
                continue
        selected.append(entry)
    if not selected:
        parser.error('No available scenes; prepare data and train a model or specify existing local roots')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(selected)} scenes: {args.output}')


if __name__ == '__main__':
    main()
