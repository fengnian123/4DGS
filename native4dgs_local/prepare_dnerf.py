"""Stage only the selected D-NeRF scenes; omit stale initialization clouds."""
import argparse
import json
from pathlib import Path
import shutil
from project import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='Extracted directory containing both scene folders')
    parser.add_argument('--copy', action='store_true', help='Copy rather than create relative-independent symlinks')
    args = parser.parse_args()
    for scene in ('jumpingjacks', 'bouncingballs'):
        original = (args.source / scene).resolve()
        destination = ROOT / 'data/paper/dnerf' / scene
        if destination.exists():
            raise FileExistsError(f'Refusing to merge with existing prepared data: {destination}')
        for split in ('train', 'test'):
            metadata = json.loads((original / f'transforms_{split}.json').read_text())
            assert metadata['frames'] and all('time' in row for row in metadata['frames'])
        destination.mkdir(parents=True)
        for item in original.iterdir():
            if item.name.lower() == 'points3d.ply':
                continue
            target = destination / item.name
            if args.copy:
                shutil.copytree(item, target) if item.is_dir() else shutil.copyfile(item, target)
            else:
                target.symlink_to(item, target_is_directory=item.is_dir())
        print(destination)


if __name__ == '__main__':
    main()
