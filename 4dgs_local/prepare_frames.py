"""Decode videos to complete, atomically published frame folders."""
import concurrent.futures
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'data/raw/coffee_martini'


def prepare(video):
    destination = video.with_suffix('') / 'images'
    if destination.exists():
        return f'{video.name}: existing folder, left untouched'
    staging = Path(tempfile.mkdtemp(prefix=f'.frames-{video.stem}-', dir=ROOT))
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(video),
                    '-vf', 'scale=676:507:flags=lanczos', '-frames:v', '300',
                    '-start_number', '0', '-compression_level', '1', '-threads', '2',
                    str(staging / '%04d.png')], check=True)
    if len(list(staging.glob('*.png'))) != 300:
        raise RuntimeError(f'{video}: incomplete extraction')
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        return f'{video.name}: active extractor retained; staging={staging}'
    staging.rename(destination)
    return f'{video.name}: 300 complete frames'


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(prepare, sorted(ROOT.glob('cam*.mp4'))):
            print(result, flush=True)
