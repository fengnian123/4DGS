"""Download original public data from its provider, without redistributing it."""
import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URLS = {
    'coffee': ('coffee_martini.zip', 'https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip'),
    'printer': ('vrig_3dprinter.zip', 'https://github.com/google/hypernerf/releases/download/v0.1/vrig_3dprinter.zip'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', choices=URLS)
    args = parser.parse_args()
    name, url = URLS[args.scene]
    destination = ROOT / '4dgs_local/data' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix('.zip.partial')
    urllib.request.urlretrieve(url, temporary)
    temporary.rename(destination)
    print(destination)


if __name__ == '__main__':
    main()
