"""Download the author's public preprocessed scene, handling Drive confirmation."""
from pathlib import Path
import argparse
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--id', default='15PirJRqsT5lLjuGdLWALBDFMQanj8FTh')
parser.add_argument('--name', default='paper-windmill')
args = parser.parse_args()
session = requests.Session()
url = 'https://drive.usercontent.google.com/download'
params = dict(id=args.id, export='download')
response = session.get(url, params=params, stream=True, timeout=120)
response.raise_for_status()
if 'text/html' in response.headers.get('content-type', ''):
    soup = BeautifulSoup(response.text, 'html.parser')
    form = soup.find('form')
    if form is None:
        raise RuntimeError(soup.get_text()[:1000])
    fields = {i['name']: i.get('value', '') for i in form.select('input[name]')}
    response = session.get(url, params=fields, stream=True, timeout=120)
    response.raise_for_status()
    if 'text/html' in response.headers.get('content-type', ''):
        raise RuntimeError(response.text[:500])
destination = ROOT / 'data' / (args.name + '.zip')
size = 0
with destination.open('wb') as handle:
    for chunk in response.iter_content(8 * 1024 * 1024):
        handle.write(chunk)
        size += len(chunk)
        print(f'{size / 1024**2:.0f} MiB', flush=True)
print(destination, flush=True)
