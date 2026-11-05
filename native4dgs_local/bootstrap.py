"""Download the pinned upstream, apply one cumulative patch, verify source hashes."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile
from project import CODE, PACKAGE


def verify():
    entries = json.loads((PACKAGE / 'provenance/source_manifest.json').read_text())
    failures = [row['path'] for row in entries if not (CODE / row['path']).is_file()
                or hashlib.sha256((CODE / row['path']).read_bytes()).hexdigest() != row['sha256']]
    if failures:
        raise RuntimeError(f'Source differs from the trained snapshot: {failures}')
    print(f'Verified {len(entries)} source files against the trained snapshot.')


def main():
    if CODE.exists():
        verify()  # Never replace an existing working tree.
        return
    provenance = json.loads((PACKAGE / 'provenance/upstream.json').read_text())
    request = urllib.request.Request(provenance['url'], headers={'User-Agent': 'native4dgs-reproduction'})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != provenance['sha256']:
        raise RuntimeError('The upstream archive hash changed; inspect before proceeding.')
    CODE.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        prefix = archive.namelist()[0].split('/')[0] + '/'
        for entry in archive.infolist():
            relative = entry.filename.removeprefix(prefix)
            if entry.is_dir() or not relative:
                continue
            target = (CODE / relative).resolve()
            if not target.is_relative_to(CODE):
                raise ValueError(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
    patch = PACKAGE.parent / 'patches/native4dgs_yang2024.patch'
    for args in (['--check'], []):
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'apply', *args, str(patch)], cwd=CODE, check=True)
    verify()


if __name__ == '__main__':
    main()
