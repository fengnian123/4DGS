"""Check frame coverage, finite metrics, encoded videos and temporal variation."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('results', nargs='+')
args = parser.parse_args()
for name in args.results:
    path = ROOT / 'results' / name
    m = json.loads((path / 'metrics.json').read_text())
    n = m['test_frames']
    assert len(list((path / 'renders').glob('*.png'))) == n
    assert len(list((path / 'ground_truth').glob('*.png'))) == n
    assert len(m['frames']) == n
    assert np.isfinite([[f['psnr'], f['ssim']] for f in m['frames']]).all()
    assert np.isclose(m['psnr'], np.mean([f['psnr'] for f in m['frames']]))
    assert np.isclose(m['ssim'], np.mean([f['ssim'] for f in m['frames']]))
    videos = {}
    for file in path.glob('*.mp4'):
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
                           '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width,height,nb_frames',
                           '-of', 'json', str(file)], text=True))['streams'][0]
        assert probe['codec_name'] == 'h264'
        assert int(probe['nb_frames']) > 1
        if file.stem == 'comparison':
            assert int(probe['nb_frames']) == n
        videos[file.name] = probe
    assert len(videos) == 3
    cap = cv2.VideoCapture(str(path / 'fixed_view_time.mp4'))
    ok, first = cap.read()
    assert ok
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, middle = cap.read()
    assert ok
    cap.release()
    # Ignore the changing caption; check rendered image content only.
    change = float(np.abs(first[40:].astype(float) - middle[40:].astype(float)).mean() / 255)
    assert change > 0.001, f'{name}: time query returned almost identical images'
    result = dict(result=name, frame_coverage=True, finite_metrics=True,
                  fixed_view_time_mean_absolute_change=change, videos=videos)
    (path / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
