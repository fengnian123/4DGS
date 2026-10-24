from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
capture = cv2.VideoCapture(str(ROOT / 'clip/front/clip.h264'))
for frame in range(41):
    ok, raw = capture.read()
    if not ok:
        raise RuntimeError(f'Cannot decode raw frame {frame}')
capture.release()
original = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
corrected = Image.open(ROOT / 'clip/front/00040.png').convert('RGB')
width, height = corrected.size
canvas = Image.new('RGB', (width * 2, height + 28), '#17202a')
preview = original.copy()
preview.thumbnail((width, height))
canvas.paste(preview, ((width-preview.width)//2, 28+(height-preview.height)//2))
canvas.paste(corrected, (width, 28))
draw = ImageDraw.Draw(canvas)
draw.text((8, 8), f'Original fisheye {original.width}x{original.height}', fill='white')
draw.text((width+8, 8), 'Calibrated pinhole view, vertical FOV 95 degrees', fill='white')
canvas.save(ROOT / 'clip/fisheye_correction.png')
