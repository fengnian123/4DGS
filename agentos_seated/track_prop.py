import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
from sam2.build_sam import build_sam2_video_predictor


@torch.inference_mode()
@torch.autocast('cuda', dtype=torch.bfloat16)
def main():
    images = sorted((ROOT / 'clip/front').glob('[0-9]*.png'))
    frame_dir = ROOT / 'prop_tracking_frames'
    destination = ROOT / 'prop_masks'
    base_masks = ROOT / 'base_masks'
    for folder in (frame_dir, destination, base_masks):
        folder.mkdir(exist_ok=True)
    for image in images:
        Image.open(image).save(frame_dir / f'{image.stem}.jpg', quality=96)
        shutil.copyfile(ROOT / 'agentos_office/masks' / image.name, base_masks / image.name)
    checkpoint = os.environ.get('SAM2_CHECKPOINT', str(ROOT.parent / 'agentos_dynamic/models/sam2.1_hiera_tiny.pt'))
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml', checkpoint, apply_postprocessing=False)
    state = predictor.init_state(str(frame_dir), offload_video_to_cpu=True)
    seed = 30
    prompt = {'frame': seed, 'box': [271, 106, 457, 323], 'points': [[350, 140], [356, 270], [520, 190]], 'labels': [1, 1, 0], 'origin': 'Selected manually on rectified raw frame; user screenshot annotations not used.'}
    predictor.add_new_points_or_box(state, frame_idx=seed, obj_id=1, box=np.asarray(prompt['box'],np.float32), points=np.asarray(prompt['points'],np.float32), labels=np.asarray(prompt['labels'],np.int32))
    masks = {}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for reverse in (False, True):
            for frame, object_ids, logits in predictor.propagate_in_video(state, start_frame_idx=seed, reverse=reverse):
                masks[frame] = (logits[0, 0] > 0).cpu().numpy()
    quality = []
    for frame, image in enumerate(images):
        prop = masks[frame]
        cv2.imwrite(str(destination / image.name), prop.astype(np.uint8)*255)
        base = np.asarray(Image.open(base_masks / image.name)) > 0
        merged = base | prop
        cv2.imwrite(str(ROOT / 'agentos_office/masks' / image.name), merged.astype(np.uint8)*255)
        quality.append({'frame': frame, 'prop_fraction': float(prop.mean()), 'foreground_fraction': float(merged.mean())})
        if frame % 10 == 0 or frame in (30, 39, 79):
            rgb = np.asarray(Image.open(image)).copy()
            rgb[prop] = (rgb[prop]*.45 + np.array([255,100,0])*.55).astype(np.uint8)
            Image.fromarray(rgb).save(destination / f'preview_{frame:05d}.jpg')
    (ROOT / 'prop_quality.json').write_text(json.dumps({'prompt': prompt, 'per_frame': quality}, indent=2))
    print(json.dumps(quality[::10], indent=2), flush=True)


if __name__ == '__main__':
    main()
