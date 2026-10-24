import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'agentos_seated'
from sam2.build_sam import build_sam2_video_predictor


@torch.inference_mode()
@torch.autocast('cuda',dtype=torch.bfloat16)
def main():
    sequence=ROOT/'agentos_office'
    sequence.mkdir(exist_ok=True)
    if not (ROOT/'clip').exists(): (ROOT/'clip').symlink_to(BASE/'clip',target_is_directory=True)
    shutil.copytree(BASE/'agentos_office/images',sequence/'images',dirs_exist_ok=True)
    for name in ('masks','bootstapir'): (sequence/name).mkdir(exist_ok=True)
    checkpoint = os.environ.get('SAM2_CHECKPOINT', str(ROOT.parent / 'agentos_dynamic/models/sam2.1_hiera_tiny.pt'))
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',checkpoint,apply_postprocessing=False)
    seed=30
    base=cv2.imread(str(BASE/f'base_masks/{seed:05d}.png'),0)>0
    base[:,:240]=False
    count,labels,stats,centers=cv2.connectedComponentsWithStats(base.astype(np.uint8),8)
    component=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    prompt=labels==component
    state=predictor.init_state(str(BASE/'prop_tracking_frames'),offload_video_to_cpu=True)
    predictor.add_new_mask(state,frame_idx=seed,obj_id=1,mask=prompt)
    tracked={}
    for reverse in (False,True):
        for frame,ids,logits in predictor.propagate_in_video(state,start_frame_idx=seed,reverse=reverse): tracked[frame]=(logits[0,0]>0).cpu().numpy()
    records=[]
    for frame in range(80):
        old=cv2.imread(str(BASE/f'base_masks/{frame:05d}.png'),0)>0
        yy,xx=np.indices(old.shape)
        other=old&((xx<240)|((xx<360)&(yy>260)))
        prop=cv2.imread(str(BASE/f'prop_masks/{frame:05d}.png'),0)>0
        mask=tracked[frame]|other|prop
        cv2.imwrite(str(sequence/f'masks/{frame:05d}.png'),mask.astype(np.uint8)*255)
        src=np.load(BASE/f'agentos_office/bootstapir/{frame:05d}_{frame:05d}.npy')[:,:2].round().astype(int)
        keep=mask[src[:,1].clip(0,511),src[:,0].clip(0,767)]
        for target in range(80):
            array=np.load(BASE/f'agentos_office/bootstapir/{frame:05d}_{target:05d}.npy')
            np.save(sequence/f'bootstapir/{frame:05d}_{target:05d}.npy',array[keep])
        if frame in (0,20,40,48,60,79):
            rgb=np.array(Image.open(BASE/f'clip/front/{frame:05d}.png'))
            rgb[mask]=(rgb[mask]*.5+np.array([0,220,100])*.5).astype(np.uint8)
            Image.fromarray(rgb).save(ROOT/f'mask_check_{frame:05d}.jpg')
        records.append({'frame':frame,'foreground_fraction':float(mask.mean()),'retained_queries':int(keep.sum()),'original_queries':len(src)})
    shutil.copytree(BASE/'prop_masks',ROOT/'prop_masks',dirs_exist_ok=True)
    shutil.copyfile(BASE/'prop_quality.json',ROOT/'prop_quality.json')
    (ROOT/'mask_quality.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records[::10],indent=2),flush=True)


if __name__=='__main__':main()
