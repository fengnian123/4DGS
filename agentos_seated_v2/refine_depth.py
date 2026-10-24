import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent
data=np.load(ROOT/'agentos_office/agentos_office.npz')
baseline=np.load(ROOT.parent/'agentos_seated/agentos_office/agentos_office.npz')
bundle=np.load(ROOT/'camera_refinement.npz')
poses=np.linalg.inv(bundle['refined_c2w']); depth=baseline['depths'].copy()
records=[]
for frame in range(80):
    keep=bundle['frame_ids']==frame
    ids=bundle['point_ids'][keep]; uv=bundle['observations'][keep]
    if 'heldout_points' in bundle:valid=~bundle['heldout_points'][ids]
    else: valid=np.ones(len(ids),bool)
    ids,uv=ids[valid],uv[valid]
    camera=bundle['landmarks'][ids]@poses[frame,:3,:3].T+poses[frame,:3,3]
    image=cv2.imread(str(ROOT/f'clip/front/{frame:05d}.png')).astype(np.float32)/255
    pixels=uv.round().astype(int)
    original=depth[frame,pixels[:,1],pixels[:,0]]
    residual=1/np.maximum(camera[:,2],.3)-1/original
    residual=np.clip(residual,-.25,.25)
    yy,xx=np.indices(depth.shape[1:]); query=np.c_[xx.ravel(),yy.ravel()]
    distances,neighbors=cKDTree(uv).query(query,k=min(8,len(uv)),workers=4)
    colors=image[pixels[:,1],pixels[:,0]]
    color_distance=((colors[neighbors]-image.reshape(-1,1,3))**2).sum(-1)
    weights=np.exp(-distances**2/(2*24**2)-color_distance/(2*.10**2))*(distances<60)
    correction=(weights*residual[neighbors]).sum(-1)/(weights.sum(-1)+1e-8)
    confidence=np.clip(weights.sum(-1),0,1)
    mask=cv2.imread(str(ROOT/f'agentos_office/masks/{frame:05d}.png'),0)==0
    correction=(correction*confidence).reshape(depth.shape[1:])*mask
    updated=np.clip(1/np.maximum(1/depth[frame]+correction,1/15),.3,15)
    change=np.abs(updated-depth[frame])
    # Corrections remain small compared with the measured scene dimensions.
    updated=np.where(change<.2,updated,depth[frame])
    depth[frame]=updated
    records.append({'frame':frame,'static_anchor_count':len(ids),'median_background_depth_change_m':float(np.median(change[mask]))})
np.savez_compressed(ROOT/'agentos_office/agentos_office.npz',cam_c2w=data['cam_c2w'],intrinsic=data['intrinsic'],depths=depth)
(ROOT/'depth_refinement.json').write_text(json.dumps(records,indent=2))
print('Refined static depth using bundle-consistent landmarks',flush=True)
