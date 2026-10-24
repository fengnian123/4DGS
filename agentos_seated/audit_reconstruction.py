import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'optimization_audit'
OUT.mkdir(exist_ok=True)
data = np.load(ROOT / 'agentos_office/agentos_office.npz')
K, poses, depths = data['intrinsic'], data['cam_c2w'], data['depths']
images = [cv2.imread(str(ROOT / f'clip/front/{i:05d}.png')) for i in range(80)]
masks = [cv2.imread(str(ROOT / f'agentos_office/masks/{i:05d}.png'),0)>0 for i in range(80)]
gray = [cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images]
sift = cv2.SIFT_create(nfeatures=3000,contrastThreshold=.025)
features=[]
for g,m in zip(gray,masks):
    static=(1-cv2.dilate(m.astype(np.uint8),np.ones((15,15),np.uint8)))*255
    features.append(sift.detectAndCompute(g,static))
records=[]
for a,b in [(0,10),(0,40),(20,40),(40,60),(40,79),(60,79)]:
    ka,da=features[a]; kb,db=features[b]
    matches=cv2.BFMatcher().knnMatch(da,db,k=2)
    matches=[p for p,q in matches if p.distance<.7*q.distance]
    u=np.array([ka[p.queryIdx].pt for p in matches],np.float32)
    v=np.array([kb[p.trainIdx].pt for p in matches],np.float32)
    _,inliers=cv2.findFundamentalMat(u,v,cv2.FM_RANSAC,1.5,.999)
    u,v=u[inliers[:,0]>0],v[inliers[:,0]>0]
    xy=np.rint(u).astype(int)
    z=depths[a][xy[:,1],xy[:,0]]
    xyz=(np.c_[u,np.ones(len(u))]@np.linalg.inv(K).T)*z[:,None]
    world=xyz@poses[a,:3,:3].T+poses[a,:3,3]
    camera=world@np.linalg.inv(poses[b])[:3,:3].T+np.linalg.inv(poses[b])[:3,3]
    projection=camera@K.T
    projected=projection[:,:2]/projection[:,2:]
    error=np.linalg.norm(projected-v,axis=-1)
    relative=np.linalg.inv(poses[b])@poses[a]
    success,r,t,selected=cv2.solvePnPRansac(xyz.astype(np.float32),v,K,None,iterationsCount=300,reprojectionError=3.,confidence=.999)
    row={'source':a,'target':b,'static_matches':len(u),'lidar_pose_plus_mono_depth_median_pixel_error':float(np.median(error)),'pnp_inliers':len(selected) if success else 0}
    if success:
        pred=cv2.projectPoints(xyz,r,t,K,None)[0][:,0]
        row['pnp_median_pixel_error']=float(np.median(np.linalg.norm(pred-v,axis=-1)))
        row['relative_translation_difference_m']=float(np.linalg.norm(t[:,0]-relative[:3,3]))
    records.append(row)
    print(row,flush=True)
rows=[]
for i in (0,20,40,48,60,79):
    rgb=cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB)
    overlay=rgb.copy(); overlay[masks[i]]=(overlay[masks[i]]*.5+np.array([0,220,100])*.5).astype(np.uint8)
    row=Image.new('RGB',(1536,540),'#17202a')
    row.paste(Image.fromarray(rgb),(0,28));row.paste(Image.fromarray(overlay),(768,28))
    ImageDraw.Draw(row).text((8,8),f'Frame {i} | real / current foreground mask',fill='white')
    rows.append(row)
canvas=Image.new('RGB',(1536,540*len(rows)))
for i,row in enumerate(rows):canvas.paste(row,(0,540*i))
canvas.save(OUT/'mask_audit.jpg',quality=92)
(OUT/'geometry_audit.json').write_text(json.dumps(records,indent=2))
