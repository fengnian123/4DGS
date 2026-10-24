import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'agentos_seated'


def main():
    cv2.setNumThreads(4)
    data=np.load(BASE/'agentos_office/agentos_office.npz')
    K=data['intrinsic'].astype(float); poses=data['cam_c2w'].astype(float); initial=np.linalg.inv(poses)
    images=[cv2.imread(str(BASE/f'clip/front/{i:05d}.png'),0) for i in range(80)]
    masks=[cv2.dilate(cv2.imread(str(ROOT/f'agentos_office/masks/{i:05d}.png'),0),np.ones((15,15),np.uint8))>0 for i in range(80)]
    points=[]; point_sigma=[]; observations=[]; frame_ids=[]; point_ids=[]
    for anchor in (0,15,30,45,60,79):
        corners=cv2.goodFeaturesToTrack(images[anchor],maxCorners=160,qualityLevel=.015,minDistance=10,mask=(~masks[anchor]).astype(np.uint8)*255)
        uv=corners[:,0]; nearest=uv.round().astype(int)
        z=data['depths'][anchor,nearest[:,1],nearest[:,0]].copy()
        sparse=np.load(BASE/f'clip/lidar_depth/{anchor:05d}.npy')
        sy,sx=np.where(sparse>.4)
        distances,indices=cKDTree(np.c_[sx,sy]).query(uv)
        lidar=sparse[sy[indices],sx[indices]]
        good=(distances<2.5)&(lidar>z*.7)&(lidar<z*1.3)
        z[good]=lidar[good]
        xyz=(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T)*z[:,None]
        world=xyz@poses[anchor,:3,:3].T+poses[anchor,:3,3]
        offset=len(points)
        points.extend(world);point_sigma.extend(np.where(good,.03,.12))
        for j,p in enumerate(uv):
            observations.append(p);frame_ids.append(anchor);point_ids.append(offset+j)
        for direction in (-1,1):
            current=uv.copy(); valid=np.ones(len(uv),bool); previous=anchor
            for frame in range(anchor+direction,80 if direction==1 else -1,direction):
                ids=np.where(valid)[0]
                if len(ids)<4:break
                source=current[ids,None].astype(np.float32)
                target,status,error=cv2.calcOpticalFlowPyrLK(images[previous],images[frame],source,None,winSize=(25,25),maxLevel=4)
                back,reverse,_=cv2.calcOpticalFlowPyrLK(images[frame],images[previous],target,None,winSize=(25,25),maxLevel=4)
                pixels=target[:,0].round().astype(int)
                accepted=(status[:,0]>0)&(reverse[:,0]>0)&(np.linalg.norm(back[:,0]-source[:,0],axis=1)<.7)&(error[:,0]<18)
                accepted&=(pixels[:,0]>8)&(pixels[:,0]<759)&(pixels[:,1]>8)&(pixels[:,1]<503)
                accepted&=~masks[frame][pixels[:,1].clip(0,511),pixels[:,0].clip(0,767)]
                valid[ids[~accepted]]=False
                kept=ids[accepted];current[kept]=target[accepted,0]
                for j in kept:
                    observations.append(current[j].copy());frame_ids.append(frame);point_ids.append(offset+j)
                previous=frame
        print('Tracked static landmarks from frame',anchor,flush=True)
    xyz=np.array(points);sigma=np.array(point_sigma);obs=np.array(observations);fi=np.array(frame_ids);pi=np.array(point_ids)
    counts=np.bincount(pi,minlength=len(xyz)); keep=counts>=8
    remap=np.cumsum(keep)-1; valid=keep[pi]
    xyz=xyz[keep];sigma=sigma[keep];obs=obs[valid];fi=fi[valid];pi=remap[pi[valid]]
    # Hold out entire feature tracks, not random repeated observations of the same track.
    heldout=np.random.default_rng(42).random(len(xyz))<.15
    train=~heldout[pi]
    obs_train,frames,ids=obs[train],fi[train],pi[train]
    camera_indices=np.array([i for i in range(80) if i!=40]); cam_remap=np.full(80,-1);cam_remap[camera_indices]=np.arange(79)
    ncam=79*6
    def unpack(x):
        delta=np.zeros((80,6));delta[camera_indices]=x[:ncam].reshape(-1,6)
        local=Rotation.from_rotvec(delta[:,:3]).as_matrix()
        rotations=local@initial[:,:3,:3]
        translations=np.einsum('nij,nj->ni',local,initial[:,:3,3])+delta[:,3:]
        return rotations,translations,xyz+x[ncam:].reshape(-1,3),delta
    def project(rotations,translations,world,frames,ids):
        camera=np.einsum('nij,nj->ni',rotations[frames],world[ids])+translations[frames]
        projected=camera@K.T
        return projected[:,:2]/np.maximum(projected[:,2:],.05)
    def residual(x):
        r,t,p,d=unpack(x)
        reprojection=(project(r,t,p,frames,ids)-obs_train).reshape(-1)
        pose_prior=np.r_[d[camera_indices,:3].reshape(-1)/.025,d[camera_indices,3:].reshape(-1)/.04]
        point_prior=((p-xyz)/sigma[:,None]).reshape(-1)
        smooth=(d[2:]-2*d[1:-1]+d[:-2])/np.array([.01]*3+[.015]*3)
        return np.r_[reprojection,pose_prior,point_prior,smooth.reshape(-1)]
    nobs=len(ids);npoints=len(xyz);nres=nobs*2+79*6+npoints*3+78*6
    sparsity=lil_matrix((nres,ncam+npoints*3),dtype=np.int8)
    for j,(frame,point) in enumerate(zip(frames,ids)):
        if frame!=40:sparsity[2*j:2*j+2,cam_remap[frame]*6:cam_remap[frame]*6+6]=1
        sparsity[2*j:2*j+2,ncam+point*3:ncam+point*3+3]=1
    offset=nobs*2
    for j in range(79):
        sparsity[offset+j*3:offset+j*3+3,j*6:j*6+3]=1
        sparsity[offset+79*3+j*3:offset+79*3+j*3+3,j*6+3:j*6+6]=1
    offset+=79*6
    for j in range(npoints):sparsity[offset+j*3:offset+j*3+3,ncam+j*3:ncam+j*3+3]=1
    offset+=npoints*3
    for frame in range(1,79):
        for neighbor in (frame-1,frame,frame+1):
            if neighbor!=40:sparsity[offset+(frame-1)*6:offset+frame*6,cam_remap[neighbor]*6:cam_remap[neighbor]*6+6]=1
    init=np.zeros(ncam+npoints*3)
    initial_errors=np.linalg.norm(project(initial[:,:3,:3],initial[:,:3,3],xyz,fi,pi)-obs,axis=1)
    print('Bundle observations',len(obs),'landmarks',npoints,'initial median pixels',np.median(initial_errors),flush=True)
    result=least_squares(residual,init,jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=2.,x_scale='jac',max_nfev=70,verbose=1,ftol=1e-4,xtol=1e-4,gtol=1e-4)
    r,t,p,delta=unpack(result.x)
    # Validate held-out tracks without optimizing their point coordinates.
    final_errors=np.linalg.norm(project(r,t,xyz,fi,pi)-obs,axis=1)
    report={'observations':len(obs),'landmarks':npoints,'median_before_pixels':float(np.median(initial_errors)),'median_after_pixels':float(np.median(np.linalg.norm(project(r,t,p,fi,pi)-obs,axis=1))), 'heldout_tracks_before_pixels':float(np.median(initial_errors[heldout[pi]])), 'heldout_tracks_after_pixels':float(np.median(final_errors[heldout[pi]])), 'max_rotation_delta_degrees':float(np.max(np.linalg.norm(delta[:,:3],axis=1))*180/np.pi),'max_camera_translation_parameter_delta_m':float(np.max(np.linalg.norm(delta[:,3:],axis=1)))}
    print(json.dumps(report,indent=2),flush=True)
    if not report['heldout_tracks_after_pixels']<report['heldout_tracks_before_pixels']:
        raise RuntimeError('Camera refinement did not improve held-out static tracks')
    refined=np.tile(np.eye(4),(80,1,1));refined[:,:3,:3]=r;refined[:,:3,3]=t
    np.savez_compressed(ROOT/'agentos_office/agentos_office.npz',cam_c2w=np.linalg.inv(refined).astype(np.float32),intrinsic=data['intrinsic'],depths=data['depths'])
    np.savez_compressed(ROOT/'camera_refinement.npz',initial_c2w=poses,refined_c2w=np.linalg.inv(refined),landmarks=p,frame_ids=fi,point_ids=pi,observations=obs,heldout_points=heldout)
    (ROOT/'camera_refinement.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
