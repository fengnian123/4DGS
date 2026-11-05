"""Check actual CUDA derivatives and compare viewer cameras with training cameras."""
from pathlib import Path
import argparse,sys,json,time,hashlib,math
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

from project import ROOT as R, CODE, prepare_environment
prepare_environment()
parser=argparse.ArgumentParser()
parser.add_argument('--code',default=str(CODE))
parser.add_argument('--label',default='corrected')
parser.add_argument('--skip-scene',action='store_true')
args=parser.parse_args();repo=Path(args.code).resolve()
sys.path.insert(0,str(repo))
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.loss_utils import ssim
from utils.image_utils import psnr

def tensor(value):return torch.tensor(value,dtype=torch.float32,device='cuda')
def toy(degree_t):
    pc=GaussianModel(3,gaussian_dim=4,time_duration=[0.,1.],rot_4d=True,force_sh_3d=False,sh_degree_t=2)
    pc.active_sh_degree=3;pc.active_sh_degree_t=degree_t
    pc._xyz=nn.Parameter(tensor([[.35,.23,3.]]))
    pc._features_dc=nn.Parameter(tensor([[[.2,.3,.4]]]))
    features=torch.zeros((1,47,3),device='cuda')
    # Only DC spatial terms at each time frequency: isolate the cosine derivative.
    features[0,15]=tensor([.10,.20,.15]);features[0,31]=tensor([.12,-.05,.07])
    pc._features_rest=nn.Parameter(features)
    pc._scaling=nn.Parameter(torch.log(tensor([[.20,.17,.15]])))
    pc._opacity=nn.Parameter(tensor([[1.]]))
    pc._rotation=nn.Parameter(tensor([[1.,0.,0.,0.]]));pc._rotation_r=nn.Parameter(tensor([[1.,0.,0.,0.]]))
    pc._t=nn.Parameter(tensor([[.23]]));pc._scaling_t=nn.Parameter(torch.log(tensor([[10.]])))
    return pc

def gradients():
    cam=Camera(0,np.eye(3),np.zeros(3),.8,.8,None,None,'gradient_probe',0,timestamp=.41,resolution=(40,40),meta_only=True).cuda()
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,env_map_res=0,debug=False)
    bg=torch.zeros(3,device='cuda');rows=[]
    for degree_t in [0,1,2]:
        pc=toy(degree_t)
        def loss():return (render(cam,pc,pipe,bg)['render']*tensor([.7,.2,.1])[:,None,None]).sum()
        loss().backward()
        probes=[('sh0',pc._features_dc,(0,0,0)),('time_center',pc._t,(0,0))]
        probes += [(f'sh{i}',pc._features_rest,(0,i-1,0)) for i in range(1,16*(degree_t+1))]
        for label,param,index in probes:
            analytical=float(param.grad[index]);values=[]
            for eps in [1e-3,5e-4]:
                with torch.no_grad():
                    original=float(param[index]);param[index]=original+eps;plus=float(loss())
                    param[index]=original-eps;minus=float(loss());param[index]=original
                values.append((plus-minus)/(2*eps))
            numerical=sum(values)/len(values)
            tolerance=0.01+0.015*max(abs(analytical),abs(numerical))
            rows.append({'degree_t':degree_t,'parameter':label,'backward':analytical,
                'finite_difference':numerical,'finite_difference_two_eps':values,
                'passed':abs(analytical-numerical)<tolerance})
    # Core 4D geometry with nonzero space/time coupling and constant color.
    pc=toy(0)
    with torch.no_grad():
        pc._rotation.copy_(tensor([[.97,.10,-.06,.13]]))
        pc._rotation_r.copy_(tensor([[.98,-.08,.03,.10]]))
        pc._scaling_t.copy_(torch.log(tensor([[.6]])))
    # Probe interior pixels: whole-image finite differences can cross the
    # rasterizer's discrete alpha cutoff and do not then measure its local derivative.
    assert float(render(cam,pc,pipe,bg)['alpha'][0,23:25,24:26].min())>.05
    def loss():return (render(cam,pc,pipe,bg)['render'][:,23:25,24:26]*tensor([.7,.2,.1])[:,None,None]).sum()
    loss().backward()
    for attr,width in [('_xyz',3),('_t',1),('_scaling',3),('_scaling_t',1),('_rotation',4),('_rotation_r',4),('_opacity',1)]:
        param=getattr(pc,attr)
        for i in range(width):
            analytical=float(param.grad[0,i]);values=[]
            for eps in [1e-3,5e-4]:
                with torch.no_grad():
                    original=float(param[0,i]);param[0,i]=original+eps;plus=float(loss())
                    param[0,i]=original-eps;minus=float(loss());param[0,i]=original
                values.append((plus-minus)/(2*eps))
            numerical=sum(values)/len(values)
            rows.append({'kind':'coupled_4d_geometry','parameter':f'{attr}[{i}]','backward':analytical,
                'finite_difference':numerical,'finite_difference_two_eps':values,
                'passed':abs(analytical-numerical)<.03+.03*max(abs(analytical),abs(numerical))})
    return rows

@torch.no_grad()
def scene_checks():
    import viewer
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import cameraList_from_camInfos
    from utils.data_utils import CameraDataset
    state=viewer.load('jumpingjacks_auditfix_full');cfg=state['cfg'];pc=state['pc']
    model=SimpleNamespace(**OmegaConf.to_container(cfg.ModelParams));model.dataloader=True
    pipe=SimpleNamespace(**OmegaConf.to_container(cfg.PipelineParams));bg=torch.zeros(3,device='cuda')
    rows=[];averages={}
    for split in ['test','train']:
        infos=readCamerasFromTransforms(model.source_path,f'transforms_{split}.json',False,time_duration=[0.,1.],dataloader=True)
        cameras=cameraList_from_camInfos(infos,1.,model);dataset=CameraDataset(cameras,False)
        indices=list(range(len(dataset))) if split=='test' else [0,49,99,149,199]
        scores=[]
        for i in indices:
            gt,cam=dataset[i];cam=cam.cuda();gt=gt.cuda()
            out=render(cam,pc,pipe,bg)['render'].clamp(0,1);scores.append(float(psnr(out,gt).mean()))
            if split=='test':
                interactive,_,_=viewer.make_camera(state,{'camera':i,'time':cam.timestamp,'yaw':0.,'pitch':0.,'zoom':1.,'width':0})
                alternate=render(interactive,pc,pipe,bg)['render'].clamp(0,1)
                rows.append({'index':i,'time':cam.timestamp,'max_view_matrix_error':float((cam.world_view_transform-interactive.world_view_transform).abs().max()),
                    'max_projection_error':float((cam.full_proj_transform-interactive.full_proj_transform).abs().max()),
                    'mean_render_error':float((out-alternate).abs().mean()),'max_render_error':float((out-alternate).abs().max())})
        averages[split]={'views':len(scores),'psnr':sum(scores)/len(scores),'per_view':scores}
    model.dataloader=False
    infos=readCamerasFromTransforms(model.source_path,'transforms_test.json',False,time_duration=[0.,1.],dataloader=False)
    exact=CameraDataset(cameraList_from_camInfos(infos,1.,model),False);scores=[]
    for gt,cam in exact:scores.append(float(psnr(render(cam.cuda(),pc,pipe,bg)['render'].clamp(0,1),gt.cuda()).mean()))
    averages['test_release_loader']={'views':len(scores),'psnr':sum(scores)/len(scores)}
    return {'checkpoint_iteration':state['iteration'],'test_vs_viewer':rows,'same_checkpoint_psnr':averages}

record={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'code':'configured Native 4DGS source','label':args.label,
    'kernel_sha256':hashlib.sha256((repo/'diff-gaussian-rasterization/cuda_rasterizer/backward.cu').read_bytes()).hexdigest(),
    'gradient_checks':gradients()}
if not args.skip_scene:record['scene_checks']=scene_checks()
out=R/'results/quality_audit';out.mkdir(exist_ok=True,parents=True)
(out/(args.label+'_numerics.json')).write_text(json.dumps(record,indent=2))
print('NUMERICS',json.dumps(record),flush=True)

if not all(row['passed'] for row in record['gradient_checks']):raise SystemExit('Finite-difference checks failed')
