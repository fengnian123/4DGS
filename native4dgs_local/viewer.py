"""Loopback-only native CUDA renderer for the experiment viewer."""
import os,sys,json,math,io,base64,gc,time,threading,re
from pathlib import Path
from types import SimpleNamespace
from project import ROOT, CODE, PACKAGE, config, definitions, prepare_environment
prepare_environment()
sys.path.insert(0,str(CODE))
import numpy as np
import torch
from PIL import Image
from flask import Flask,request,jsonify,send_from_directory
from omegaconf import OmegaConf
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
app=Flask(__name__)
lock=threading.Lock()
loaded={}

def experiments():
    return definitions()

def checkpoint(name):
    directory=ROOT/'runs'/name
    if (directory/'viewer_model.pth').exists():
        return directory/'viewer_model.pth'
    if (directory/'preview_model.pth').exists():
        return directory/'preview_model.pth'
    candidates=list(directory.glob('chkpnt[0-9]*.pth'))
    if candidates:return max(candidates,key=lambda p:int(p.stem[6:]))
    p=directory/'chkpnt_best.pth'
    return p if p.exists() else None

def load(name,explicit_checkpoint=None):
    definitions=experiments()
    if name not in definitions:raise ValueError('Unknown experiment')
    path=Path(explicit_checkpoint) if explicit_checkpoint else checkpoint(name)
    if path is not None and not path.resolve().is_relative_to((ROOT/'runs'/name).resolve()):
        raise ValueError('Checkpoint must belong to this experiment')
    if path is None:raise ValueError('This experiment has no completed checkpoint yet')
    cache_key=(name,str(path),path.stat().st_mtime)
    if loaded.get('key')==cache_key:return loaded
    loaded.clear();gc.collect();torch.cuda.empty_cache()
    cfg=config(name)
    pc=GaussianModel(cfg.ModelParams.sh_degree,gaussian_dim=cfg.gaussian_dim,
        time_duration=list(cfg.time_duration),rot_4d=cfg.rot_4d,force_sh_3d=cfg.force_sh_3d,
        sh_degree_t=2 if cfg.PipelineParams.eval_shfs_4d else 0)
    # Only our own checkpoints in the isolated experiment directory are loaded.
    # Stage full training checkpoints on CPU, then transfer only model tensors.
    # Adam moments are not used for rendering and can triple peak GPU memory.
    data,iteration=torch.load(path,map_location='cpu',weights_only=False)
    data=list(data);data[11 if pc.gaussian_dim==4 else 10]={}
    data=tuple(value.detach().cuda() if torch.is_tensor(value) else value for value in data)
    pc.restore(data,None)
    del data
    source=Path(cfg.ModelParams.source_path)
    transforms=json.loads((source/'transforms_test.json').read_text())
    loaded.update(key=cache_key,pc=pc,cfg=cfg,iteration=iteration,transforms=transforms,source=source)
    return loaded

def axis_angle(axis,angle):
    axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    x,y,z=axis
    K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)

def make_camera(state,params):
    tr=state['transforms'];frames=tr['frames']
    idx=max(0,min(len(frames)-1,int(params.get('camera',0))))
    frame=frames[idx]
    src=state['source']/(frame['file_path'].removeprefix('./')+'.png')
    with Image.open(src) as im:ow,oh=im.size
    requested=int(params.get('width',640))
    if requested==0:requested=round(ow/max(1,int(state['cfg'].ModelParams.resolution)))
    w=max(160,min(1920,requested));h=round(w*oh/ow)
    fx=tr.get('fl_x',ow/(2*math.tan(tr.get('camera_angle_x',math.pi/3)/2)))
    fy=tr.get('fl_y',fx)
    fx=frame.get('fl_x',fx)*w/ow;fy=frame.get('fl_y',fy)*h/oh
    cx=frame.get('cx',tr.get('cx',ow/2))*w/ow
    cy=frame.get('cy',tr.get('cy',oh/2))*h/oh
    pose=np.array(frame['transform_matrix'],dtype=float)
    yaw=float(params.get('yaw',0));pitch=np.clip(float(params.get('pitch',0)),-1.45,1.45)
    zoom=np.clip(float(params.get('zoom',1)),0.25,4)
    rotate=axis_angle([0,0,1],yaw)@axis_angle(pose[:3,0],pitch)
    center=pose[:3,3].copy()
    pose[:3,:3]=rotate@pose[:3,:3]
    pose[:3,3]=rotate@(center*zoom)
    pose[:3,3]+=pose[:3,0]*float(params.get('panx',0))+pose[:3,1]*float(params.get('pany',0))
    pose[:3,1:3]*=-1
    w2c=np.linalg.inv(pose)
    t=float(np.clip(float(params.get('time',0)),*state['cfg'].time_duration))
    cam=Camera(0,w2c[:3,:3].T,w2c[:3,3],2*math.atan(w/(2*fx)),2*math.atan(h/(2*fy)),
        None,None,'interactive',0,timestamp=t,cx=cx,cy=cy,fl_x=fx,fl_y=fy,
        resolution=(w,h),meta_only=True).cuda()
    return cam,src,frame

@torch.no_grad()
def render_frame(params):
    name=str(params['experiment']);version=str(params.get('checkpoint_view','best'))
    if version not in ('best','final'):raise ValueError('Unknown checkpoint view')
    explicit=None
    if version=='final':
        definition=experiments().get(name)
        if not definition:raise ValueError('Unknown experiment')
        explicit=ROOT/'runs'/name/f'chkpnt{definition["steps"]}.pth'
        if not explicit.exists():raise ValueError('Final checkpoint is not available')
    state=load(name,explicit_checkpoint=explicit);cfg=state['cfg'];pc=state['pc']
    cam,src,frame=make_camera(state,params)
    pipe=SimpleNamespace(**OmegaConf.to_container(cfg.PipelineParams))
    bg=torch.tensor([1.,1.,1.] if cfg.ModelParams.white_background else [0.,0.,0.],device='cuda')
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();result=render(cam,pc,pipe,bg);end.record();torch.cuda.synchronize()
    ms=start.elapsed_time(end)
    mode=params.get('mode','color')
    image=result['render'].clamp(0,1)
    if mode=='alpha':image=result['alpha'].clamp(0,1).repeat(3,1,1)
    elif mode=='depth':
        valid=result['alpha'][0]>.1
        depth=result['depth'][0]
        vals=depth[valid]
        if vals.numel():
            lo,hi=torch.quantile(vals,torch.tensor([.02,.98],device='cuda'))
            d=((depth-lo)/(hi-lo+1e-8)).clamp(0,1)
            image=torch.stack([1-d,torch.sin(d*math.pi),d])*valid[None]
    array=(image.permute(1,2,0).cpu().numpy()*255).astype('uint8')
    im=Image.fromarray(array)
    if mode=='compare':
        with Image.open(src) as raw:
            raw=raw.convert('RGBA');background=Image.new('RGBA',raw.size,(255,255,255,255) if cfg.ModelParams.white_background else (0,0,0,255))
            gt=Image.alpha_composite(background,raw).convert('RGB').resize(im.size,Image.Resampling.LANCZOS)
        combined=Image.new('RGB',(im.width*2,im.height));combined.paste(gt);combined.paste(im,(im.width,0));im=combined
    buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=88)
    active=int((pc.get_marginal_t(cam.timestamp)>.05).sum()) if pc.gaussian_dim==4 else pc.get_xyz.shape[0]
    return {'image':base64.b64encode(buffer.getvalue()).decode(),'render_ms':round(ms,2),
        'points':pc.get_xyz.shape[0],'active_time_points':active,'iteration':state['iteration'],
        'time':cam.timestamp,'reference_time':frame.get('time',0),'width':im.width,'height':im.height,'checkpoint_view':version}

@app.get('/api/models')
@app.get('/models')
def models():
    out=[]
    for name,info in experiments().items():
        complete=(ROOT/'runs'/name/f'chkpnt{info["steps"]}.pth').exists()
        preview=(ROOT/'runs'/name/'preview_model.pth').exists() and bool(info.get('preview_iteration'))
        if info.get('role') in ('paper_full','auditfix_full','release_reference') and not (complete or preview):continue
        cp=checkpoint(name)
        if cp:
            cfg=config(name)
            tr=json.loads((Path(cfg.ModelParams.source_path)/'transforms_test.json').read_text())
            versions=[{'id':'best','label':'最佳测试结果' if info.get('role')=='auditfix_full' else '当前参考存档'}]
            display=dict(info)
            if preview and not complete:
                display['label']=info['label'].replace('完整训练','训练中预览')
                versions=[{'id':'best','label':f'训练中预览 · {info["preview_iteration"]:,} 步'}]
            final=ROOT/'runs'/name/f'chkpnt{info.get("steps",0)}.pth'
            if final.exists():versions.append({'id':'final','label':'训练最后一步'})
            out.append(dict(id=name,**display,training_complete=complete,time_range=list(cfg.time_duration),checkpoint_views=versions,
                cameras=[{'id':i,'name':Path(f['file_path']).name,'time':f.get('time',0)} for i,f in enumerate(tr['frames'])]))
    return jsonify(out)

@app.get('/api/progress')
@app.get('/progress')
def progress():
    rows=[]
    for name,info in experiments().items():
        complete=(ROOT/'runs'/name/f'chkpnt{info["steps"]}.pth').exists()
        rows.append({'id':name,'label':info['label'],'target_steps':info['steps'],
                     'step':info['steps'] if complete else 0,'stage':'complete' if complete else 'preparing'})
    return jsonify(rows)

@app.post('/api/render')
@app.post('/render')
def render_api():
    try:
        with lock:return jsonify(render_frame(request.get_json()))
    except Exception as e:
        app.logger.exception('Rendering failed')
        return jsonify(error=str(e)),400

@app.get('/health')
def health():return jsonify(ok=True,device=torch.cuda.get_device_name())

@app.get('/')
def frontend():return send_from_directory(PACKAGE/'viewer','index.html')

@app.get('/app.js')
def frontend_script():return send_from_directory(PACKAGE/'viewer','app.js')

if __name__=='__main__':
    import argparse
    from werkzeug.serving import make_server
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    server=make_server('127.0.0.1',args.port,app,threaded=False)
    print(f'Native 4DGS viewer: http://127.0.0.1:{server.server_port}/',flush=True)
    server.serve_forever()
