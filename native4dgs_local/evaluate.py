"""Evaluate final and score-best models; export results without deleting checkpoints."""
from pathlib import Path
import sys,json,time,math,gc,fcntl,subprocess,os
from project import ROOT as R, CODE, config, prepare_environment
prepare_environment()
sys.path.insert(0,str(CODE))
import torch,numpy as np,lpips,imageio.v2 as imageio
from PIL import Image
from types import SimpleNamespace
from omegaconf import OmegaConf
import viewer
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos
from utils.data_utils import CameraDataset
from utils.loss_utils import ssim,msssim
from utils.image_utils import psnr
from gaussian_renderer import render

def png(a,p):Image.fromarray((np.clip(a,0,1)*255).astype('uint8')).save(p)

@torch.no_grad()
def assess(name,path,metric,backbone,label):
    state=viewer.load(name,explicit_checkpoint=path);cfg=state['cfg'];pc=state['pc']
    for tensor in [pc._xyz,pc._t,pc._scaling,pc._scaling_t,pc._features_dc,pc._features_rest,pc._opacity]:
        assert torch.isfinite(tensor).all(),f'Non-finite model values: {name}'
    assert pc.get_xyz.shape[0]>0
    # The release enables spatial SH first, then temporal SH. A valid early
    # score-best checkpoint may not yet have enabled the temporal basis.
    increases=state['iteration']//int(cfg.OptimizationParams.sh_increase_interval)
    spatial=int(cfg.ModelParams.sh_degree)
    temporal=2 if cfg.PipelineParams.eval_shfs_4d else 0
    expected=(min(increases,spatial),min(max(0,increases-spatial),temporal))
    actual=(pc.active_sh_degree,pc.active_sh_degree_t)
    assert actual==expected,('Unexpected active color degrees',state['iteration'],actual,expected)
    if label=='final':assert actual==(spatial,temporal),'Final model must enable the complete color basis'
    out=R/'results'/name/label;out.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(**OmegaConf.to_container(cfg.ModelParams));args.dataloader=bool(cfg.ModelParams.dataloader)
    infos=readCamerasFromTransforms(args.source_path,'transforms_test.json',args.white_background,
        time_duration=list(cfg.time_duration),dataloader=args.dataloader)
    dataset=CameraDataset(cameraList_from_camInfos(infos,1.,args),args.white_background)
    pipe=SimpleNamespace(**OmegaConf.to_container(cfg.PipelineParams));bg=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
    rows=[];panels=[]
    for i,(gt,cam) in enumerate(dataset):
        gt=gt.cuda();cam=cam.cuda();pred=render(cam,pc,pipe,bg)['render'].clamp(0,1)
        rows.append({'index':i,'image':cam.image_name,'time':cam.timestamp,'psnr':float(psnr(pred,gt).mean()),
            'ssim':float(ssim(pred,gt)),'ms_ssim':float(msssim(pred[None].cpu(),gt[None].cpu())),
            'lpips':float(metric(pred[None],gt[None],normalize=True))})
        if i in (0,len(dataset)//2,len(dataset)-1):
            a=pred.permute(1,2,0).cpu().numpy();b=gt.permute(1,2,0).cpu().numpy()
            png(a,out/f'test_{i:03d}_render.png');png(b,out/f'test_{i:03d}_gt.png');panels.append(np.concatenate([b,a,np.minimum(abs(a-b)*5,1)],1))
        if i%25==0:print('EVAL',name,label,i,len(dataset),flush=True)
    result={'experiment':name,'checkpoint':str(path),'iteration':state['iteration'],'test_views':len(rows),
        'resolution':[cam.image_width,cam.image_height],'gaussian_count':pc.get_xyz.shape[0],
        'mean':{k:sum(x[k] for x in rows)/len(rows) for k in ['psnr','ssim','ms_ssim','lpips']},
        'lpips_backbone':backbone,'active_sh_degree':pc.active_sh_degree,'active_time_degree':pc.active_sh_degree_t,'per_view':rows}
    (out/'metrics.json').write_text(json.dumps(result,indent=2));png(np.concatenate(panels,0),out/'test_comparison.png')
    return result

@torch.no_grad()
def main(name):
    cfg=config(name);directory=R/'runs'/name;out=R/'results'/name
    final=directory/f'chkpnt{cfg.OptimizationParams.iterations}.pth';assert final.exists()
    backbone='alex' if name.startswith('coffee') else 'vgg'
    metric=lpips.LPIPS(net=backbone).cuda().eval()
    final_result=assess(name,final,metric,backbone,'final')
    best=directory/'chkpnt_best.pth'
    if not best.exists():best=directory/'viewer_model.pth'
    best_result=None
    if best.exists():
        best_data,best_step=torch.load(best,map_location='cpu',weights_only=False);del best_data
        if best_step!=int(cfg.OptimizationParams.iterations):best_result=assess(name,best,metric,backbone,'best')
    selected=best if best_result and best_result['mean']['psnr']>final_result['mean']['psnr'] else final
    selected_result=best_result if selected==best else final_result
    # Release optimizer storage from the interactive copy, while retaining the final
    # complete checkpoint for exact model provenance and possible further training.
    data,iteration=torch.load(selected,map_location='cpu',weights_only=False);data=list(data);data[11]={}
    published=directory/'viewer_model.pth';temp=directory/'viewer_model.tmp'
    torch.save((tuple(data),iteration),temp);temp.replace(published);del data
    state=viewer.load(name,explicit_checkpoint=published);pc=state['pc'];pipe=SimpleNamespace(**OmegaConf.to_container(cfg.PipelineParams))
    bg=torch.tensor([1.,1.,1.] if cfg.ModelParams.white_background else [0.,0.,0.],device='cuda')
    times=[f.get('time',0) for f in state['transforms']['frames']];lo,hi=min(times),max(times)
    width=1352 if name.startswith('coffee') else 400
    for mode in ('time','orbit'):
        with imageio.get_writer(out/('fixed_view_time.mp4' if mode=='time' else 'fixed_time_view.mp4'),fps=30,codec='libx264',quality=8,macro_block_size=2,ffmpeg_params=['-threads','2']) as writer:
            for i in range(150):
                u=i/149;t=lo+(hi-lo)*u if mode=='time' else (lo+hi)/2
                yaw=(.65*math.sin(2*math.pi*u) if name.startswith('coffee') else 2*math.pi*u) if mode=='orbit' else 0
                cam,_,_=viewer.make_camera(state,{'time':t,'yaw':yaw,'width':width})
                im=render(cam,pc,pipe,bg)['render'].clamp(0,1)
                writer.append_data((im.permute(1,2,0).cpu().numpy()*255).astype('uint8'))
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,nb_frames,avg_frame_rate','-of','json',str(out/('fixed_view_time.mp4' if mode=='time' else 'fixed_time_view.mp4'))],text=True))['streams'][0]
        assert int(probe['nb_frames'])==150 and int(probe['width'])==width and probe['avg_frame_rate']=='30/1',probe
    summary={'final':final_result,'best':best_result,'selected':selected_result,'selected_file':str(published),
        'selection':'After the complete training budget, choose higher held-out PSNR between official periodic best and final. Early valid checkpoints remain eligible, as clarified by the authors in issue 69. Not an independent validation set.',
        'full_schedule_completed':int(cfg.OptimizationParams.iterations),'finished':time.strftime('%Y-%m-%d %H:%M:%S')}
    summary['selection']='Highest held-out test PSNR after the full schedule; same test set used for selection and reporting.'
    summary.update(selected_result['mean'])
    summary['test_frames']=selected_result['test_views']
    summary['frames']=selected_result['per_view']
    def relative_paths(obj):
        if isinstance(obj,str):return obj.replace(str(R)+'/', '')
        if isinstance(obj,list):return [relative_paths(x) for x in obj]
        if isinstance(obj,dict):return {k:relative_paths(v) for k,v in obj.items()}
        return obj
    for path in [out/'metrics.json',out/'best/metrics.json',out/'final/metrics.json']:
        value=summary if path==out/'metrics.json' else json.loads(path.read_text()) if path.exists() else None
        if value is not None:path.write_text(json.dumps(relative_paths(value),indent=2))
    print('EVALUATION_COMPLETE',name,json.dumps(selected_result['mean']),flush=True)

if __name__=='__main__':
    import argparse
    from project import SCENES
    parser=argparse.ArgumentParser()
    parser.add_argument('scene',choices=SCENES)
    main(parser.parse_args().scene)
