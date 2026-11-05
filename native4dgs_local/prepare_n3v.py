"""Full 300-frame N3V preparation with CUDA COLMAP dense initialization."""
from pathlib import Path
import concurrent.futures,subprocess,json,sqlite3,os,time,threading,traceback,importlib.util
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import pycolmap
from project import ROOT as R, CODE
raw=R/'data/N3V_raw/coffee_martini';dest=R/'data/paper/N3V/coffee_martini'
images=dest/'images';work=dest/'preprocess';first=work/'first_frames'
for p in (images,work,first):p.mkdir(parents=True,exist_ok=True)
status_path=R/'logs/paper_n3v_preparation.json'
status_path.parent.mkdir(parents=True,exist_ok=True)
state={'pid':os.getpid(),'started':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':'starting','cameras_extracted':[],
       'fps':30,'frames_per_camera':300,'raw_resolution':[2704,2028],'training_resolution':[1352,1014],
       'initialization':'COLMAP fixed-pose triangulation and CUDA geometric PatchMatch stereo + fusion from 17 training cameras at t=0',
       'pycolmap':pycolmap.__version__,'held_out_camera':'cam00','gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
mutex=threading.Lock()
def update(**values):
    with mutex:
        state.update(values);state['updated']=time.strftime('%Y-%m-%d %H:%M:%S')
        tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status_path)
        print('STATUS',json.dumps(values),flush=True)
def extract(video):
    marker=work/(video.stem+'_extracted.json')
    files=list(images.glob(video.stem+'_*.png'))
    if not (marker.exists() and len(files)==300):
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','2','-i',str(video),
            '-vsync','0','-frames:v','300','-start_number','0','-threads','2',str(images/(video.stem+'_%04d.png'))],check=True)
        files=list(images.glob(video.stem+'_*.png'));assert len(files)==300
        for idx in (0,150,299):
            with Image.open(images/f'{video.stem}_{idx:04d}.png') as im:assert im.size==(2704,2028);im.verify()
        marker.write_text(json.dumps({'frames':300,'resolution':[2704,2028]}))
    with mutex:done=list(state['cameras_extracted'])+[video.stem]
    update(cameras_extracted=done)
    return video.stem
def official_camera_poses(raw,videos):
    bounds=np.load(raw/'poses_bounds.npy')
    p=bounds[:,:15].reshape(-1,3,5);H,W,fl=p[0,:,-1]
    p=np.concatenate([p[...,1:2],p[...,0:1],-p[...,2:3],p[...,3:4]],-1)
    p=np.concatenate([p,np.tile([[[0,0,0,1]]],(len(p),1,1))],axis=1)
    p[:,0:3,1]*=-1;p[:,0:3,2]*=-1;p=p[:,[1,0,2,3],:];p[:,2,:]*=-1
    spec=importlib.util.spec_from_file_location('official_converter',CODE/'scripts/n3v2blender.py')
    converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
    up=p[:,:3,1].sum(0);up/=np.linalg.norm(up)
    rotation=np.pad(converter.rotmat(up,[0,0,1]),[0,1]);rotation[-1,-1]=1;p=rotation@p
    total=np.zeros(3);weight=0.
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            point,w=converter.closest_point_2_lines(p[i,:3,3],p[i,:3,2],p[j,:3,3],p[j,:3,2])
            if w>.01:total+=point*w;weight+=w
    p[:,:3,3]-=total/weight;p[:,:3,3]*=4/np.linalg.norm(p[:,:3,3],axis=-1).mean()
    metadata={'w':int(W),'h':int(H),'fl_x':float(fl),'fl_y':float(fl),'cx':float(W//2),'cy':float(H//2)}
    return {v.stem:p[i].tolist() for i,v in enumerate(videos)},metadata

try:
    assert pycolmap.has_cuda
    videos=sorted(raw.glob('cam*.mp4'));assert len(videos)==18
    # Compute poses directly from the original data using the released converter.
    # No dependency on the obsolete reduced-data experiment is retained.
    poses,metadata=official_camera_poses(raw,videos)
    W,H,fl=metadata['w'],metadata['h'],metadata['fl_x']
    assert (int(W),int(H))==(2704,2028)
    metadata={'w':int(W),'h':int(H),'fl_x':float(fl),'fl_y':float(fl),'cx':float(W//2),'cy':float(H//2)}
    for split in ('train','test'):
        frames=[{'file_path':f'images/{v.stem}_{k:04d}','transform_matrix':poses[v.stem],'time':k/30.}
                for v in videos if (v.stem=='cam00')==(split=='test') for k in range(300)]
        (dest/f'transforms_{split}.json').write_text(json.dumps(dict(metadata,frames=frames),indent=2))
    update(stage='extracting_first_frames')
    for v in videos:
        if v.stem=='cam00':continue
        target=first/(v.stem+'_0000.png')
        if not target.exists():
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','2','-i',str(v),'-frames:v','1','-threads','2',str(target)],check=True)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(extract,v) for v in videos]
        if not (dest/'points3d.ply').exists():
            dense=work/'dense'
            maps_ready=all(len(list((dense/'stereo'/kind).glob('*.geometric.bin')))==17 for kind in ['depth_maps','normal_maps'])
            if not maps_ready:
                database=work/'database.db'
                reader=pycolmap.ImageReaderOptions();reader.camera_model='PINHOLE';reader.camera_params=f'{fl},{fl},{W//2},{H//2}'
                opts=pycolmap.FeatureExtractionOptions();opts.num_threads=4;opts.gpu_index='0'
                update(stage='colmap_features')
                pycolmap.extract_features(database,first,camera_mode=pycolmap.CameraMode.SINGLE,reader_options=reader,extraction_options=opts,device=pycolmap.Device.cuda)
                matching=pycolmap.FeatureMatchingOptions();matching.num_threads=4;matching.gpu_index='0'
                update(stage='colmap_matching')
                pycolmap.match_exhaustive(database,matching_options=matching,device=pycolmap.Device.cuda)
                known=work/'known';known.mkdir(exist_ok=True)
                with sqlite3.connect(database) as con:rows=list(con.execute('SELECT image_id,name,camera_id FROM images ORDER BY image_id'))
                assert len(rows)==17 and all(not row[1].startswith('cam00') for row in rows)
                (known/'cameras.txt').write_text(''.join(f'{cid} PINHOLE {int(W)} {int(H)} {fl} {fl} {W//2} {H//2}\n' for cid in sorted(set(x[2] for x in rows))))
                lines=[]
                for image_id,name,camera_id in rows:
                    c2w=np.array(poses[name[:5]])@np.diag([1,-1,-1,1]);w2c=np.linalg.inv(c2w);q=Rotation.from_matrix(w2c[:3,:3]).as_quat()
                    values=[q[3],*q[:3],*w2c[:3,3]]
                    lines.append(f'{image_id} '+' '.join(map(str,values))+f' {camera_id} {name}\n\n')
                (known/'images.txt').write_text(''.join(lines));(known/'points3D.txt').write_text('')
                sparse=work/'triangulated';sparse.mkdir(exist_ok=True)
                tri=pycolmap.IncrementalPipelineOptions();tri.num_threads=4
                update(stage='colmap_triangulation')
                rec=pycolmap.triangulate_points(pycolmap.Reconstruction(known),database,first,sparse,refine_intrinsics=False,options=tri)
                assert rec.num_points3D()>1000
                update(sparse_points=rec.num_points3D(),stage='colmap_undistort')
                dense=work/'dense'
                pycolmap.undistort_images(dense,sparse,first,num_threads=4)
                pm=pycolmap.PatchMatchOptions();pm.gpu_index='0';pm.num_threads=4;pm.cache_size=8.
                update(stage='colmap_dense_stereo')
                pycolmap.patch_match_stereo(dense,options=pm)
            fusion=pycolmap.StereoFusionOptions();fusion.num_threads=4;fusion.cache_size=8.
            update(stage='colmap_fusion')
            (work/'fused').mkdir(exist_ok=True)
            rec=pycolmap.stereo_fusion(work/'fused',dense,options=fusion,output_type='bin')
            assert rec.num_points3D()>10000,'Dense initialization unexpectedly small'
            rec.export_PLY(dest/'points3d.ply')
            update(dense_points=rec.num_points3D(),stage='waiting_for_frame_extraction')
        for future in concurrent.futures.as_completed(futures):future.result()
    assert len(list(images.glob('*.png')))==5400
    update(stage='complete',train_images=5100,test_images=300,finished=time.strftime('%Y-%m-%d %H:%M:%S'))
except Exception:
    update(stage='failed',error=traceback.format_exc());raise
