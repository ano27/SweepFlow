"""Raw, current-sweep readers for ONCE, DSERT-RoLL Livox and nuScenes.

Training images are stored as target paths, separate from raw XYZI inputs.
Object annotations are not read by this training/inference package.
"""
from __future__ import annotations
from ..progress import Progress,activity,reported,iterate,detail,reused,event
import json, pickle, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from ..common import atomic_json,read_json,read_jsonl

CANON_FROM_AUTO = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])


def sec(value):
    x=float(value)
    return x/(1e9 if abs(x)>=1e17 else 1e6 if abs(x)>=1e14 else 1e3 if abs(x)>=1e11 else 1.)


def rigid(x,label='transform'):
    a=np.asarray(x,dtype=np.float64)
    if a.shape==(3,4):a=np.vstack((a,[0,0,0,1]))
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[3],[0,0,0,1],atol=1e-5):
        raise ValueError(f'{label}: expected finite homogeneous 4x4')
    R=a[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=3e-3) or not np.isclose(np.linalg.det(R),1,atol=3e-3):
        raise ValueError(f'{label}: rotation is not rigid/right handed')
    return a


def quat_transform(record):
    q=np.asarray(record['rotation'],float)
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();T[:3,3]=record['translation'];return T


def resolve_file(root,seq,raw):
    if raw is None:raise FileNotFoundError(f'Missing sensor path for {seq}')
    p=Path(str(raw).replace('\\','/')).expanduser(); root=Path(root);seq=Path(seq)
    candidates=[p] if p.is_absolute() else [root/p,seq/p]
    # Some distributed label.pkl files carry machine-specific prefixes. Only
    # the named sensor directory + filename may be relocated, never guessed.
    candidates += [seq/p.parent.name/p.name]
    found=sorted({x.resolve() for x in candidates if x.is_file()})
    if len(found)!=1:raise FileNotFoundError(f'Sensor path resolution found {len(found)} files: raw={raw}; candidates={candidates}')
    return str(found[0])


def names_file(path):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(p)
    names=[Path(s.split('#')[0].strip().rstrip('/')).name for s in p.read_text().splitlines() if s.split('#')[0].strip()]
    if len(names)!=len(set(names)):raise ValueError(f'Duplicate sequence in {p}')
    return names






def make_spec(path,K,T,name,dt=0.,state='rectified',distortion=None,dt_source='sensor_timestamp'):
    with Image.open(path) as im:w,h=im.size
    K=np.asarray(K,float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0:raise ValueError(f'Invalid K {name}')
    return {'name':name,'image_path':str(path),'width':w,'height':h,'intrinsics':K.tolist(),
            'lidar_to_camera':rigid(T,name).tolist(),'distortion':distortion,'image_state':state,
            'timestamp_delta_sec':float(dt),'time_alignment_source':dt_source}


@reported('inventory/once')
def once_inventory(cfg):
    root=Path(cfg['root']);cams=cfg['cameras'];splits=root/'ImageSets'
    if not splits.is_dir() and (root/'data'/'ImageSets').is_dir():splits=root/'data'/'ImageSets'
    test=set(names_file(splits/cfg.get('test_list','val.txt')))
    pools=[]
    for fn in cfg.get('train_lists',['train.txt','raw_small.txt']):pools.extend(names_file(splits/fn))
    assigned=set(pools)|test
    overlap=set(pools)&test
    if overlap:raise ValueError(f'ONCE train/test sequence overlap: {sorted(overlap)[:8]}')
    rows=[];errors=[]
    for seq in iterate(sorted(assigned),'once/sequences',unit='sequences',eta=False):
        candidates=[root/'data'/seq,root/'data'/'data'/seq,root/seq]
        found=[p for p in candidates if (p/(seq+'.json')).is_file()]
        if len(found)!=1:
            errors.append({'sequence':seq,'reason':'metadata_not_found_or_ambiguous','candidates':list(map(str,candidates))});continue
        folder=found[0];meta=read_json(folder/(seq+'.json'));cal=meta.get('calib',meta.get('calibration'))
        if not cal:raise ValueError(f'ONCE no calibration: {folder}')
        missing=[c for c in cams if c not in cal or not (folder/c).is_dir()]
        if missing:errors.append({'sequence':seq,'reason':'missing_camera','cameras':missing});continue
        tag=meta.get('meta_info',{});group='test' if seq in test else 'train_pool'
        for frame in iterate(meta['frames'],f'once/{seq}/frames',unit='frames'):
            detail(sequence=seq,frame=frame.get('frame_id'),accepted_so_far=len(rows),excluded_so_far=len(errors))
            fid=str(frame['frame_id']);lidar=folder/'lidar_roof'/(fid+'.bin'); specs=[]
            if not lidar.is_file():errors.append({'sequence':seq,'frame':fid,'reason':'missing_lidar'});continue
            try:
                for c in cams:
                    cp=cal[c];ip=next((folder/c/(fid+e) for e in ['.jpg','.png','.jpeg'] if (folder/c/(fid+e)).is_file()),None)
                    if ip is None:raise FileNotFoundError(f'{folder/c}/{fid}.jpg')
                    T=np.linalg.inv(rigid(cp['cam_to_velo']))
                    specs.append(make_spec(ip,cp['cam_intrinsic'],T,c,state='raw',distortion=cp.get('distortion'),dt_source='ONCE_frame_association'))
            except FileNotFoundError as e:errors.append({'sequence':seq,'frame':fid,'reason':str(e)});continue
            rows.append(dict(scene_uid=f'once:{seq}:{fid}',sequence=seq,dataset='once',official_group=group,
                 lidar_path=str(lidar.resolve()),point_dim=4,model_from_lidar=np.eye(4).tolist(),lidar_timestamp=sec(fid),
                 camera_names=cams,camera_specs=specs,weather=str(tag.get('weather','unknown')),lighting=str(tag.get('period','unknown'))))
    if errors and cfg.get('missing_policy','error')=='error':
        raise FileNotFoundError(f'ONCE incomplete requested sensor set, {len(errors)} failures; first={errors[:3]}. No camera substitution. Use missing_policy=report_exclude only for intentional partial data.')
    return rows,errors


@reported('inventory/dsert_roll')
def dsert_inventory(cfg):
    root=Path(cfg['root']);train=set(names_file(root/'split'/'train.txt'));test=set(names_file(root/'split'/'val.txt'))
    if train&test:raise ValueError('DSERT official train/test overlap')
    paths={}
    for p in iterate(root.glob('*/*/label.pkl'),'dsert/discover-labels',unit='files'):
        detail(file=str(p))
        if p.parent.name in paths:raise ValueError(f'Duplicate DSERT sequence {p.parent.name}')
        paths[p.parent.name]=p
    missing=(train|test)-paths.keys()
    if missing:raise FileNotFoundError(f'DSERT missing official sequences: {sorted(missing)[:10]}')
    A=CANON_FROM_AUTO;A4=np.eye(4);A4[:3,:3]=A;inv=np.linalg.inv(A4);rows=[]
    for seq in iterate(sorted(train|test),'dsert/sequences',unit='sequences',eta=False):
        detail(sequence=seq)
        p=paths[seq]
        with activity('dsert/read-label-pkl',file=str(p)):
            with p.open('rb') as f:meta=pickle.load(f)  # trusted local official dataset metadata
        cal=meta['meta']['calibration'];info=meta['info']
        if isinstance(info,dict):info=list(info.values())
        for i,frame in enumerate(iterate(info,f'dsert/{seq}/frames',unit='frames')):
            detail(sequence=seq,frame=i,accepted_so_far=len(rows))
            sensor=frame['sensor'];lp=resolve_file(root,p.parent,sensor['livox_path'])
            ts=frame.get('time_stamp',frame.get('timestamp',i));fid=str(frame.get('sample_idx',frame.get('frame_idx',i)))
            specs=[]
            for c,key in [('RGB_L','rgb_left_path'),('RGB_R','rgb_right_path')]:
                ip=resolve_file(root,p.parent,sensor[key]);T=rigid(cal['Livox'][c])@inv
                specs.append(make_spec(ip,cal[c]['intrinsic'],T,c,state='rectified',dt_source='DSERT_frame_association_no_per_sensor_timestamp'))
            rows.append(dict(scene_uid=f'dsert_roll:{seq}:{fid}',sequence=seq,dataset='dsert_roll',
                official_group='test' if seq in test else 'train_pool',lidar_path=lp,point_dim=4,
                model_from_lidar=A4.tolist(),lidar_timestamp=sec(ts),camera_names=['RGB_L','RGB_R'],camera_specs=specs,
                weather=str(meta['meta'].get('weather',p.parent.parent.name)),lighting=str(meta['meta'].get('light','unknown'))))
    return rows,[]


@reported('inventory/nuscenes')
def nuscenes_inventory(cfg):
    root=Path(cfg['root']);tables=root/cfg.get('version','v1.0-trainval')
    def table(name):
        with activity('nuscenes/read-table',table=name,path=str(tables/(name+'.json'))):
            return read_json(tables/(name+'.json'))
    scenes=table('scene');sample=table('sample');sd=table('sample_data')
    calibr={x['token']:x for x in table('calibrated_sensor')};poses={x['token']:x for x in table('ego_pose')};sensors={x['token']:x for x in table('sensor')}
    # Split file from official SDK (downloaded by assets stage) or explicit JSON.
    split=read_json(cfg['splits_file']);train=set(split['train']);test=set(split['val'])
    if train&test:raise ValueError('nuScenes train/val overlap')
    present={x['name'] for x in scenes};missing=(train|test)-present
    if missing and not cfg.get('allow_subset',False):raise FileNotFoundError(f'nuScenes missing {len(missing)} official scenes')
    scene={x['token']:x for x in scenes};keyframes=defaultdict(dict)
    for x in iterate(sd,'nuscenes/index-sample-data',unit='records'):
        if x['is_key_frame']:
            c=calibr[x['calibrated_sensor_token']];channel=sensors[c['sensor_token']]['channel'];keyframes[x['sample_token']][channel]=x
    rows=[];cams=cfg['cameras'];A4=np.eye(4);A4[:3,:3]=CANON_FROM_AUTO
    for s in iterate(sorted(sample,key=lambda x:(scene[x['scene_token']]['name'],x['timestamp'])),'nuscenes/build-scenes',unit='samples'):
        detail(sample=s['token'],accepted_so_far=len(rows))
        seq=scene[s['scene_token']]['name']
        if seq not in train|test:continue
        k=keyframes[s['token']]
        if any(c not in k for c in ['LIDAR_TOP']+cams):raise ValueError(f'nuScenes incomplete keyframe {s["token"]}')
        li=k['LIDAR_TOP'];Lc=quat_transform(calibr[li['calibrated_sensor_token']]);Le=quat_transform(poses[li['ego_pose_token']])
        # Canonical axes are ego-oriented with the CURRENT LiDAR origin; rotate
        # points once, and invert that same matrix in every camera transform.
        M=np.eye(4);M[:3,:3]=A4[:3,:3]@Lc[:3,:3];Minv=np.linalg.inv(M)
        world_from_lidar=Le@Lc;specs=[]
        for name in cams:
            c=k[name];cc=calibr[c['calibrated_sensor_token']]
            T=np.linalg.inv(quat_transform(cc))@np.linalg.inv(quat_transform(poses[c['ego_pose_token']]))@world_from_lidar@Minv
            ip=resolve_file(root,root,c['filename'])
            specs.append(make_spec(ip,cc['camera_intrinsic'],T,name,dt=(c['timestamp']-li['timestamp'])/1e6,state='rectified'))
        description=scene[s['scene_token']].get('description','').lower()
        rows.append(dict(scene_uid=f'nuscenes:{s["token"]}',sequence=seq,dataset='nuscenes',official_group='test' if seq in test else 'train_pool',
             lidar_path=resolve_file(root,root,li['filename']),point_dim=5,model_from_lidar=M.tolist(),lidar_timestamp=li['timestamp']/1e6,
             camera_names=cams,camera_specs=specs,weather='rain' if 'rain' in description else 'not_marked_rain',
             lighting='night' if 'night' in description else 'not_marked_night'))
    return rows,[]


def read_points(row):
    p=Path(row['lidar_path']);available=True
    if p.suffix.lower()=='.npy':
        a=np.load(p,allow_pickle=False)
        if a.dtype.names:
            fields=a.dtype.names
            if not all(k in fields for k in ('x','y','z')):raise ValueError(f'{p}: missing XYZ')
            key=next((k for k in ('intensity','reflectivity','reflectance') if k in fields),None);available=key is not None
            a=np.stack([np.asarray(a[k]).reshape(-1) for k in ('x','y','z')]+[np.asarray(a[key]).reshape(-1) if key else np.zeros(a.size)],1)
        elif a.ndim!=2 or a.shape[1]<3:raise ValueError(f'{p}: invalid point array')
        if a.shape[1]==3:available=False;a=np.c_[a,np.zeros(len(a))]
    elif p.suffix.lower()=='.bin':
        dim=int(row['point_dim']);a=np.fromfile(p,dtype=np.float32)
        if a.size%dim:raise ValueError(f'{p}: bytes do not match point_dim={dim}')
        a=a.reshape(-1,dim)
    else:raise ValueError(f'{p}: only raw .npy/.bin supported')
    a=np.asarray(a[:,:4],np.float32);good=np.isfinite(a).all(1)&(np.linalg.norm(a[:,:3],axis=1)>1e-4)
    raw_ids=np.flatnonzero(good);a=a[good].copy();M=rigid(row['model_from_lidar']);a[:,:3]=a[:,:3]@M[:3,:3].T+M[:3,3]
    if not len(a):raise ValueError(f'No valid raw points: {p}')
    return a,available,raw_ids


def partition(rows,seed,dev_fraction=.1):
    byseq=defaultdict(list)
    for x in rows:byseq[x['sequence']].append(x)
    groups=defaultdict(list)
    for seq,rr in byseq.items():
        if rr[0]['official_group']=='train_pool':groups[(rr[0]['weather'],rr[0]['lighting'])].append(seq)
    rng=np.random.default_rng(seed);dev=set()
    for names in groups.values():
        names=np.asarray(sorted(names));rng.shuffle(names)
        count=max(1,round(len(names)*dev_fraction)) if len(names)>1 else 0
        dev.update(names[:min(count,len(names)-1)])
    pool=[s for s,v in byseq.items() if v[0]['official_group']=='train_pool']
    if not dev and len(pool)>1:dev.add(sorted(pool)[int(rng.integers(len(pool)))])
    out={'train':[],'dev':[],'test':[]}
    for x in rows:
        split='test' if x['official_group']=='test' else 'dev' if x['sequence'] in dev else 'train'
        y=dict(x);y['split']=split;out[split].append(y)
    for split in out:
        out[split].sort(key=lambda x:(x['sequence'],x['lidar_timestamp'],x['scene_uid']))
        if not out[split]:raise ValueError(f'Empty {split} split; cannot invent test/development data')
        for i,x in enumerate(out[split]):x['index']=i
    sets=[{r['sequence'] for r in out[s]} for s in ('train','dev','test')]
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:raise AssertionError('Sequence leakage')
    return out




@reported('manifest')
def build_manifests(cfg):
    out=Path(cfg['workdir'])/'data'/cfg['dataset']/'manifests';done=out/'complete.json'
    spec=cfg['datasets'][cfg['dataset']];contract={'dataset':cfg['dataset'],'spec':spec,'seed':cfg['split_seed'],'image_hw':cfg['data']['image_hw']}
    if done.exists():
        if read_json(done)['contract']!=contract:raise ValueError('Manifest configuration changed; choose a fresh run root')
        event('manifest','REUSED',dataset=cfg['dataset'],path=done,counts=read_json(done).get('counts'))
        return
    detail(dataset=cfg['dataset'],phase='scan_raw_metadata_and_image_headers',root=spec['root'])
    rows,errors={'once':once_inventory,'dsert_roll':dsert_inventory,'nuscenes':nuscenes_inventory}[cfg['dataset']](spec)
    if len({r['scene_uid'] for r in rows})!=len(rows):raise ValueError('Duplicate scene_uid in raw data')
    event('manifest','SCAN_DONE',dataset=cfg['dataset'],accepted=len(rows),excluded=len(errors))
    detail(phase='partition_and_write')
    parts=partition(rows,cfg['split_seed'],spec.get('dev_fraction',.1));out.mkdir(parents=True,exist_ok=True)
    from ..common import atomic_jsonl
    for split,rr in parts.items():
        with activity('manifest/write',dataset=cfg['dataset'],split=split,rows=len(rr)):
            atomic_jsonl(out/(split+'.jsonl'),rr)
    atomic_json(done,{'state':'completed','contract':contract,'counts':{s:len(v) for s,v in parts.items()},'sequences':{s:sorted({x['sequence'] for x in v}) for s,v in parts.items()},'excluded':errors,'condition_fields':'PCD and calibration only'})
