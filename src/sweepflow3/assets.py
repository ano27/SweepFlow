"""Explicit, selective asset download. Never downloads SD3.5 or datasets."""
from __future__ import annotations
from .progress import Progress,activity,reported,iterate,detail,reused,event
import inspect,json,os,shutil,tarfile,time,urllib.request,zipfile
from pathlib import Path
from .common import atomic_json,read_json

SCALR_URL='https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-pretrained.tar.gz'
DINO_URL='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'

@reported('asset-download')
def download(url,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.is_file() and p.stat().st_size:
        event('asset-download','REUSED',file=p.name,bytes=p.stat().st_size)
        return p
    part=p.with_name(p.name+'.part');offset=part.stat().st_size if part.exists() else 0
    headers={'User-Agent':'SweepFlow/1.0'}
    if offset:headers['Range']=f'bytes={offset}-'
    detail(file=p.name,phase='connect',resume_bytes=offset)
    req=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(req,timeout=120) as r:
        append=offset>0 and r.status==206
        if not append:offset=0
        typ=r.headers.get('Content-Type','')
        if 'text/html' in typ:raise RuntimeError(f'Download returned HTML, not weights/source: {url}')
        total=int(r.headers.get('Content-Length',0))+offset;size=offset;t0=time.monotonic()
        with Progress('download/'+p.name,total/1e6 if r.headers.get('Content-Length') else None,initial=offset/1e6,unit='MB') as progress, part.open('ab' if append else 'wb') as f:
            while True:
                b=r.read(4*1024*1024)
                if not b:break
                f.write(b);size+=len(b)
                progress.advance(len(b)/1e6)
        if total and size!=total:raise IOError(f'Incomplete download {size}/{total}: {p}')
    os.replace(part,p);return p

@reported('asset-extract')
def safe_extract(archive,out):
    detail(archive=str(archive),phase='validate_and_extract')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def check(name):
        target=(out/name).resolve()
        if not target.is_relative_to(out.resolve()):raise ValueError('Unsafe archive member path')
    if str(archive).endswith('.zip'):
        with zipfile.ZipFile(archive) as z:
            for x in z.infolist():
                check(x.filename)
                if ((x.external_attr>>16)&0o170000)==0o120000:raise ValueError('Source archive contains a symlink')
            z.extractall(out)
    else:
        with tarfile.open(archive) as z:
            for x in z.getmembers():
                check(x.name)
                if not (x.isfile() or x.isdir()):raise ValueError('Weight archive contains a link or special file')
            if 'filter' in inspect.signature(z.extractall).parameters:z.extractall(out,filter='data')
            else:z.extractall(out)

@reported('asset-source')
def source_repo(root,organization,repository,branch):
    dest=root/repository
    if (dest/'UPSTREAM.json').is_file():return dest
    api=f'https://api.github.com/repos/{organization}/{repository}/commits/{branch}'
    req=urllib.request.Request(api,headers={'User-Agent':'SweepFlow'})
    with urllib.request.urlopen(req,timeout=60) as r:commit=json.load(r)['sha']
    archive=download(f'https://codeload.github.com/{organization}/{repository}/zip/{commit}',root/f'{repository}-{commit}.zip')
    staging=root/(repository+'.unpack');safe_extract(archive,staging)
    children=[p for p in staging.iterdir() if p.is_dir()]
    if len(children)!=1:raise ValueError('Unexpected source archive structure')
    if dest.exists():raise FileExistsError(f'Incomplete source directory; inspect before reusing: {dest}')
    shutil.move(str(children[0]),dest);shutil.rmtree(staging)
    atomic_json(dest/'UPSTREAM.json',{'repository':f'{organization}/{repository}','commit':commit})
    return dest

@reported('assets-prepare')
def prepare_assets(root, *, teacher=True, point_weights=True):
    root=Path(root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    metadata=root/'assets.json'
    result=read_json(metadata) if metadata.is_file() else {}
    def present(key):
        if key not in result:return False
        p=Path(result[key]).expanduser()
        return (p if p.is_absolute() else root/p).exists()
    if not present('waffle_source'):
        result['waffle_source']=source_repo(root,'valeoai','WaffleIron','master').name
    if point_weights and not present('scalr_weights'):
        weight=root/'scalr_wi48_768.pth'
        if not weight.is_file():
            archive=download(SCALR_URL,root/'ScaLR_pretrained.tar.gz')
            staging=root/'scalr_extract';safe_extract(archive,staging)
            candidates=list(staging.rglob('model.pth'))
            if len(candidates)!=1:raise ValueError(f'Expected one pretraining model.pth, found {candidates}')
            shutil.move(str(candidates[0]),weight);shutil.rmtree(staging)
        result.update(scalr_weights=weight.name,scalr_kind='annotation_free_pretraining_not_segmentation',scalr_url=SCALR_URL)
    if teacher:
        if not present('dino_source'):
            result['dino_source']=source_repo(root,'facebookresearch','dinov2','main').name
        if not present('dino_weights'):
            result['dino_weights']=download(DINO_URL,root/'dinov2_vitl14_pretrain.pth').name
        result['dino_url']=DINO_URL
    atomic_json(metadata,result)
    return result


def asset_paths(root, *, teacher=False, point_weights=True):
    """The trained point encoder does not require the RGB teacher assets."""
    root=Path(root).expanduser().resolve()
    p=root/'assets.json'
    if not p.is_file():
        raise FileNotFoundError(f'{p}: run sweepflow assets or supply local asset paths.')
    a=read_json(p)
    required=('waffle_source',)+(('scalr_weights',) if point_weights else ())+(('dino_source','dino_weights') if teacher else ())
    for key in ('waffle_source','scalr_weights','dino_source','dino_weights'):
        if key not in a:
            if key in required:raise ValueError(f'Missing {key} in {p}')
            continue
        value=Path(a[key]).expanduser()
        a[key]=str((root/value).resolve()) if not value.is_absolute() else str(value.resolve())
        if key in required and not Path(a[key]).exists():
            raise FileNotFoundError(f'Missing {key}: {a[key]}')
    return a
