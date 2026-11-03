"""Fetch public official project resources and record exact source provenance."""
import argparse, hashlib, json, time, urllib.request, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def verify_digest(data,expected):
    actual=hashlib.sha256(data).hexdigest()
    if expected and actual!=expected: raise ValueError(f'SHA256 mismatch: expected {expected}, got {actual}')
    return actual

def pinned_source(manifest,key,default_url):
    return dict(manifest[key]) if key in manifest else {'url':default_url}

def check_file(path,expected):
    with Path(path).open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if expected and actual!=expected:raise ValueError(f'SHA256 mismatch for {path}: expected {expected}, got {actual}')
    return actual

def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        print('EXISTS',path.relative_to(ROOT),path.stat().st_size,flush=True)
        return
    partial=path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            start=partial.stat().st_size if partial.exists() else 0
            req=urllib.request.Request(url,headers={'User-Agent':'AIAA3201-course-project',**({'Range':f'bytes={start}-'} if start else {})})
            with urllib.request.urlopen(req,timeout=90) as response:
                if response.status!=206: start=0
                total=int(response.headers.get('Content-Length',0))+start
                print('DOWNLOAD',path.relative_to(ROOT),start,total,flush=True)
                last=time.monotonic();done=start
                with partial.open('ab' if start else 'wb') as out:
                    while chunk:=response.read(1024*1024):
                        out.write(chunk);done+=len(chunk)
                        if time.monotonic()-last>20:
                            print('PROGRESS',path.name,round(done/1e6,1),'MB',flush=True);last=time.monotonic()
                if total and done!=total: raise IOError(f'Truncated download: {done}/{total}')
            partial.replace(path)
            return
        except Exception as exc:
            print('RETRY',attempt+1,url,str(exc),flush=True)
            if attempt==2: raise

def safe_extract(archive,dest,prefix=None):
    dest=Path(dest).resolve();dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name=member.filename
            if prefix: name='/'.join(name.split('/')[1:])
            if not name: continue
            target=(dest/name).resolve()
            if not target.is_relative_to(dest): raise ValueError('Unsafe zip path')
            if member.is_dir(): target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(member) as source,target.open('wb') as out:
                    import shutil;shutil.copyfileobj(source,out)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['small','nyu','repos','weights','all'],default='all');args=parser.parse_args()
    manifest_path=ROOT/'data/download_manifest.json'
    lock_path=ROOT/'resources.lock.json'
    manifest=json.loads(lock_path.read_text()) if lock_path.exists() else (json.loads(manifest_path.read_text()) if manifest_path.exists() else {})
    files={
      'small':[
       ('https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip','data/downloads/MiddEval3-data-Q.zip'),
       ('https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip','data/downloads/MiddEval3-GT0-Q.zip'),
       ('https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat','data/nyu/splits.mat')],
      'nyu':[('https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat','data/nyu/nyu_depth_v2_labeled.mat')],
      'weights':[
       ('https://huggingface.co/lpiccinelli/unidepth-v2-vits14/resolve/main/config.json','checkpoints/unidepth-v2-vits14/config.json'),
       ('https://huggingface.co/lpiccinelli/unidepth-v2-vits14/resolve/main/model.safetensors','checkpoints/unidepth-v2-vits14/model.safetensors'),
       ('https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth','checkpoints/depth_anything_v2_vits.pth'),
       ('https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth','checkpoints/video_depth_anything_vits.pth')]}
    if args.group in ('repos','all'):
        for repo in ['lpiccinelli-eth/UniDepth','DepthAnything/Depth-Anything-V2','DepthAnything/Video-Depth-Anything','cvpr-org/author-kit']:
            name=repo.split('/')[1];dest=ROOT/'third_party'/name
            if dest.exists(): continue
            record=pinned_source(manifest,f'third_party/{name}',f'https://codeload.github.com/{repo}/zip/main')
            commit=record.get('commit','main');url=record['url']
            archive=ROOT/'data/downloads'/f'{name}-{commit}.zip'
            download(url,archive)
            checksum=check_file(archive,record.get('archive_sha256'))
            safe_extract(archive,dest,True)
            manifest[f'third_party/{name}']={**record,'url':url,'commit':commit,'archive_sha256':checksum}
            manifest_path.parent.mkdir(parents=True,exist_ok=True);manifest_path.write_text(json.dumps(manifest,indent=2))
    for group,items in files.items():
        if args.group not in (group,'all'): continue
        for url,relative in items:
            record=pinned_source(manifest,relative,url);url=record['url']
            path=ROOT/relative;download(url,path)
            digest=check_file(path,record.get('sha256'))
            manifest[relative]={'url':url,'bytes':path.stat().st_size,'sha256':digest}
            manifest_path.parent.mkdir(parents=True,exist_ok=True);manifest_path.write_text(json.dumps(manifest,indent=2))
            if path.suffix=='.zip': safe_extract(path,ROOT/'data/middlebury')
    print('DONE',args.group,flush=True)

if __name__=='__main__': main()
