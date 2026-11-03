"""Official UniDepthV2 RGB-only inference; caches raw float32 depth in meters."""
from _bootstrap import ROOT,add_repo
add_repo('UniDepth')
import argparse,json,time,platform,hashlib
import numpy as np
import h5py,torch
from PIL import Image
from depth_project.protocol import load_splits

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['all','train','test'],default='all')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--resolution-level',type=int,default=-1)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this recorded experiment')
    from unidepth.models import UniDepthV2
    checkpoint=ROOT/'checkpoints/unidepth-v2-vits14'
    model=UniDepthV2.from_pretrained(str(checkpoint)).cuda().eval()
    if args.resolution_level>=0: model.resolution_level=args.resolution_level
    shapes=[]
    handle=model.pixel_encoder.register_forward_pre_hook(lambda m,a: shapes.append(list(a[0].shape)))
    out=ROOT/'results/nyu';out.mkdir(parents=True,exist_ok=True)
    cache=out/'predictions';cache.mkdir(exist_ok=True)
    with (checkpoint/'model.safetensors').open('rb') as f: weight_hash=hashlib.file_digest(f,'sha256').hexdigest()
    identity={'model':'lpiccinelli/unidepth-v2-vits14','weight_sha256':weight_hash,
              'resolution_level':args.resolution_level,'precision':'official infer CUDA FP16 autocast',
              'rgb_only':True,'torch':torch.__version__,'device':torch.cuda.get_device_name(),
              'platform':platform.platform(),'source_shape':[480,640]}
    meta=out/'run_config.json'
    if meta.exists() and json.loads(meta.read_text())!=identity:
        raise ValueError('Existing predictions use different settings; use a separate results directory')
    meta.write_text(json.dumps(identity,indent=2))
    if args.demo:
        rgb=np.array(Image.open(ROOT/'third_party/UniDepth/assets/demo/rgb.png').convert('RGB'))
        tensor=torch.from_numpy(rgb.copy()).permute(2,0,1)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        pred=model.infer(tensor);torch.cuda.synchronize()
        d=pred['depth'][0,0].float().cpu().numpy()
        if not np.isfinite(d).all() or (d<=0).any(): raise ValueError('Invalid smoke prediction')
        np.save(out/'smoke_depth.npy',d)
        print({'shape':d.shape,'min_m':float(d.min()),'max_m':float(d.max()),
               'seconds':time.perf_counter()-start,'peak_MB':torch.cuda.max_memory_allocated()/1e6,'input_shapes':shapes},flush=True)
        return
    train,test=load_splits(ROOT/'data/nyu/splits.mat')
    ids=np.concatenate([train,test]) if args.split=='all' else (train if args.split=='train' else test)
    if args.limit: ids=ids[:args.limit]
    with h5py.File(ROOT/'data/nyu/nyu_depth_v2_labeled.mat','r') as data:
        if data['images'].shape!=(1449,3,640,480) or data['depths'].shape!=(1449,640,480):
            raise ValueError(f'Unexpected NYU arrays: {data["images"].shape}, {data["depths"].shape}')
        rgb=data['images'][int(ids[0])].transpose(2,1,0).copy()
        tensor=torch.from_numpy(rgb).permute(2,0,1)
        with torch.inference_mode():
            for _ in range(2): model.infer(tensor)
            torch.cuda.synchronize()
            for n,i in enumerate(ids):
                i=int(i);path=cache/f'{i:04d}.npy'
                if path.exists():
                    d=np.load(path,mmap_mode='r')
                    if d.shape!=(480,640) or d.dtype!=np.float32 or not np.isfinite(d).all():
                        raise ValueError(f'Invalid cached prediction {path}')
                    continue
                rgb=data['images'][i].transpose(2,1,0).copy()
                tensor=torch.from_numpy(rgb).permute(2,0,1)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter();prediction=model.infer(tensor);torch.cuda.synchronize()
                seconds=time.perf_counter()-start
                d=prediction['depth'][0,0].float().cpu().numpy()
                if d.shape!=(480,640):
                    import cv2;d=cv2.resize(d,(640,480),interpolation=cv2.INTER_LINEAR)
                if not np.isfinite(d).all() or np.any(d<=0): raise ValueError(f'Invalid depth: {i}')
                temp=path.with_suffix('.tmp.npy');np.save(temp,d.astype(np.float32));temp.replace(path)
                record={'id':i,'split':'train' if i in set(train) else 'test','seconds':seconds,
                        'peak_allocated_MB':torch.cuda.max_memory_allocated()/1e6,
                        'peak_reserved_MB':torch.cuda.max_memory_reserved()/1e6,
                        'network_input_shape':shapes[-1]}
                with (out/'timing.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                del prediction
                if n%20==0 or n==len(ids)-1: print(f'NYU {n+1}/{len(ids)} id={i} {seconds:.3f}s peak={record["peak_allocated_MB"]:.0f}MB',flush=True)
    handle.remove()
    print('NYU INFERENCE COMPLETE',len(ids),flush=True)

if __name__=='__main__':main()
