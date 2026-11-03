"""Same-frame official image/video relative-depth comparison and recorded demo."""
from _bootstrap import ROOT,add_repo
add_repo('Depth-Anything-V2');add_repo('Video-Depth-Anything')
import argparse,gc,json,time
import cv2,numpy as np,torch
from depth_project.protocol import fixed_range,validate_clip_length

def colorize(depth,limits):
    lo,hi=limits
    normal=np.clip((depth-lo)/max(hi-lo,1e-8),0,1)
    return cv2.cvtColor(cv2.applyColorMap(np.round(normal*255).astype(np.uint8),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)

def panel(image,label):
    bar=np.full((38,image.shape[1],3),22,dtype=np.uint8)
    cv2.putText(bar,label,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.56,(245,245,245),1,cv2.LINE_AA)
    return np.concatenate([bar,image],axis=0)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input-size',type=int,default=518)
    parser.add_argument('--allow-short-clip',action='store_true',help='Disclose and process all real frames if the official clip contains fewer than 100')
    args=parser.parse_args()
    torch.set_num_threads(4)
    from utils.dc_utils import read_video_frames
    source=ROOT/'third_party/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4'
    frames,fps=read_video_frames(str(source),100,-1,640)
    count=validate_clip_length(len(frames),args.allow_short_clip)
    out=ROOT/'results/video';out.mkdir(parents=True,exist_ok=True)
    config={'input_size':args.input_size,'max_res':640,'requested_frames':100,'frames':count,'fps':float(fps),
            'short_source_disclosed':count<100,
            'source':'davis_rollercoaster.mp4','frame_shape':list(frames[0].shape),
            'precision':'CUDA FP16 autocast','device':torch.cuda.get_device_name(),
            'torch':torch.__version__,'relative_depth':True}
    cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:
        if (out/'vda_depths.npz').exists() or (out/'dav2_depths.npz').exists():
            raise ValueError('Cached depths have different settings; use a new output directory')
    cp.write_text(json.dumps(config,indent=2))
    models={'vda':('video_depth_anything_vits.pth','Video Depth Anything-S'),
            'dav2':('depth_anything_v2_vits.pth','Depth Anything V2-S')}
    arrays={};stats={}
    for name,(weight,label) in models.items():
        saved=out/f'{name}_depths.npz';statfile=out/f'{name}_timing.json'
        if saved.exists() and statfile.exists():
            arrays[name]=np.load(saved)['depths'];stats[name]=json.loads(statfile.read_text());continue
        if name=='vda':
            from video_depth_anything.video_depth import VideoDepthAnything
            model=VideoDepthAnything(encoder='vits',features=64,out_channels=[48,96,192,384],metric=False)
        else:
            from depth_anything_v2.dpt import DepthAnythingV2
            model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
        model.load_state_dict(torch.load(ROOT/'checkpoints'/weight,map_location='cpu',weights_only=True),strict=True)
        model=model.cuda().eval()
        print('INFER',label,'input_size',args.input_size,flush=True)
        shapes=[]
        # infer_* calls self.forward directly, bypassing hooks on the root model.
        # patch_embed is invoked normally and observes the actual BCHW tensor.
        handle=model.pretrained.patch_embed.register_forward_pre_hook(lambda m,a:shapes.append(list(a[0].shape)))
        with torch.inference_mode():
            if name=='vda':
                # Warm one official window; exclude it from reported inference time.
                model.infer_video_depth(frames[:32],fps,input_size=args.input_size,device='cuda',fp32=False)
            else:
                with torch.autocast('cuda',dtype=torch.float16):
                    model.infer_image(frames[0][:,:,::-1].copy(),args.input_size)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            if name=='vda':
                depths,_=model.infer_video_depth(frames,fps,input_size=args.input_size,device='cuda',fp32=False)
            else:
                depth_list=[]
                with torch.autocast('cuda',dtype=torch.float16):
                    for i,frame in enumerate(frames):
                        depth_list.append(model.infer_image(frame[:,:,::-1].copy(),args.input_size))
                        if i%20==0:print('FRAMEWISE',i+1,'/',count,flush=True)
                depths=np.stack(depth_list)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if depths.shape!=frames.shape[:3] or not np.isfinite(depths).all():raise ValueError('Invalid video output')
        if not shapes:raise RuntimeError('Actual network input dimensions were not recorded')
        record={'model':label,'seconds_total':elapsed,'seconds_per_source_frame':elapsed/count,
                'fps_effective':count/elapsed,'peak_allocated_MB':torch.cuda.max_memory_allocated()/1e6,
                'peak_reserved_MB':torch.cuda.max_memory_reserved()/1e6,
                'network_shapes':sorted({tuple(s) for s in shapes}),
                'normalization':fixed_range(depths),'checkpoint_bytes':(ROOT/'checkpoints'/weight).stat().st_size,
                'timing_scope':'official inference pipeline, excludes warmup, loading and visualization'}
        np.savez_compressed(saved,depths=depths.astype(np.float32))
        statfile.write_text(json.dumps(record,indent=2));arrays[name]=depths;stats[name]=record
        handle.remove();del model;gc.collect();torch.cuda.empty_cache()
        print(record,flush=True)
    h,w=frames.shape[1:3]
    comparison=cv2.VideoWriter(str(out/'rgb_depth_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*3,h+38))
    demo=cv2.VideoWriter(str(out/'depth_aware_blur_demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*3,h+38))
    if not comparison.isOpened() or not demo.isOpened():raise RuntimeError('Video codec failed')
    # One frozen relative near/far threshold over the whole clip, no per-frame fit.
    lo,hi=stats['vda']['normalization']
    focus=float(np.quantile(arrays['vda'],.85));transition=max((hi-lo)*.12,1e-6)
    pixel_changes=[]
    for i,frame in enumerate(frames):
        da=colorize(arrays['dav2'][i],stats['dav2']['normalization'])
        vd=colorize(arrays['vda'][i],stats['vda']['normalization'])
        row=np.concatenate([panel(frame,'RGB input'),panel(da,'Depth Anything V2-S | relative'),panel(vd,'Video Depth Anything-S | relative')],axis=1)
        comparison.write(cv2.cvtColor(row,cv2.COLOR_RGB2BGR))
        alpha=np.clip((focus-arrays['vda'][i])/transition,0,1)[...,None]
        alpha=alpha*alpha*(3-2*alpha)
        blur=cv2.GaussianBlur(frame,(0,0),sigmaX=9)
        application=np.clip(frame*(1-alpha)+blur*alpha,0,255).astype(np.uint8)
        pixel_changes.append(float(np.abs(application.astype(float)-frame).mean()))
        app=np.concatenate([panel(frame,'RGB input'),panel(vd,'Relative depth | fixed clip range'),panel(application,'Depth-aware background blur')],axis=1)
        demo.write(cv2.cvtColor(app,cv2.COLOR_RGB2BGR))
        if i in sorted(set([0,24,49,count-1])):
            cv2.imwrite(str(out/f'comparison_{i:03d}.jpg'),cv2.cvtColor(row,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out/f'application_{i:03d}.jpg'),cv2.cvtColor(app,cv2.COLOR_RGB2BGR))
    comparison.release();demo.release()
    verification={}
    for path in [out/'rgb_depth_comparison.mp4',out/'depth_aware_blur_demo.mp4']:
        cap=cv2.VideoCapture(str(path));decoded_count=0
        actual_fps=cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok,frame=cap.read()
            if not ok:break
            decoded_count+=1
        cap.release()
        if decoded_count!=count or abs(actual_fps-fps)>.01:raise ValueError('Video frames/FPS verification failed')
        verification[path.name]={'decoded_frames':decoded_count,'source_frames':count,'requested_frames':100,'fps':actual_fps,'width':w*3,'height':h+38}
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    (out/'application.json').write_text(json.dumps({'focus_threshold_relative':focus,'focus_quantile':.85,'transition':transition,'sigma_pixels':9,
        'mean_absolute_rgb_change_0_255':float(np.mean(pixel_changes))},indent=2))
    print('VIDEO COMPLETE',verification,flush=True)

if __name__=='__main__':main()
