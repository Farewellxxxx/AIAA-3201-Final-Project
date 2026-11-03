"""Run the required 15-scene trainQ stereo exercise."""
from _bootstrap import ROOT
import json,time,csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from depth_project.data import read_pfm,parse_calibration,stereo_metrics
from depth_project.protocol import disparity_to_depth

def main():
    scenes=sorted((ROOT/'data/middlebury/MiddEval3/trainingQ').glob('*'))
    if not scenes:
        roots=list((ROOT/'data/middlebury').rglob('trainingQ'))+list((ROOT/'data/middlebury').rglob('trainQ'))
        if roots: scenes=sorted(p for p in roots[0].iterdir() if p.is_dir())
    scenes=[p for p in scenes if (p/'calib.txt').exists()]
    if len(scenes)!=15: raise ValueError(f'Expected 15 training scenes; got {len(scenes)}')
    out=ROOT/'results/stereo';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for scene in scenes:
        c=parse_calibration((scene/'calib.txt').read_text())
        left=cv2.imread(str(scene/'im0.png'));right=cv2.imread(str(scene/'im1.png'))
        gt=read_pfm(scene/'disp0GT.pfm')
        ndisp=int(np.ceil(c['ndisp']/16)*16)
        model=cv2.StereoSGBM_create(minDisparity=0,numDisparities=ndisp,blockSize=5,
            P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=10,
            speckleWindowSize=100,speckleRange=2,preFilterCap=31,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        l=cv2.cvtColor(left,cv2.COLOR_BGR2GRAY);r=cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
        start=time.perf_counter();disparity=model.compute(l,r).astype(np.float32)/16
        elapsed=time.perf_counter()-start
        depth=disparity_to_depth(disparity,c['focal_px'],c['baseline_mm'],c['doffs'])
        np.savez_compressed(out/f'{scene.name}.npz',disparity=disparity,depth_m=depth,valid=np.isfinite(depth))
        record={'scene':scene.name,**stereo_metrics(disparity,gt),'seconds':elapsed,'num_disparities':ndisp}
        records.append(record);print(record,flush=True)
        fig,axs=plt.subplots(1,4,figsize=(12,3),layout='constrained')
        panels=[(cv2.cvtColor(left,cv2.COLOR_BGR2RGB),'RGB',None),
                (np.ma.masked_where(disparity<=0,disparity),'SGBM disparity (px)','inferno'),
                (disparity<=0,'Invalid predictions','gray'),
                (depth,'Depth (m)','viridis_r')]
        for ax,(im,title,cmap) in zip(axs,panels):
            artist=ax.imshow(im,cmap=cmap);ax.set_title(title,fontsize=10);ax.axis('off')
            if title in ('SGBM disparity (px)','Depth (m)'): fig.colorbar(artist,ax=ax,shrink=.65)
        fig.savefig(out/f'{scene.name}.png',dpi=150);plt.close(fig)
    summary={'scene_count':len(records),'mean_scene_mae_px':float(np.mean([r['mae_px'] for r in records])),
             'mean_scene_valid_fraction':float(np.mean([r['valid_fraction'] for r in records])),
             'mean_seconds':float(np.mean([r['seconds'] for r in records])),
             'invalid_rule':'nonfinite disparity or disparity <= 0; OpenCV raw disparity divided by 16',
             'mae_domain':'finite positive GT intersected with valid predictions',
             'coverage_domain':'all finite positive GT pixels','records':records,'opencv':cv2.__version__}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    with (out/'per_scene.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    c=parse_calibration((scenes[0]/'calib.txt').read_text());d=np.linspace(1,c['ndisp'],300)
    z=c['focal_px']*c['baseline_mm']/(d+c['doffs'])/1000
    sensitivity=c['focal_px']*c['baseline_mm']/(d+c['doffs'])**2/1000
    fig,ax=plt.subplots(figsize=(5,3),layout='constrained');ax.plot(z,sensitivity)
    ax.set(xlabel='Depth (m)',ylabel='Depth sensitivity (m / disparity pixel)',title=f'{scenes[0].name}: sensitivity to disparity')
    ax.grid(alpha=.2);fig.savefig(out/'sensitivity.png',dpi=200);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2),flush=True)

if __name__=='__main__': main()
