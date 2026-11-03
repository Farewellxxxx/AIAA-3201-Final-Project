"""Data formats and stereo bookkeeping."""
from pathlib import Path
import re
import numpy as np

def read_pfm(source):
    if not hasattr(source,'read'):
        with Path(source).open('rb') as stream: return read_pfm(stream)
    magic=source.readline().strip()
    if magic not in (b'Pf',b'PF'): raise ValueError('Not a PFM file')
    line=source.readline()
    while line.startswith(b'#'): line=source.readline()
    width,height=map(int,line.split());scale=float(source.readline())
    data=np.frombuffer(source.read(),dtype='<f4' if scale<0 else '>f4')
    shape=(height,width,3) if magic==b'PF' else (height,width)
    # Middlebury stores disparities directly in pixels. The header sign specifies
    # byte order; its magnitude is not a conversion factor for benchmark depths.
    return np.flipud(data.reshape(shape)).astype(np.float32)

def parse_calibration(text):
    fields=dict(line.split('=',1) for line in text.splitlines() if '=' in line)
    cam=np.array([float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',fields['cam0'])]).reshape(3,3)
    return {'focal_px':float(cam[0,0]),'baseline_mm':float(fields['baseline']),
            'doffs':float(fields['doffs']),'ndisp':int(fields['ndisp']),
            'intrinsics':cam.tolist()}

def stereo_metrics(pred,gt):
    eligible=np.isfinite(gt)&(gt>0)
    valid=eligible&np.isfinite(pred)&(pred>0)
    if not eligible.any() or not valid.any(): raise ValueError('Empty stereo evaluation')
    return {'mae_px':float(np.abs(pred[valid]-gt[valid]).mean()),
            'valid_fraction':float(valid.sum()/eligible.sum()),
            'gt_pixels':int(eligible.sum()),'valid_pixels':int(valid.sum())}
