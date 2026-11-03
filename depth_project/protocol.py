"""Course protocol, with no model or test-set scale fitting hidden in metrics."""
import numpy as np
from scipy.io import loadmat

CROP = (slice(45,471), slice(41,601))
MIN_DEPTH, MAX_DEPTH = 0.001, 10.0

def validate_clip_length(frame_count,allow_short=False):
    if frame_count==100 or (allow_short and 0<frame_count<100): return frame_count
    raise ValueError(f'Expected 100 frames; source yielded {frame_count}. A short source requires explicit disclosure.')

def evaluation_mask(gt):
    gt=np.asarray(gt)
    if gt.shape != (480,640):
        raise ValueError(f'Expected NYU depth shape (480,640), got {gt.shape}')
    mask=np.zeros(gt.shape,dtype=bool)
    mask[CROP]=True
    return mask & np.isfinite(gt) & (gt>MIN_DEPTH) & (gt<MAX_DEPTH)

def nyu_metrics(pred,gt):
    pred,gt=np.asarray(pred,dtype=np.float64),np.asarray(gt,dtype=np.float64)
    if pred.shape != gt.shape: raise ValueError('Resize predictions to GT shape first')
    mask=evaluation_mask(gt)
    if not mask.any(): raise ValueError('No valid ground-truth pixels')
    p,g=pred[mask],gt[mask]
    if not np.isfinite(p).all(): raise ValueError('Nonfinite model predictions')
    p=np.clip(p,MIN_DEPTH,MAX_DEPTH)
    return {'absrel':float(np.mean(np.abs(p-g)/g)),
            'rmse':float(np.sqrt(np.mean((p-g)**2))),
            'delta1':float(np.mean(np.maximum(p/g,g/p)<1.25))}

def disparity_to_depth(disparity,focal_px,baseline_mm,doffs):
    d=np.asarray(disparity,dtype=np.float64)
    valid=np.isfinite(d)&(d>0)&((d+doffs)>0)
    out=np.full(d.shape,np.nan,dtype=np.float32)
    out[valid]=(focal_px*baseline_mm/(d[valid]+doffs)/1000).astype(np.float32)
    return out

def load_splits(path,expected_counts=(795,654)):
    mat=loadmat(path)
    train=np.asarray(mat['trainNdxs']).reshape(-1).astype(np.int64)-1
    test=np.asarray(mat['testNdxs']).reshape(-1).astype(np.int64)-1
    if min(train.min(),test.min())<0: raise ValueError('Invalid MATLAB indices')
    if len(set(train))!=len(train) or len(set(test))!=len(test):
        raise ValueError('Duplicate split IDs')
    if set(train)&set(test): raise ValueError('Train/test split overlap')
    if expected_counts and (len(train),len(test))!=expected_counts:
        raise ValueError('Unexpected split sizes')
    return train,test

def fixed_range(depths):
    d=np.asarray(depths)
    if not np.isfinite(d).all() or d.size==0: raise ValueError('Invalid depth clip')
    return float(d.min()),float(d.max())
