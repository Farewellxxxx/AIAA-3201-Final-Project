"""Draw actual difficult NYU cases using raw, uncalibrated Part 2 predictions."""
from _bootstrap import ROOT
import csv
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from depth_project.protocol import evaluation_mask, load_splits


def main():
    ids = [468,87,516]
    _, test = load_splits(ROOT / 'data/nyu/splits.mat')
    if not set(ids) <= set(test):
        raise ValueError('Figure IDs must belong to the test split')
    scores = {int(r['id']): float(r['absrel']) for r in
              csv.DictReader((ROOT / 'results/nyu/baseline/per_image.csv').open())}
    fig, axes = plt.subplots(3,4,figsize=(11,6.5),layout='constrained')
    with h5py.File(ROOT / 'data/nyu/nyu_depth_v2_labeled.mat','r') as data:
        for row, i in enumerate(ids):
            rgb = data['images'][i].transpose(2,1,0)
            gt = data['depths'][i].T
            pred = np.load(ROOT / 'results/nyu/predictions' / f'{i:04d}.npy')
            error = np.ma.masked_where(~evaluation_mask(gt),
                np.abs(np.clip(pred,.001,10)-gt)/np.maximum(gt,.001))
            panels = [(rgb,f'RGB | ID {i}',None,None),
                      (gt,'Filled GT (m)','viridis_r',10),
                      (pred,'Raw UniDepthV2 (m)','viridis_r',10),
                      (error,f'Relative error | AbsRel {scores[i]:.3f}','magma',.5)]
            for ax,(array,title,cmap,maximum) in zip(axes[row],panels):
                image = ax.imshow(array,cmap=cmap,vmin=0 if cmap else None,vmax=maximum)
                ax.set_title(title,fontsize=9);ax.axis('off')
                if cmap:fig.colorbar(image,ax=ax,fraction=.025,pad=.01)
    destination = ROOT / 'results/nyu/baseline/difficult_cases.jpg'
    fig.savefig(destination,dpi=150)
    plt.close(fig)
    print(destination,flush=True)


if __name__ == '__main__':
    main()
