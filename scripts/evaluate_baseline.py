"""Part 2: evaluate raw UniDepthV2 predictions on the official NYU test split."""
from _bootstrap import ROOT
import csv
import json

import h5py
import numpy as np

from depth_project.protocol import load_splits, nyu_metrics


def evaluate_test_set(cache, depths, test_ids, expected_count=654):
    ids = [int(i) for i in test_ids]
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise ValueError('Test evaluation requires every unique official test ID')
    missing = [i for i in ids if not (cache / f'{i:04d}.npy').exists()]
    if missing:
        raise ValueError(f'Missing {len(missing)} test predictions; first IDs {missing[:5]}')
    rows = []
    for i in ids:
        pred = np.load(cache / f'{i:04d}.npy')
        if pred.shape != (480,640) or pred.dtype != np.float32 or not np.isfinite(pred).all():
            raise ValueError(f'Invalid raw float32 prediction for image {i}')
        rows.append({'id': i, **nyu_metrics(pred, depths[i].T)})
    metrics = {'count': len(rows), 'scale': 1.,
               **{key: float(np.mean([r[key] for r in rows]))
                  for key in ['absrel','rmse','delta1']}}
    return rows, metrics


def main():
    _, test = load_splits(ROOT / 'data/nyu/splits.mat')
    source = ROOT / 'results/nyu'
    out = source / 'baseline'
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(ROOT / 'data/nyu/nyu_depth_v2_labeled.mat', 'r') as data:
        rows, metrics = evaluate_test_set(source / 'predictions', data['depths'], test)
    result = {'model': 'UniDepthV2 ViT-S', 'stage': 'Part 2 reproduction',
              'protocol': {'crop': [45,471,41,601], 'depth_range_m': [.001,10],
                           'aggregation': 'equal per-image mean',
                           'test_scale_alignment': False, 'calibration': False},
              'metrics': metrics}
    timing_path = source / 'timing.jsonl'
    if timing_path.exists():
        records = [json.loads(line) for line in timing_path.read_text().splitlines()]
        test_ids = set(map(int, test))
        timing = {r['id']: r for r in records if r['id'] in test_ids}
        if set(timing) != test_ids:
            raise ValueError('Timing log does not cover every test image')
        result['timing'] = {
            'scope': 'test split only; official preprocessing and synchronized inference',
            'record_count': len(timing),
            'mean_seconds': float(np.mean([r['seconds'] for r in timing.values()])),
            'median_seconds': float(np.median([r['seconds'] for r in timing.values()])),
            'peak_allocated_MB': max(r['peak_allocated_MB'] for r in timing.values()),
            'input_shapes': sorted({tuple(r['network_input_shape']) for r in timing.values()})}
    with (out / 'per_image.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['id','absrel','rmse','delta1'])
        writer.writeheader()
        writer.writerows(rows)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
