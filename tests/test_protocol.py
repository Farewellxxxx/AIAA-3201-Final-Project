import unittest
import numpy as np
from depth_project.protocol import (nyu_metrics, disparity_to_depth,
                                    fixed_range, load_splits, validate_clip_length)
from scipy.io import savemat
from tempfile import TemporaryDirectory
from pathlib import Path
from io import BytesIO


class ProtocolTests(unittest.TestCase):
    def test_short_clip_requires_explicit_disclosure(self):
        with self.assertRaises(ValueError): validate_clip_length(70)
        self.assertEqual(validate_clip_length(70,allow_short=True),70)
        self.assertEqual(validate_clip_length(100),100)
        with self.assertRaises(ValueError): validate_clip_length(0,allow_short=True)

    def test_crop_and_invalid_gt(self):
        gt = np.full((480,640), 2.0)
        pred = gt.copy()
        pred[:45] = 9
        pred[:, :41] = 9
        pred[471:] = 9
        pred[:,601:] = 9
        gt[50,50] = np.nan
        gt[51,50] = 0
        got = nyu_metrics(pred,gt)
        self.assertEqual(got['absrel'], 0)
        self.assertEqual(got['rmse'], 0)
        self.assertEqual(got['delta1'], 1)

    def test_raw_scale_and_clipping(self):
        gt = np.full((480,640),2.0)
        got = nyu_metrics(gt*2,gt)
        self.assertEqual(got['absrel'],1)
        self.assertEqual(got['rmse'],2)
        self.assertEqual(got['delta1'],0)
        self.assertEqual(nyu_metrics(gt*100,gt)['absrel'],4)

    def test_nonfinite_prediction_is_not_silently_excluded(self):
        gt = np.ones((480,640))
        pred = gt.copy(); pred[50,50] = np.nan
        with self.assertRaises(ValueError): nyu_metrics(pred,gt)

    def test_geometry_offset_and_units(self):
        d=np.array([10.0,0.0,-1.0,np.nan])
        got=disparity_to_depth(d,1000,100,10)
        self.assertEqual(got[0],5.0)
        self.assertTrue(np.isnan(got[1:]).all())

    def test_clip_range_is_shared(self):
        a=np.array([[[1.,2.]],[[3.,4.]]])
        self.assertEqual(fixed_range(a),(1.,4.))

    def test_matlab_split_index_conversion_and_overlap(self):
        p=BytesIO()
        savemat(p,{'trainNdxs':np.array([[1],[3]]),'testNdxs':np.array([[2]])})
        p.seek(0)
        train,test=load_splits(p,expected_counts=None)
        np.testing.assert_array_equal(train,[0,2])
        np.testing.assert_array_equal(test,[1])
        p=BytesIO()
        savemat(p,{'trainNdxs':np.array([[1]]),'testNdxs':np.array([[1]])})
        p.seek(0)
        with self.assertRaises(ValueError): load_splits(p,expected_counts=None)

if __name__=='__main__': unittest.main()
