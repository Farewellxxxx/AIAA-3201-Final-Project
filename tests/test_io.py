import unittest
from io import BytesIO
import numpy as np
from depth_project.data import read_pfm, parse_calibration, stereo_metrics

class DataTests(unittest.TestCase):
    def test_pfm_bottom_up_and_endian(self):
        source=BytesIO(b'Pf\n2 2\n-1.0\n'+np.array([3,4,1,2],dtype='<f4').tobytes())
        np.testing.assert_array_equal(read_pfm(source),[[1,2],[3,4]])

    def test_middlebury_header_does_not_rescale_pixel_disparity(self):
        source=BytesIO(b'Pf\n2 1\n-0.003922\n'+np.array([20,40],dtype='<f4').tobytes())
        np.testing.assert_array_equal(read_pfm(source),[[20,40]])

    def test_calibration_and_coverage(self):
        c=parse_calibration('cam0=[1000 0 300; 0 1000 200; 0 0 1]\nbaseline=100\ndoffs=10\nndisp=128')
        self.assertEqual(c['focal_px'],1000)
        self.assertEqual(c['baseline_mm'],100)
        got=stereo_metrics(np.array([2.,-1.,np.nan,5.]),np.array([1.,2.,3.,np.inf]))
        self.assertEqual(got['mae_px'],1)
        self.assertAlmostEqual(got['valid_fraction'],1/3)

if __name__=='__main__': unittest.main()
