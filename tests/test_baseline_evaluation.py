"""Ensure Part 2 can be evaluated without training predictions or calibration."""
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


class BaselineEvaluationTests(unittest.TestCase):
    def evaluator(self):
        path = SCRIPTS / 'evaluate_baseline.py'
        self.assertTrue(path.exists(), 'Part 2 needs a standalone raw-scale evaluator')
        spec = importlib.util.spec_from_file_location('evaluate_baseline', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.evaluate_test_set

    def test_raw_scale_requires_only_test_predictions(self):
        evaluate = self.evaluator()
        depths = np.stack([np.ones((640,480)), np.full((640,480), np.nan),
                           np.full((640,480), 2.)])
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            for i in [0,2]:
                np.save(cache / f'{i:04d}.npy', np.full((480,640), 2, dtype=np.float32))
            rows, metrics = evaluate(cache, depths, [0,2], expected_count=2)
        self.assertEqual([r['id'] for r in rows], [0,2])
        self.assertEqual(metrics['scale'], 1.)
        self.assertEqual(metrics['absrel'], .5)
        self.assertEqual(metrics['rmse'], .5)
        self.assertEqual(metrics['delta1'], .5)

    def test_incomplete_or_duplicate_test_ids_are_rejected(self):
        evaluate = self.evaluator()
        with TemporaryDirectory() as tmp:
            for ids in [[0], [0,0]]:
                with self.assertRaises(ValueError):
                    evaluate(Path(tmp), None, ids, expected_count=2)

    def test_missing_test_prediction_is_rejected(self):
        evaluate = self.evaluator()
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Missing'):
                evaluate(Path(tmp), None, [0], expected_count=1)


if __name__ == '__main__':
    unittest.main()
