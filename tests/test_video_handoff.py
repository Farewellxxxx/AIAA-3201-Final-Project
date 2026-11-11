import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


class VideoHandoffTests(unittest.TestCase):
    def setUp(self):
        from depth_project.video_render import load_depths, write_depth_metadata

        self.load_depths = load_depths
        self.write_depth_metadata = write_depth_metadata
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name) / "dav2_depths.npz"
        self.depths = np.ones((3, 6, 8), dtype=np.float32)
        np.savez(self.cache, depths=self.depths)
        self.digest = "a" * 64

    def test_rejects_different_source_fps_and_frame_order(self):
        self.write_depth_metadata(self.cache, self.digest, 24.0, 3)
        np.testing.assert_array_equal(
            self.load_depths(self.cache, self.digest, 24.0)[0], self.depths
        )
        for digest, fps in [("b" * 64, 24.0), (self.digest, 30.0)]:
            with self.assertRaises(ValueError):
                self.load_depths(self.cache, digest, fps)
        metadata_path = self.cache.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        metadata["frame_indices"] = [0, 2, 1]
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "frame order"):
            self.load_depths(self.cache, self.digest, 24.0)

    def test_legacy_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "Missing metadata"):
            self.load_depths(self.cache, self.digest, 24.0)
        depths, verified = self.load_depths(
            self.cache, self.digest, 24.0, allow_legacy=True
        )
        np.testing.assert_array_equal(depths, self.depths)
        self.assertFalse(verified)


if __name__ == "__main__":
    unittest.main()
