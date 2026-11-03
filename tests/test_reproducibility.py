import unittest,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from download import verify_digest, pinned_source

class ReproTests(unittest.TestCase):
    def test_checksum_mismatch_is_rejected(self):
        expected=hashlib.sha256(b'original weights').hexdigest()
        self.assertEqual(verify_digest(b'original weights',expected),expected)
        with self.assertRaises(ValueError): verify_digest(b'changed weights',expected)

    def test_recorded_source_wins_over_latest(self):
        saved={'third_party/Model':{'url':'https://example.com/zip/abc123','commit':'abc123','archive_sha256':'known'}}
        self.assertEqual(pinned_source(saved,'third_party/Model','https://example.com/zip/main')['commit'],'abc123')

if __name__=='__main__':unittest.main()
