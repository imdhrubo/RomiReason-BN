import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romireason_bn.acquisition import write_acquisition_record  # noqa: E402


class AcquisitionTests(unittest.TestCase):
    def test_record_captures_pin_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "sample.jsonl"
            artifact.write_text('{"id": 1}\n', encoding="utf-8")
            record = root / "manifest.json"
            write_acquisition_record(
                record,
                dataset_id="org/example",
                revision="a" * 40,
                split="train",
                config_name=None,
                row_count=1,
                artifact_paths=[artifact],
            )
            payload = json.loads(record.read_text(encoding="utf-8"))
        self.assertEqual(payload["revision"], "a" * 40)
        self.assertEqual(payload["artifacts"][0]["sha256"], hashlib.sha256(b'{"id": 1}\n').hexdigest())


if __name__ == "__main__":
    unittest.main()
