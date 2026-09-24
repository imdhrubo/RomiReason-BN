import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romireason_bn.seed_freeze import (  # noqa: E402
    freeze_curated_native,
    freeze_native_seeds,
    native_dataset_summary,
)


class SeedFreezeTests(unittest.TestCase):
    def test_assigns_stable_ids_and_preserves_source_fields(self):
        seeds = freeze_native_seeds(
            [
                {
                    "source_dataset": "source",
                    "source_item_id": "1",
                    "task_type": "basic_understanding",
                    "text": "প্রশ্ন",
                    "answer": "A",
                    "source_subject": "language",
                }
            ]
        )
        self.assertEqual(seeds[0]["item_id"], "rrbn-pilot-0001")
        self.assertEqual(seeds[0]["native_text"], "প্রশ্ন")
        self.assertEqual(seeds[0]["source_subject"], "language")

    def test_freezes_curated_rows_with_nested_provenance(self):
        frozen = freeze_curated_native(
            [
                {
                    "dedup_record_id": "b",
                    "source_dataset": "source",
                    "source_item_id": "1",
                    "source_revision": "pin",
                    "task_type": "math_reasoning",
                    "text": "প্রশ্ন",
                    "answer": "৭",
                    "dedup_action": "unique",
                    "structural_quality_status": "pass",
                }
            ]
        )
        self.assertEqual(frozen[0]["item_id"], "rrbn-native-00001")
        self.assertEqual(frozen[0]["source_provenance"]["source_revision"], "pin")
        self.assertEqual(frozen[0]["curation_provenance"]["dedup_action"], "unique")
        self.assertEqual(native_dataset_summary(frozen)["task_counts"]["math_reasoning"], 1)


if __name__ == "__main__":
    unittest.main()
