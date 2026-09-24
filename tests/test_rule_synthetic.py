import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romireason_bn.rule_synthetic import (  # noqa: E402
    build_rule_synthetic_records,
    load_rule_catalog,
    render_rule_synthetic,
)


class RuleSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_rule_catalog(ROOT / "configs/generation/rule_synthetic_v1.json")

    def test_renderer_removes_romanization_diacritics_and_preserves_formula(self):
        text, rules = render_rule_synthetic("śāōna r̥tu: \\sqrt{π}", self.catalog)
        self.assertEqual(text, "shaona ritu: \\sqrt{π}")
        self.assertIn("RBV1_VOCALIC_R", rules)
        self.assertIn("RBV1_SIBILANT_SHA", rules)

    def test_renderer_records_identity_when_no_rule_is_eligible(self):
        text, rules = render_rule_synthetic("x = kata?", self.catalog)
        self.assertEqual(text, "x = kata?")
        self.assertEqual(rules, ("RBV1_IDENTITY_NO_ELIGIBLE_MARK",))

    def test_builder_keeps_only_complete_cohort_and_preserves_lineage(self):
        rows = [{
            "item_id": "rrbn-native-00001", "task_type": "math_reasoning", "text": "śāōna",
            "answer": "A", "source_dataset": "test", "source_item_id": "1", "code_mixed": False,
            "canonical_engine": "engine",
        }, {
            "item_id": "rrbn-native-00002", "task_type": "math_reasoning", "text": "kata",
            "answer": "B", "source_dataset": "test", "source_item_id": "2", "code_mixed": False,
            "canonical_engine": "engine",
        }]
        records, rejected = build_rule_synthetic_records(rows, {"rrbn-native-00001"}, self.catalog)
        self.assertEqual(rejected, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], "shaona")
        self.assertEqual(records[0]["condition"], "synthetic_variant")
        self.assertEqual(records[0]["provenance"], "synthetic")


if __name__ == "__main__":
    unittest.main()
