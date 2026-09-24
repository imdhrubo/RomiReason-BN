import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romireason_bn.schema import Record  # noqa: E402
from romireason_bn.validation import load_jsonl, validate_records  # noqa: E402


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example_path = ROOT / "data/examples/pilot_example.jsonl"
        cls.raw_rows = [
            json.loads(line)
            for line in cls.example_path.read_text(encoding="utf-8").splitlines()
        ]

    def test_example_is_valid(self):
        records = load_jsonl(self.example_path)
        self.assertEqual(validate_records(records), [])

    def test_provenance_must_match_condition(self):
        row = dict(self.raw_rows[1])
        row["provenance"] = "llm_generated"
        with self.assertRaisesRegex(ValueError, "requires provenance deterministic"):
            Record.from_dict(row)

    def test_llm_generated_requires_generation_metadata(self):
        row = dict(self.raw_rows[2])
        row["generator_model"] = None
        with self.assertRaisesRegex(ValueError, "require generator_model"):
            Record.from_dict(row)

    def test_synthetic_requires_rules(self):
        row = dict(self.raw_rows[3])
        row["variant_rule_ids"] = []
        with self.assertRaisesRegex(ValueError, "require variant_rule_ids"):
            Record.from_dict(row)

    def test_group_requires_all_conditions(self):
        records = [Record.from_dict(row) for row in self.raw_rows[:-1]]
        self.assertIn("rrbn-0001: missing synthetic_variant", validate_records(records))

    def test_group_answers_must_match(self):
        rows = [dict(row) for row in self.raw_rows]
        rows[-1]["answer"] = "6"
        records = [Record.from_dict(row) for row in rows]
        self.assertIn("rrbn-0001: answers differ across paired forms", validate_records(records))

    def test_romanized_row_cannot_contain_bengali_script(self):
        rows = [dict(row) for row in self.raw_rows]
        rows[1]["text"] += " বাংলা"
        records = [Record.from_dict(row) for row in rows]
        self.assertTrue(
            any("Romanized condition contains Bengali script" in error for error in validate_records(records))
        )


if __name__ == "__main__":
    unittest.main()
