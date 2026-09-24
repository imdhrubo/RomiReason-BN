import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romireason_bn.quality import structural_quality_filter  # noqa: E402
from romireason_bn.task_mapping import map_candidate_tasks  # noqa: E402


class TaskMappingTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {
            "default_task_by_source": {"BMWP": "math_reasoning"},
            "benqa_file_patterns": [
                {"pattern": "Math", "task_type": "math_reasoning"},
                {"pattern": ".*", "task_type": "factual_qa"},
            ],
            "bnmmlu_subject_patterns": [
                {"pattern": "Algebra", "task_type": "math_reasoning"},
                {"pattern": ".*", "task_type": "factual_qa"},
            ],
        }

    def test_maps_benqa_by_source_file(self):
        mapped, quarantined = map_candidate_tasks(
            [{"source_dataset": "BEnQA", "source_file": "10th-Math.csv", "task_type": "unassigned", "text": "প্রশ্ন", "answer": "A"}],
            self.mapping,
        )
        self.assertFalse(quarantined)
        self.assertEqual(mapped[0]["task_type"], "math_reasoning")
        self.assertEqual(mapped[0]["source_task_type"], "unassigned")

    def test_quarantines_unmapped_source(self):
        mapped, quarantined = map_candidate_tasks(
            [{"source_dataset": "unknown", "task_type": "x"}], self.mapping
        )
        self.assertFalse(mapped)
        self.assertEqual(quarantined[0]["dedup_action"], "quarantine_unmapped_task")

    def test_structural_quality_requires_mcq_options_for_mcq_answer(self):
        accepted, quarantined = structural_quality_filter(
            [{"text": "বাংলা প্রশ্ন", "answer": "A"}]
        )
        self.assertFalse(accepted)
        self.assertIn(
            "mcq_answer_without_all_rendered_options",
            quarantined[0]["structural_quality_reasons"],
        )

    def test_structural_quality_keeps_lowercase_algebra_answer(self):
        accepted, quarantined = structural_quality_filter(
            [{"text": "a/b এর লব কী", "answer": "a"}]
        )
        self.assertFalse(quarantined)
        self.assertEqual(accepted[0]["structural_quality_status"], "pass")

    def test_structural_quality_accepts_native_free_response(self):
        accepted, quarantined = structural_quality_filter(
            [{"text": "বাংলা প্রশ্ন", "answer": "৭"}]
        )
        self.assertFalse(quarantined)
        self.assertEqual(accepted[0]["structural_quality_status"], "pass")


if __name__ == "__main__":
    unittest.main()
