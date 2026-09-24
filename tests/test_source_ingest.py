import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romireason_bn.source_ingest import (  # noqa: E402
    normalize_bennumeval_row,
    normalize_bennumeval_rows,
    normalize_benqa_row,
    normalize_benqa_rows,
    normalize_bnmmlu_row,
    normalize_bnmmlu_rows,
    normalize_ganit_rows,
    normalize_bluck_csv,
    normalize_banglamath_csv,
    normalize_bmwp_rows,
)


class SourceIngestTests(unittest.TestCase):
    def test_bennumeval_batch_retains_malformed_row_as_rejection(self):
        accepted, rejected = normalize_bennumeval_rows(
            "CA",
            [{"Q No.": 1, "Question": "প্রশ্ন", "Answer": "৭"}, {"Q No.": 2, "Answer": "৮"}],
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected[0]["source_row_index"], "2")

    def test_cq_keeps_options(self):
        row = {"Q No.": 1, "Question": "প্রশ্ন", "Answer": "Option 2", "option1": "এক", "option2": "দুই"}
        normalized = normalize_bennumeval_row("CQ", row)
        self.assertIn("বিকল্প 1: এক", normalized["text"])
        self.assertIn("বিকল্প 2: দুই", normalized["text"])

    def test_qnli_keeps_premise_and_hypothesis(self):
        row = {"Q No.": 1, "Premise": "ভিত্তি", "Hypothesis": "অনুমান", "Answer": "entailment"}
        normalized = normalize_bennumeval_row("QNLI", row)
        self.assertIn("প্রতিজ্ঞা: ভিত্তি", normalized["text"])
        self.assertIn("অনুমান: অনুমান", normalized["text"])

    def test_bnmmlu_keeps_all_options_and_answer_label(self):
        row = {
            "Unique_Serial": 7,
            "subject_name": "Bengali Language & Syntax",
            "question": "প্রশ্ন",
            "correct_answer": "b",
            "options": ["এক", "দুই", "তিন", "চার"],
        }
        normalized = normalize_bnmmlu_row(row)
        self.assertEqual(normalized["answer"], "B")
        self.assertIn("D) চার", normalized["text"])

    def test_bnmmlu_batch_retains_bad_options_as_rejection(self):
        valid = {
            "Unique_Serial": 7, "subject_name": "Bengali Language & Syntax",
            "question": "প্রশ্ন", "correct_answer": "B",
            "options": ["এক", "দুই", "তিন", "চার"],
        }
        invalid = dict(valid, Unique_Serial=8, options=["এক", "দুই"])
        accepted, rejected = normalize_bnmmlu_rows([valid, invalid])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected[0]["source_item_id"], "8")

    def test_ganit_batch_requires_upstream_valid_and_answer(self):
        valid = {
            "id": 1, "valid": 1, "problem": "প্রশ্ন", "bengali_solution": "৪",
            "source_name": "source", "difficulty": "easy",
        }
        invalid = dict(valid, id=2, valid=0)
        accepted, rejected = normalize_ganit_rows([valid, invalid])
        self.assertEqual(accepted[0]["answer"], "৪")
        self.assertEqual(rejected[0]["source_item_id"], "2")

    def test_bluck_normalizer_uppercases_answer_and_keeps_options(self):
        row = {
            "question": "প্রশ্ন", "a": "এক", "b": "দুই", "c": "তিন", "d": "চার",
            "answer": "c",
        }
        accepted, rejected = normalize_bluck_csv(Path("Semantics/test.csv"), [row])
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["answer"], "C")
        self.assertIn("D) চার", accepted[0]["text"])

    def test_banglamath_normalizer_keeps_answer_and_grade(self):
        accepted, rejected = normalize_banglamath_csv(
            [{"Question": "প্রশ্ন", "Answer": "উত্তর", "Grade": "six", "Steps": "2"}]
        )
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["source_grade"], "six")

    def test_bmwp_normalizer_keeps_equation(self):
        accepted, rejected = normalize_bmwp_rows(
            [{"Problem": "প্রশ্ন", "Sollution": 42, "Equation": "৪০+২", "Class": "six"}]
        )
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["answer"], "42")
        self.assertEqual(accepted[0]["source_equation"], "৪০+২")

    def test_benqa_uses_bengali_fields(self):
        row = {
            "English Question": "Ignored",
            "Bengali Question": "বাংলা প্রশ্ন",
            "A Bn": "এক",
            "B Bn": "দুই",
            "C Bn": "তিন",
            "D Bn": "চার",
            "Correct Answer": "d",
        }
        normalized = normalize_benqa_row("8th-Science.csv", 3, row)
        self.assertEqual(normalized["answer"], "D")
        self.assertIn("A) এক", normalized["text"])
        self.assertNotIn("Ignored", normalized["text"])

    def test_benqa_batch_retains_malformed_answer_as_rejection(self):
        valid = {
            "Bengali Question": "প্রশ্ন",
            "A Bn": "এক", "B Bn": "দুই", "C Bn": "তিন", "D Bn": "চার",
            "Correct Answer": "a",
        }
        invalid = dict(valid, **{"Correct Answer": "*"})
        accepted, rejected = normalize_benqa_rows("source.csv", [valid, invalid])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected[0]["source_item_id"], "source.csv:2")


if __name__ == "__main__":
    unittest.main()
