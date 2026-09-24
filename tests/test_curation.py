import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romireason_bn.curation import (  # noqa: E402
    exact_deduplication,
    exact_deduplication_summary,
    exact_duplicate_clusters,
    normalize_answer_for_deduplication,
    normalize_for_deduplication,
    preprocess_dedup_rows,
    lexical_near_deduplication,
    mcq_order_invariant_deduplication,
    strict_near_deduplication,
    validate_source_answers,
)
from romireason_bn.semantic_dedup import (  # noqa: E402
    _complete_link_components,
    _jaccard_3gram,
    _semantic_fields,
)


class CurationTests(unittest.TestCase):
    def test_normalization_unifies_unicode_and_layout(self):
        self.assertEqual(normalize_for_deduplication("  বাংলা\nপ্রশ্ন  "), "বাংলা প্রশ্ন")

    def test_exact_clusters_keep_all_source_provenance(self):
        first = {"text": "প্রশ্ন", "answer": "A", "task_type": "basic_understanding", "source_dataset": "one"}
        second = {"text": " প্রশ্ন ", "answer": "a", "task_type": "basic_understanding", "source_dataset": "two"}
        clusters = exact_duplicate_clusters([first, second])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(next(iter(clusters.values()))), 2)

    def test_answer_normalization_unifies_bengali_digits(self):
        self.assertEqual(normalize_answer_for_deduplication(" ১২ "), "12")

    def test_preprocess_preserves_original_content(self):
        source = {
            "dedup_record_id": "one",
            "source_dataset": "source",
            "source_item_id": "1",
            "text": " প্রশ্ন ১২ ",
            "answer": " ৭ ",
        }
        accepted, rejected = preprocess_dedup_rows([source])
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["text"], " প্রশ্ন ১২ ")
        self.assertEqual(accepted[0]["text_normalized"], "প্রশ্ন ১২")
        self.assertEqual(accepted[0]["numeric_signature"], ["12"])

    def test_exact_dedup_merges_matching_answers_across_task_labels(self):
        rows, rejected = preprocess_dedup_rows(
            [
                {
                    "dedup_record_id": "a", "source_dataset": "one",
                    "source_item_id": "1", "task_type": "math_reasoning",
                    "text": "প্রশ্ন", "answer": "৭",
                },
                {
                    "dedup_record_id": "b", "source_dataset": "two",
                    "source_item_id": "2", "task_type": "basic_understanding",
                    "text": " প্রশ্ন ", "answer": "7",
                },
            ]
        )
        self.assertFalse(rejected)
        retained, clusters, quarantined = exact_deduplication(rows)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["dedup_action"], "exact_merged")
        self.assertEqual(clusters[0]["declared_task_types"], ["basic_understanding", "math_reasoning"])
        self.assertFalse(quarantined)

    def test_exact_answer_conflict_is_quarantined(self):
        rows, _ = preprocess_dedup_rows(
            [
                {"dedup_record_id": "a", "source_dataset": "one", "source_item_id": "1", "task_type": "math_reasoning", "text": "প্রশ্ন", "answer": "১"},
                {"dedup_record_id": "b", "source_dataset": "two", "source_item_id": "2", "task_type": "math_reasoning", "text": "প্রশ্ন", "answer": "২"},
            ]
        )
        retained, clusters, quarantined = exact_deduplication(rows)
        self.assertFalse(retained)
        self.assertEqual(clusters[0]["decision"], "quarantine_answer_conflict")
        self.assertEqual(len(quarantined), 2)

    def test_exact_summary_accounts_for_every_input(self):
        rows, _ = preprocess_dedup_rows(
            [
                {"dedup_record_id": "a", "source_dataset": "one", "source_item_id": "1", "task_type": "math_reasoning", "text": "প্রশ্ন", "answer": "১"},
                {"dedup_record_id": "b", "source_dataset": "two", "source_item_id": "2", "task_type": "math_reasoning", "text": "প্রশ্ন", "answer": "১"},
            ]
        )
        retained, clusters, quarantined = exact_deduplication(rows)
        summary = exact_deduplication_summary(rows, retained, clusters, quarantined)
        self.assertEqual(summary["collapsed_duplicate_records"], 1)

    def test_bmwp_source_answer_validation_quarantines_mismatch(self):
        accepted, quarantined = validate_source_answers(
            [
                {
                    "source_dataset": "BMWP", "answer": "৭",
                    "source_equation": "৩+৪",
                },
                {
                    "source_dataset": "BMWP", "answer": "৮",
                    "source_equation": "৩+৪",
                },
                {
                    "source_dataset": "BEnQA", "answer": "A",
                },
            ]
        )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0]["source_answer_validation"], "equation_mismatch")

    def test_strict_near_merges_punctuation_only_variation(self):
        rows, _ = preprocess_dedup_rows(
            [
                {"dedup_record_id": "a", "source_dataset": "one", "source_item_id": "1", "task_type": "math_reasoning", "text": "প্রশ্ন: ১২", "answer": "৭"},
                {"dedup_record_id": "b", "source_dataset": "two", "source_item_id": "2", "task_type": "math_reasoning", "text": "প্রশ্ন ১২।", "answer": "7"},
            ]
        )
        retained, clusters = strict_near_deduplication(rows)
        self.assertEqual(len(retained), 1)
        self.assertEqual(clusters[0]["decision"], "merge_punctuation_layout_equivalent")

    def test_lexical_near_merges_one_character_typo_with_same_numbers(self):
        rows, _ = preprocess_dedup_rows(
            [
                {"dedup_record_id": "a", "source_dataset": "one", "source_item_id": "1", "task_type": "math_reasoning", "text": "রহিমের কাছে ১২টি আম আছে", "answer": "৭"},
                {"dedup_record_id": "b", "source_dataset": "two", "source_item_id": "2", "task_type": "math_reasoning", "text": "রহিমের কাছে ১২টি আম আচে", "answer": "7"},
            ]
        )
        retained, clusters = lexical_near_deduplication(rows, threshold=0.90)
        self.assertEqual(len(retained), 1)
        self.assertEqual(clusters[0]["decision"], "merge_lexical_near_identical")

    def test_mcq_order_invariant_dedup_merges_reordered_options(self):
        rows, _ = preprocess_dedup_rows(
            [
                {
                    "dedup_record_id": "a", "source_dataset": "one",
                    "source_item_id": "1", "task_type": "factual_qa",
                    "text": "প্রশ্ন\nA) এক\nB) দুই\nC) তিন\nD) চার",
                    "answer": "B",
                },
                {
                    "dedup_record_id": "b", "source_dataset": "two",
                    "source_item_id": "2", "task_type": "factual_qa",
                    "text": "প্রশ্ন\nA) দুই\nB) এক\nC) তিন\nD) চার",
                    "answer": "A",
                },
            ]
        )
        retained, clusters, quarantined = mcq_order_invariant_deduplication(rows)
        self.assertEqual(len(retained), 1)
        self.assertFalse(quarantined)
        self.assertEqual(clusters[0]["decision"], "merge_mcq_order_invariant")

    def test_semantic_fields_compare_mcqs_by_stem_and_correct_option(self):
        text, answer, is_mcq, numbers = _semantic_fields(
            {
                "native_text": "২ + ৩ কত?\nA) ৪\nB) ৫\nC) ৬\nD) ৭",
                "answer": "B",
            }
        )
        self.assertEqual(text, "২ + ৩ কত?")
        self.assertEqual(answer, "৫")
        self.assertTrue(is_mcq)
        self.assertEqual(numbers, ("2", "3"))

    def test_character_jaccard_is_exact_not_minhash_estimate(self):
        self.assertEqual(_jaccard_3gram("abcde", "abcde"), 1.0)
        self.assertLess(_jaccard_3gram("abcde", "xyzab"), 1.0)

    def test_semantic_clusters_do_not_merge_through_a_bridge(self):
        edges = {0: {1}, 1: {0, 2}, 2: {1}}
        clusters = _complete_link_components(edges, [0, 1, 2])
        self.assertEqual(clusters, [[0, 1], [2]])


if __name__ == "__main__":
    unittest.main()
