import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romireason_bn.analysis import analysis_summary, item_level_outcomes  # noqa: E402
from romireason_bn.evaluation import build_evaluation_jobs, load_response_rows, parse_and_score, split_inference_and_scoring_jobs, stratified_smoke_rows  # noqa: E402
from romireason_bn.model_planning import build_model_run_plan  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_mcq_scoring_is_strict(self):
        self.assertEqual(parse_and_score("A", "A"), ("parsed", True))
        self.assertEqual(parse_and_score("Answer: b", "B"), ("parsed", True))
        self.assertEqual(parse_and_score("The answer is A", "A"), ("unscorable", False))

    def test_free_response_normalizes_digits_and_whitespace(self):
        self.assertEqual(parse_and_score("  42 ", "৪২"), ("parsed", True))

    def test_final_tag_schema_aware_scoring(self):
        parser = "final_tag_schema_aware_v1"
        self.assertEqual(parse_and_score("work\n<final>B</final>", "B", parser), ("parsed", True))
        self.assertEqual(parse_and_score("<final>বিকল্প 1</final>", "Option 1", parser), ("parsed", True))
        self.assertEqual(parse_and_score("<final>4/35</final>", "৪/৩৫", parser), ("parsed", True))
        self.assertEqual(parse_and_score("<final>11.2</final>", "১১.২০", parser), ("parsed", True))
        self.assertEqual(parse_and_score("<final>neutral</final>", "neutral", parser), ("parsed", True))
        self.assertEqual(parse_and_score("<final>A</final> extra", "A", parser), ("unscorable", False))

    def test_think_answer_schema_aware_scoring(self):
        parser = "think_answer_schema_aware_v1"
        self.assertEqual(parse_and_score("<think>work</think><answer>3</answer>", "৩", parser), ("parsed", True))
        self.assertEqual(parse_and_score("<think>work</think><answer>A</answer>", "A", parser), ("parsed", True))
        self.assertEqual(parse_and_score("<answer>A</answer>", "A", parser), ("unscorable", False))
        self.assertEqual(parse_and_score("<think>work</think><answer>A</answer> extra", "A", parser), ("unscorable", False))

    def test_answer_only_schema_aware_scoring(self):
        parser = "answer_only_schema_aware_v1"
        self.assertEqual(parse_and_score("<answer>3</answer>", "৩", parser), ("parsed", True))
        self.assertEqual(parse_and_score("<answer>Neutral</answer>", "neutral", parser), ("parsed", True))
        self.assertEqual(parse_and_score("reasoning <answer>A</answer>", "A", parser), ("unscorable", False))

    def test_inference_jobs_do_not_contain_answer_keys(self):
        jobs = [{"form_id": "x", "prompt": "solve", "expected_answer": "A", "task_type": "math_reasoning"}]
        inference, scoring = split_inference_and_scoring_jobs(jobs)
        self.assertNotIn("expected_answer", inference[0])
        self.assertNotIn("prompt", scoring[0])
        self.assertEqual(scoring[0]["expected_answer"], "A")

    def test_evaluation_jobs_preserve_model_specific_template_kwargs(self):
        rows = [{"item_id": "item-1", "review_status": "frozen", "task_type": "math_reasoning", "answer": "A",
                 "native_text": "native", "canonical_text": "canonical",
                 "llm_generated_variants": [{"variant_id": 1, "text": "one"}, {"variant_id": 2, "text": "two"}, {"variant_id": 3, "text": "three"}],
                 "rule_synthetic_text": "synthetic"}]
        protocol = {"status": "frozen", "prompt_template": "{item_text}", "decoding": {}, "scoring": {},
                    "chat_template_kwargs_by_repository": {"example/Qwen": {"enable_thinking": False}}}
        model = {"name": "example", "repository": "example/Qwen", "revision": "r", "tokenizer_revision": "r", "tokenizer_sha256": "h"}
        jobs = build_evaluation_jobs(rows, protocol, model)
        self.assertEqual(len(jobs), 6)
        self.assertEqual(jobs[0]["chat_template_kwargs"], {"enable_thinking": False})

    def test_response_shard_directory_loads_in_order_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "shard-00001.jsonl").write_text(json.dumps({"form_id": "b"}) + "\n", encoding="utf-8")
            (directory / "shard-00000.jsonl").write_text(json.dumps({"form_id": "a"}) + "\n", encoding="utf-8")
            self.assertEqual([row["form_id"] for row in load_response_rows(directory)], ["a", "b"])
            (directory / "shard-00002.jsonl").write_text(json.dumps({"form_id": "a"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate response"):
                load_response_rows(directory)

    def test_smoke_cohort_is_balanced_and_reproducible(self):
        rows = [
            {"item_id": f"{task}-{index}", "task_type": task, "review_status": "frozen"}
            for task in ("math_reasoning", "factual_qa", "basic_understanding")
            for index in range(4)
        ]
        first = stratified_smoke_rows(rows, per_task=2, seed=2027)
        second = stratified_smoke_rows(rows, per_task=2, seed=2027)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)

    def test_item_analysis_requires_complete_llm_triplet(self):
        rows = []
        for condition, variant_id, correct in (
            ("native", 1, True), ("canonical", 1, True),
            ("llm_generated_variant", 1, True), ("llm_generated_variant", 2, False),
            ("llm_generated_variant", 3, True), ("synthetic_variant", 1, True),
        ):
            rows.append({"item_id": "item-1", "task_type": "math_reasoning", "condition": condition,
                         "variant_id": variant_id, "correct": correct})
        outcomes = item_level_outcomes(rows)
        self.assertFalse(outcomes[0]["llm_all"])
        self.assertFalse(outcomes[0]["all_forms_same_and_correct"])
        summary = analysis_summary(rows, replicates=20, seed=2027)
        self.assertEqual(summary["items"], 1)

    def test_model_plan_contains_primary_large_models_only(self):
        plan = build_model_run_plan(ROOT / "configs/models.json", ROOT / "configs/evaluation_protocol_v1.json", 2)
        self.assertEqual(len(plan["models_ready_for_final_verification"]), 4)
        self.assertEqual(len(plan["models_pending_access_or_tokenizer_verification"]), 0)
        self.assertEqual(plan["models_ready_for_final_verification"][0]["forms"], 12)


if __name__ == "__main__":
    unittest.main()
