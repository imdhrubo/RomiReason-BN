import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romireason_bn.canonicalize import (  # noqa: E402
    CANONICAL_ENGINE,
    build_canonical_forms,
    build_canonical_records,
    canonicalize_bengali,
)
from romireason_bn.validation import contains_bengali  # noqa: E402
from romireason_bn.llm_generation import (  # noqa: E402
    build_batch_requests,
    dynamic_output_token_cap,
)
from romireason_bn.llm_validation import validate_batch_rows  # noqa: E402


class CanonicalizeTests(unittest.TestCase):
    def test_iso_canonicalization_is_latin_and_converts_bengali_digits(self):
        output = canonicalize_bengali("রাহিম ৮টায় যায়")
        self.assertFalse(contains_bengali(output))
        self.assertIn("8", output)

    def test_canonicalization_normalizes_assamese_ra_in_bengali_text(self):
        self.assertFalse(contains_bengali(canonicalize_bengali("নাইট্ৰোজেন")))

    def test_canonicalization_normalizes_bengali_full_stop(self):
        self.assertEqual(canonicalize_bengali("হ্যাঁ৷"), "hyām̐.")

    def test_canonicalization_normalizes_taka_sign(self):
        self.assertEqual(canonicalize_bengali("৳৯০"), "Tk.90")

    def test_canonicalization_normalizes_malformed_au_vowel_sequence(self):
        self.assertFalse(contains_bengali(canonicalize_bengali("যোৗগ")))

    def test_canonicalization_normalizes_isolated_au_length_mark(self):
        self.assertFalse(contains_bengali(canonicalize_bengali("মৗলিক")))

    def test_canonical_forms_keep_id_answer_and_task(self):
        forms = build_canonical_forms(
            [{"item_id": "rrbn-pilot-0001", "native_text": "হ্যাঁ", "answer": "A", "task_type": "basic_understanding"}]
        )
        self.assertEqual(forms[0]["item_id"], "rrbn-pilot-0001")
        self.assertEqual(forms[0]["answer"], "A")
        self.assertEqual(forms[0]["canonical_engine"], CANONICAL_ENGINE)

    def test_canonical_records_are_paired_schema_rows(self):
        records = build_canonical_records(
            [{
                "item_id": "rrbn-native-00001", "native_text": "হ্যাঁ",
                "answer": "A", "task_type": "basic_understanding",
                "source_provenance": {"source_dataset": "test", "source_item_id": "1"},
            }]
        )
        self.assertEqual(records[0]["condition"], "canonical")
        self.assertEqual(records[0]["source_dataset"], "test")
        self.assertFalse(contains_bengali(records[0]["text"]))

    def test_llm_batch_request_is_pinned_and_structured(self):
        requests = build_batch_requests(
            [{
                "item_id": "rrbn-native-00001", "task_type": "math_reasoning",
                "native_text": "২ + ২ কত?", "answer": "৪",
                "source_provenance": {"source_dataset": "test"},
            }],
            prompt="test prompt", model="gpt-5.4-2026-03-05", max_output_tokens=600,
        )
        self.assertEqual(requests[0]["url"], "/v1/responses")
        self.assertEqual(requests[0]["body"]["model"], "gpt-5.4-2026-03-05")
        self.assertEqual(requests[0]["body"]["text"]["format"]["schema"]["required"], ["variants"])
        self.assertNotIn("ANSWER_KEY", requests[0]["body"]["input"])

    def test_llm_batch_uses_dynamic_cap_when_not_overridden(self):
        item = {
            "item_id": "rrbn-native-00001", "task_type": "basic_understanding",
            "native_text": "ক" * 300, "answer": "A",
            "source_provenance": {"source_dataset": "test"},
        }
        request = build_batch_requests([item], prompt="prompt", model="model")[0]
        self.assertEqual(
            request["body"]["max_output_tokens"], dynamic_output_token_cap(item["native_text"])
        )

    def test_llm_validator_quarantines_changed_number(self):
        item = {
            "item_id": "rrbn-native-00001", "native_text": "২ + ২ কত?\nA) ৩\nB) ৪",
        }
        output = {
            "custom_id": "rrbn-native-00001:llm-variants-v1",
            "request_max_output_tokens": 400,
            "response": {"status_code": 200, "body": {"output_text": json.dumps({"variants": [
                {"variant_id": 1, "text": "2 + 2 koto?\nA) 3\nB) 4"},
                {"variant_id": 2, "text": "2 + 2 koto?\nA) 3\nB) 4"},
                {"variant_id": 3, "text": "2 + 3 koto?\nA) 3\nB) 4"},
            ]})}},
        }
        accepted, rejected = validate_batch_rows([item], [output])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected[0]["item_id"], item["item_id"])
        self.assertIn("numbers_changed", {reason for row in rejected for reason in row["reasons"]})


if __name__ == "__main__":
    unittest.main()
