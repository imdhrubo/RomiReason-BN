import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romireason_bn.sampling import balanced_allocation, stratified_sample  # noqa: E402


class SamplingTests(unittest.TestCase):
    def test_balanced_allocation_preserves_total(self):
        self.assertEqual(
            balanced_allocation(100, ["CA", "DS", "CQ", "FiB", "QNLI", "AWP"]),
            {"CA": 17, "DS": 17, "CQ": 17, "FiB": 17, "QNLI": 16, "AWP": 16},
        )

    def test_stratified_sample_is_reproducible(self):
        pools = {"a": list(range(10)), "b": list(range(10, 20))}
        allocation = {"a": 3, "b": 2}
        self.assertEqual(
            stratified_sample(pools, allocation, 2027),
            stratified_sample(pools, allocation, 2027),
        )

    def test_stratified_sample_rejects_oversampling(self):
        with self.assertRaisesRegex(ValueError, "invalid sample count"):
            stratified_sample({"a": [1]}, {"a": 2}, 2027)


if __name__ == "__main__":
    unittest.main()
