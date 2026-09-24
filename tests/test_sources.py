import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romireason_bn.validation import audit_sources, check_layout  # noqa: E402


class SourceTests(unittest.TestCase):
    def test_project_layout(self):
        self.assertEqual(check_layout(ROOT), [])

    def test_disabled_pending_sources_are_allowed(self):
        self.assertEqual(audit_sources(ROOT / "configs/data_sources.json"), [])

    def test_enabled_source_requires_approved_metadata(self):
        payload = {
            "sources": [
                {
                    "name": "blocked",
                    "url": "https://example.invalid",
                    "enabled": True,
                    "license_status": "pending",
                    "license": None,
                    "revision": None,
                    "sha256": None,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            errors = audit_sources(path)
        self.assertTrue(any("license is not approved or author-authorized" in error for error in errors))
        self.assertTrue(any("missing revision" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
