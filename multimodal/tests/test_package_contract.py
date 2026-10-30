"""Static checks for pinned model identities and shipped interface files."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


class PackageContractTests(unittest.TestCase):
    def test_every_model_has_immutable_base_and_adapter_identity(self):
        registry = json.loads((PACKAGE / "configs/model_registry.json").read_text())
        self.assertEqual(set(registry["models"]), {"qwen3vl8b", "llavanext8b", "internvl3_8b"})
        for model in registry["models"].values():
            self.assertRegex(model["base_revision"], re.compile(r"^[0-9a-f]{40}$"))
            self.assertRegex(model["adapter_revision"], re.compile(r"^[0-9a-f]{40}$"))
            self.assertRegex(model["base_tree_sha256"], re.compile(r"^[0-9a-f]{64}$"))
            self.assertRegex(model["adapter_weight_sha256"], re.compile(r"^[0-9a-f]{64}$"))
            self.assertRegex(model["adapter_config_sha256"], re.compile(r"^[0-9a-f]{64}$"))
        provenance = json.loads((PACKAGE / "results/source_provenance.json").read_text())
        self.assertRegex(provenance["dataset"]["hf_dataset_revision"], re.compile(r"^[0-9a-f]{40}$"))

    def test_reproduction_entry_points_and_reference_are_present(self):
        required = [
            "README.md", "TEST_PROCEDURE.md", "requirements.txt", "requirements-internvl.txt",
            "configs/eval_protocol.json", "configs/model_registry.json", "scripts/run_dataset.sh",
            "scripts/mm_eval.py", "scripts/aggregate.py", "data/README.md", "models/README.md",
            "results/reference_results.json", "results/source_provenance.json",
        ]
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue((PACKAGE / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
