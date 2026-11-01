"""Exercise generation setup with the released runtime configuration."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE / "scripts"), str(PACKAGE / "src")]
import mm_eval


class StopBeforeModelLoad(Exception):
    pass


class GenerationSetupTests(unittest.TestCase):
    def test_generation_uses_the_actual_baseline_runtime_in_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"sample_id": "one", "query": "What is shown?", "image_path": "a.jpg", "img_sha256": "0" * 64}) + "\n")
            runtime = json.loads((PACKAGE / "configs" / "baseline_runtime.json").read_text())
            registry = json.loads((PACKAGE / "configs" / "model_registry.json").read_text())
            cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, set_device=MagicMock())
            torch = SimpleNamespace(cuda=cuda)
            transformers = SimpleNamespace(__version__="5.16.1")
            argv = ["mm_eval.py", "generate", "--model-key", "qwen3vl8b", "--state", "orig", "--manifest", str(manifest), "--data-root", str(root), "--base-model", str(root / "base"), "--output-dir", str(root / "out"), "--num-shards", "1"]
            with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.object(sys, "argv", argv), patch.object(mm_eval, "tree_fingerprint", return_value=registry["models"]["qwen3vl8b"]["base_tree_sha256"]), patch.object(mm_eval.EB, "load_adapter", side_effect=StopBeforeModelLoad):
                with self.assertRaises(StopBeforeModelLoad):
                    mm_eval.main()
            meta = json.loads((root / "out/manifest.orig.shard-00-of-01.jsonl.meta.json").read_text())
            self.assertEqual(meta["decode"], runtime["decode"])
            self.assertEqual(meta["internvl_preprocess"], runtime["internvl_preprocess"])
            self.assertEqual(meta["model_key"], "qwen3vl8b")


    def test_review_uses_the_frozen_m3_settings(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"sample_id": "one", "query": "What is shown?", "image_path": "a.jpg", "img_sha256": "0" * 64}) + "\n")
            m1 = root / "out/manifest.m1.shard-00-of-01.jsonl"
            m1.parent.mkdir()
            m1.write_text('{}\n')
            Path(str(m1) + '.meta.json').write_text('{"run_id":"previous"}')
            registry = json.loads((PACKAGE / "configs/model_registry.json").read_text())
            protocol = json.loads((PACKAGE / "configs/eval_protocol.json").read_text())
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, set_device=MagicMock()))
            transformers = SimpleNamespace(__version__="5.16.1")
            argv = ["mm_eval.py", "review", "--model-key", "qwen3vl8b", "--manifest", str(manifest), "--data-root", str(root), "--m1-input-dir", str(m1.parent), "--reviewer-model", str(root / "reviewer"), "--output-dir", str(m1.parent), "--num-shards", "1"]
            with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.object(sys, "argv", argv), patch.object(mm_eval, "tree_fingerprint", return_value=registry["reviewer"]["tree_sha256"]), patch.object(mm_eval, "load_existing", return_value={"one": {}}), patch.object(mm_eval.EB, "load_adapter", return_value={}), patch.object(mm_eval.GUARD, "Reviewer", side_effect=StopBeforeModelLoad) as reviewer:
                with self.assertRaises(StopBeforeModelLoad):
                    mm_eval.main()
            self.assertEqual(reviewer.call_args.args[1], protocol["m3"]["settings"])
            meta = json.loads((root / "out/manifest.m1_m3_v8.shard-00-of-01.jsonl.meta.json").read_text())
            self.assertEqual(meta["m3_settings"], protocol["m3"]["settings"])


    def test_model_tree_ignores_python_caches_but_detects_source_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'modeling.py'
            source.write_text('model source v1')
            original = mm_eval.tree_fingerprint(root)
            cache = root / '__pycache__'
            cache.mkdir()
            (cache / 'modeling.cpython-312.pyc').write_bytes(b'generated cache')
            self.assertEqual(mm_eval.tree_fingerprint(root), original)
            source.write_text('model source v2')
            self.assertNotEqual(mm_eval.tree_fingerprint(root), original)


if __name__ == "__main__":
    unittest.main()
