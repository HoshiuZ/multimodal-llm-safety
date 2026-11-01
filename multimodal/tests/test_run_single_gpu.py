"""Contract checks for the one-GPU orchestration (no model weights required)."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_single_gpu.py"
spec = importlib.util.spec_from_file_location("run_single_gpu", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SingleGpuRunnerTest(unittest.TestCase):
    def test_internvl_uses_its_python_then_common_reviewer_and_one_shard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = root / "models"
            for name in ("InternVL3-8B", "Qwen3-VL-8B-Instruct", "adapters/internvl3-8b-m1-vlguard-v1"):
                (models / name).mkdir(parents=True)
            view = root / "input" / "case"
            view.mkdir(parents=True)
            (view / "manifest.jsonl").write_text('{}\n')
            target = root / "internvl-python"
            target.symlink_to(Path(sys.executable))
            env = {"MM_MODELS_ROOT": str(models), "MM_INPUT_ROOT": str(root / "input"), "MM_OUTPUT_ROOT": str(root / "out"), "MM_INTERNVL_PYTHON": str(target)}
            with patch.dict(os.environ, env), patch.object(sys, "argv", ["run_single_gpu.py", "internvl3_8b"]), patch.object(runner, "device_description", return_value="mock GPU"), patch.object(runner.subprocess, "run") as call:
                self.assertEqual(runner.main(), 0)
            self.assertEqual(call.call_count, 5)
            calls = [x.args[0] for x in call.call_args_list]
            self.assertTrue(all(x.kwargs.get("check") is True for x in call.call_args_list))
            self.assertTrue(all(x.kwargs.get("stderr") == runner.subprocess.STDOUT for x in call.call_args_list))
            self.assertEqual([Path(c[0]) for c in calls[:3]], [target.absolute(), target.absolute(), Path(sys.executable).absolute()])
            self.assertEqual([c[3] for c in calls[:3]], ["generate", "generate", "review"])
            for c in calls[:3]:
                self.assertEqual(c[c.index("--num-shards") + 1], "1")
                self.assertEqual(c[c.index("--device-index") + 1], "0")
            self.assertEqual(calls[0][calls[0].index("--state") + 1], "orig")
            self.assertEqual(calls[1][calls[1].index("--state") + 1], "m1")
            self.assertIn("--adapter", calls[1])
            self.assertNotIn("--adapter", calls[0])
            self.assertEqual([c[c.index("--state") + 1] for c in calls[3:]], ["orig", "m1_m3_v8"])

    def test_missing_input_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("Qwen3-VL-8B-Instruct", "adapters/qwen3vl8b-m1-vlguard-v1"):
                (root / "models" / name).mkdir(parents=True)
            with patch.dict(os.environ, {"MM_MODELS_ROOT": str(root / "models"), "MM_INPUT_ROOT": str(root / "empty")}), patch.object(sys, "argv", ["run_single_gpu.py", "qwen3vl8b"]), patch.object(runner.subprocess, "run") as call:
                with self.assertRaises(FileNotFoundError):
                    runner.main()
                call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
