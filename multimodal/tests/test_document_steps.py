import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/document_steps.py'
SPEC = importlib.util.spec_from_file_location('document_steps', SCRIPT)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class JudgmentValidationTest(unittest.TestCase):
    def test_checkpoint_cache_is_ignored_but_weight_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            weight = root / 'model.safetensors'
            weight.write_bytes(b'frozen checkpoint')
            before = D.tree_digest(root)
            cache = root / '__pycache__'
            cache.mkdir()
            (cache / 'runtime.pyc').write_bytes(b'generated cache')
            self.assertEqual(D.tree_digest(root), before)
            weight.write_bytes(b'changed checkpoint')
            self.assertNotEqual(D.tree_digest(root), before)

    def fixture(self, root, prefix=''):
        filename = root / 'labels.jsonl'
        records = [dict(sample_id=prefix + sid, model_key='qwen3vl8b', judgment_status='resolved', attack_success=unsafe, harmful_content=unsafe, explicit_refusal=not unsafe, benchmark_safety='unsafe' if unsafe else 'safe', judge_prompt_version='local-v2') for sid, unsafe in [('a', True), ('b', False)]]
        filename.write_text(''.join(json.dumps(r) + '\n' for r in records))
        marker = Path(str(filename) + '.complete.json')
        marker.write_text(json.dumps(dict(model_key='qwen3vl8b', rows=2, output_sha256=D.digest(filename))))
        return filename, marker, records

    def test_counts_come_from_paired_judgments_with_explicit_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            filename, _, _ = self.fixture(Path(td), 'base::orig::')
            value = D.load_labels(dict(path=str(filename), id_prefix='base::orig::'), 'qwen3vl8b', ['a', 'b'])
            self.assertEqual((value['n'], value['attack_successes']), (2, 1))

    def test_file_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            filename, _, _ = self.fixture(Path(td))
            filename.write_text(filename.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'completion marker'):
                D.load_labels(dict(path=str(filename)), 'qwen3vl8b', ['a', 'b'])

    def test_mixed_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            filename, marker, records = self.fixture(Path(td))
            records[0]['model_key'] = 'internvl3_8b'
            filename.write_text(''.join(json.dumps(r) + '\n' for r in records))
            marker.write_text(json.dumps(dict(model_key='qwen3vl8b', rows=2, output_sha256=D.digest(filename))))
            with self.assertRaisesRegex(ValueError, 'foreign model'):
                D.load_labels(dict(path=str(filename)), 'qwen3vl8b', ['a', 'b'])

    def test_unresolved_labels_do_not_emit_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            filename, marker, records = self.fixture(Path(td))
            records[1]['judgment_status'] = 'unresolved'
            filename.write_text(''.join(json.dumps(r) + '\n' for r in records))
            marker.write_text(json.dumps(dict(model_key='qwen3vl8b', rows=2, output_sha256=D.digest(filename))))
            with self.assertRaisesRegex(ValueError, 'unresolved judgment'):
                D.load_labels(dict(path=str(filename)), 'qwen3vl8b', ['a', 'b'])

    def test_extra_or_missing_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            filename, _, _ = self.fixture(Path(td))
            with self.assertRaisesRegex(ValueError, 'judgment IDs'):
                D.load_labels(dict(path=str(filename)), 'qwen3vl8b', ['a', 'different'])

    def test_conflicting_safety_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            filename, marker, records = self.fixture(Path(td))
            records[0]['benchmark_safety'] = 'safe'
            filename.write_text(''.join(json.dumps(r) + '\n' for r in records))
            marker.write_text(json.dumps(dict(model_key='qwen3vl8b', rows=2, output_sha256=D.digest(filename))))
            with self.assertRaisesRegex(ValueError, 'labels conflict'):
                D.load_labels(dict(path=str(filename)), 'qwen3vl8b', ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
