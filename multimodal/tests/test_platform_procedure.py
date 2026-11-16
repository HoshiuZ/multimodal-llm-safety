"""Regression coverage for a fresh evaluator using the Word commands."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE / 'scripts'), str(PACKAGE / 'src')]
import document_steps as D
import evidence_retry as E
import judge_baseline_local as J
import prepare_runtime as P
import run_single_gpu as R


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


class PlatformPreparationTests(unittest.TestCase):
    def test_single_download_plan_includes_all_models_adapters_and_judge_at_pins(self):
        cfg = D.settings()
        plan = P.model_downloads(cfg)
        self.assertEqual(len(plan), 7)
        self.assertEqual({str(p.relative_to(D.REPO)) for _, _, p in plan}, {
            'multimodal/models/Qwen3-VL-8B-Instruct', 'multimodal/models/llama3-llava-next-8b-hf',
            'multimodal/models/InternVL3-8B', 'multimodal/models/Qwen2.5-14B-Instruct',
            'multimodal/models/adapters/qwen3vl8b-m1-vlguard-v1',
            'multimodal/models/adapters/llavanext8b-m1-vlguard-v1',
            'multimodal/models/adapters/internvl3-8b-m1-vlguard-v1',
        })
        self.assertTrue(all(len(revision) == 40 and revision != 'main' for _, revision, _ in plan))

    def test_existing_path_overrides_are_shared_by_checks_and_runner(self):
        with patch.dict(os.environ, {'MM_MODELS_ROOT': '/tmp/evaluator models', 'MM_OUTPUT_ROOT': '/tmp/evaluator outputs'}):
            self.assertEqual(D.settings()['models_root'], R.D.settings()['models_root'])
            self.assertEqual(D.settings()['output_root'], '/tmp/evaluator outputs')

    def test_venv_python_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            executable = Path(td) / 'venv/bin/python'
            executable.parent.mkdir(parents=True)
            executable.symlink_to(sys.executable)
            with patch.dict(os.environ, {'MM_COMMON_PYTHON': str(executable)}):
                self.assertEqual(D.environment_candidates('common')[0][1], executable)

    def test_word_conda_environments_are_discovered_from_active_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            envs = Path(td) / 'envs'
            for name in ('qwen_and_llava', 'intern'):
                executable = envs / name / 'bin/python'
                executable.parent.mkdir(parents=True)
                executable.symlink_to(sys.executable)
            with patch.dict(os.environ, {'CONDA_PREFIX': str(envs / 'intern')}, clear=True), patch.object(D.shutil, 'which', return_value=None):
                self.assertIn(envs / 'qwen_and_llava/bin/python', [exe for _, exe in D.environment_candidates('common')])
                self.assertIn(envs / 'intern/bin/python', [exe for _, exe in D.environment_candidates('InternVL3')])

    def test_judge_public_snapshot_hash_ignores_local_receipt_but_detects_weight_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            weight = root / 'model.safetensors'
            weight.write_bytes(b'checkpoint')
            original, _ = J.fingerprint_model_tree(str(root))
            (root / 'SHA256SUMS').write_text('local receipt')
            (root / '__pycache__').mkdir()
            (root / '__pycache__/model.pyc').write_bytes(b'cache')
            self.assertEqual(J.fingerprint_model_tree(str(root))[0], original)
            weight.write_bytes(b'corrupt checkpoint')
            self.assertNotEqual(J.fingerprint_model_tree(str(root))[0], original)

    def test_corrupt_download_never_replaces_existing_image(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, size): return iter([b'wrong image'])
        with tempfile.TemporaryDirectory() as td:
            image = Path(td) / 'image.png'
            image.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
            session = SimpleNamespace(get=lambda *a, **kw: Response())
            with self.assertRaisesRegex(ValueError, 'SHA-256 differs'):
                P.download_image(session, image, 'multimodal/data/image.png', hashlib.sha256(b'expected image').hexdigest())
            self.assertTrue(image.read_bytes().startswith(b'version https://git-lfs'))
            self.assertEqual(list(image.parent.glob('.download-*')), [])

    def test_environment_failure_sets_process_status_without_download(self):
        with patch.object(D, 'check_environment', return_value=None), patch.object(D.subprocess, 'run') as call:
            with patch.object(sys, 'argv', ['document_steps.py', 'environment']):
                self.assertEqual(D.main(), 1)
            call.assert_not_called()


class EvidenceSupplementTests(unittest.TestCase):
    def fixture(self, root):
        old = [dict(sample_id='resolved', model_key='llavanext8b', scenario='01-Illegal_Activitiy', judgment_status='resolved', benchmark_safety='safe', attack_success=False, harmful_content=False, explicit_refusal=True, judge_prompt_version='v2'),
               dict(sample_id='pending', model_key='llavanext8b', scenario='01-Illegal_Activitiy', judgment_status='unresolved', judge_prompt_version='v2')]
        generation = [dict(sample_id=row['sample_id'], model_key='llavanext8b', scenario=row['scenario'], query='question', response='answer') for row in old]
        gp, jp = root / 'generation.jsonl', root / 'v2.jsonl'
        jsonl(gp, generation); jsonl(jp, old)
        Path(str(jp) + '.complete.json').write_text(json.dumps(dict(model_key='llavanext8b', rows=2, output_sha256=E.sha(jp))))
        args = SimpleNamespace(phase='sequential', model_key='llavanext8b', generation_input=gp, input_judgments=jp, output_dir=root / 'retry', model_path=root / 'judge')
        return args, old

    def replicas(self, args, disagree=False):
        E.prepare(args)
        meta = json.loads((args.output_dir / 'run_metadata.json').read_text())
        for rank in range(6):
            E.save(args.output_dir / f'controls.rank-{rank:02d}.json', dict(run_id=meta['run_id'], rank=rank, passed=True, outcomes=[dict(passed=True)] * 45))
            unsafe = disagree and rank == 5
            labels = dict(benchmark_safety='unsafe' if unsafe else 'safe', harmful_content=unsafe, explicit_refusal=not unsafe)
            record = dict(sample_id='pending', status='resolved', labels=labels, attempts=[dict(raw=json.dumps(labels))], prompt_sha256='prompt')
            E.save(args.output_dir / f'replica.rank-{rank:02d}.json', dict(run_id=meta['run_id'], rank=rank, records=[record]))

    def test_six_agreeing_replicas_preserve_resolved_v2_and_enable_counting(self):
        with tempfile.TemporaryDirectory() as td:
            args, old = self.fixture(Path(td)); original = E.sha(args.input_judgments)
            self.replicas(args); E.merge(args)
            out = args.output_dir / 'judgments.v2_plus_v2r.canonical.jsonl'
            merged = E.file_rows(out)
            self.assertEqual(merged[0], old[0])
            self.assertEqual(merged[1]['source_v2_record'], old[1])
            self.assertEqual(E.sha(args.input_judgments), original)
            self.assertEqual(merged[1]['supplement_protocol'], E.SEQUENTIAL_VERSION)
            summary = D.load_labels(dict(path=str(out)), 'llavanext8b', ['resolved', 'pending'])
            self.assertEqual((summary['n'], summary['unresolved_ids'], summary['attack_successes']), (2, [], 0))

    def test_disagreement_or_failed_control_cannot_be_turned_into_a_metric(self):
        with tempfile.TemporaryDirectory() as td:
            args, _ = self.fixture(Path(td)); self.replicas(args, disagree=True); E.merge(args)
            out = args.output_dir / 'judgments.v2_plus_v2r.canonical.jsonl'
            with self.assertRaisesRegex(ValueError, 'unresolved judgment'):
                D.load_labels(dict(path=str(out)), 'llavanext8b', ['resolved', 'pending'])
            control = args.output_dir / 'controls.rank-05.json'
            data = json.loads(control.read_text()); data['passed'] = False; E.save(control, data)
            with self.assertRaisesRegex(ValueError, 'controls incomplete'):
                E.merge(args)
            data['passed'] = True; data['outcomes'][0]['passed'] = False; E.save(control, data)
            with self.assertRaisesRegex(ValueError, 'controls incomplete'):
                E.merge(args)

    def test_sequential_execution_requires_all_270_controls_and_runs_six_replicas_on_one_device(self):
        with tempfile.TemporaryDirectory() as td:
            args, _ = self.fixture(Path(td))
            cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda device: None)
            torch = SimpleNamespace(cuda=cuda)
            def evaluate(tokenizer, model, batch, torch, device):
                self.assertEqual(device, 'cuda:0')
                return [dict(sample_id=row['sample_id'], status='resolved', labels=row.get('expected', dict(benchmark_safety='safe', harmful_content=False, explicit_refusal=True)), attempts=[dict(raw='answer')], prompt_sha256='prompt') for row in batch]
            registry = D.read_json(PACKAGE / 'configs/model_registry.json')
            with patch.dict(sys.modules, {'torch': torch, 'transformers': SimpleNamespace(__version__='5.16.1')}), patch.object(E.J, 'fingerprint_model_tree', return_value=(registry['judge']['tree_sha256'], [])), patch.object(E.J, 'load_judge_model', return_value=(object(), object())), patch.object(E, 'evaluate', side_effect=evaluate) as call:
                E.sequential(args)
            self.assertEqual(call.call_count, 12)
            complete = D.read_json(args.output_dir / 'judgments.v2_plus_v2r.canonical.jsonl.complete.json')
            self.assertEqual((complete['controls_passed'], complete['unresolved_rows']), (270, 0))

    def test_word_judge_automatically_selects_supplement_and_preserves_original_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args, old = self.fixture(root)
            cfg = dict(input_root=str(root / 'input'), output_root=str(root / 'outputs'), models_root=str(root / 'models'))
            for view in ('full', 'base', 'figstep', 'hades'):
                jsonl(root / f'input/{view}/manifest.jsonl', [dict(sample_id=row['sample_id']) for row in old])
                for state in D.STATES:
                    records = [dict(row) for row in old]
                    if (view, state) != ('figstep', 'm1_m3_v8'):
                        records[1].update(judgment_status='resolved', benchmark_safety='safe', attack_success=False, harmful_content=False, explicit_refusal=True)
                    destination = root / f'outputs/llavanext8b/{view}/manifest.{state}.judgments.jsonl'
                    jsonl(destination, records)
                    Path(str(destination) + '.complete.json').write_text(json.dumps(dict(model_key='llavanext8b', rows=2, output_sha256=E.sha(destination))))
                    jsonl(root / f'outputs/llavanext8b/{view}/manifest.{state}.jsonl', E.file_rows(args.generation_input))
            raw = root / 'outputs/llavanext8b/figstep/manifest.m1_m3_v8.judgments.jsonl'
            before = E.sha(raw)
            def run(command, **kwargs):
                retry_args = SimpleNamespace(phase='sequential', model_key='llavanext8b', generation_input=Path(command[command.index('--generation-input') + 1]), input_judgments=raw, output_dir=Path(command[command.index('--output-dir') + 1]))
                self.replicas(retry_args); E.merge(retry_args)
            with patch.object(D.subprocess, 'run', side_effect=run) as call:
                D.judge(cfg, 'llavanext8b')
            self.assertEqual(call.call_count, 1)
            self.assertEqual(E.sha(raw), before)
            selected = D.judgment_source(cfg, 'llavanext8b', 'figstep', 'm1_m3_v8')
            self.assertNotEqual(Path(selected['path']), raw)
            self.assertEqual(D.load_labels(selected, 'llavanext8b', ['resolved', 'pending'])['unresolved_ids'], [])
            # New generation bytes invalidate an older supplemented result.
            generated = root / 'outputs/llavanext8b/figstep/manifest.m1_m3_v8.jsonl'
            generated.write_text(generated.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                D.judgment_source(cfg, 'llavanext8b', 'figstep', 'm1_m3_v8')


if __name__ == '__main__':
    unittest.main()
