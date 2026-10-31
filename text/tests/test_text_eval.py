"""Benign synthetic CPU fixtures; never research/model evaluation results."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import digest, canonical, read, write, loads
from eval.data import validate_rows, load_bundle, COUNTS
from eval.metrics import c_value, metrics
from eval.scoring import option_sequences, score_one, token_preflight
from eval.overlay import read_overlay, restore_overlay
from eval.paired_audit import audit_pair
from eval.s_audit import audit_s_pair
from eval.worker import run_worker, records, aggregate
from eval import models


def row(i=0, domain='bio'):
    return {'question_id': f'synthetic-{i}', 'source_row_sha256': digest(['synthetic', i]),
            'domain': domain, 'question': 'Which color is named first?',
            'choices': ['red', 'green', 'blue', 'yellow'], 'answer_index': 0}


def bundle():
    return {'kind': 'S', 'source': 'benign-test-only', 'scenes': {
        name: [row(i, domain) for i, domain in enumerate(('bio', 'cyber', 'chem'))]
        for name in ('original', 'syntax', 'paraphrase')}}


class Tokenizer:
    pad_token_id, eos_token_id = 0, 1
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        # A-D span different lengths and use distinct token IDs.
        for i, letter in enumerate('ABCD'):
            if text.endswith(' ' + letter):
                return [2, 3] + [4 + i] * (i + 1)
        return [2, 3]


class TestMetrics(unittest.TestCase):
    def test_historical_s_integer_counts(self):
        root = Path(__file__).resolve().parents[1]
        ref = read(root / 'results/reference_results.json')
        for family, expected in [('qwen', 95.26597749421202), ('internvl', 89.61748633879782), ('llava', 94.2006006006006)]:
            got = metrics(ref['models'][family]['S_counts'], 'S')
            self.assertAlmostEqual(got['S_percent'], expected, places=11)
        self.assertAlmostEqual(float(100*c_value(2272,1285,3668)),72.84132841328413)
        self.assertAlmostEqual(float(100*c_value(2173,1177,3668)),79.29936305732484)
        self.assertNotEqual(c_value(421,257,734), c_value(2272,1285,3668))

    def test_edge_metrics(self):
        self.assertLess(c_value(8,9,20),0)
        self.assertGreater(c_value(8,0,20),1)
        self.assertIsNone(c_value(5,1,20))
        self.assertIsNone(c_value(4,1,20))
        counts={k:{'base':8,'cleared':7 if k=='original' else 0,'n':20} for k in ('original','syntax','paraphrase')}
        self.assertGreater(metrics(counts,'S')['S_percent'],100)
        counts['syntax']['base']=5
        self.assertIsNone(metrics(counts,'S')['S_percent'])
        counts={k:{'base':8,'cleared':8,'n':20} for k in counts}
        self.assertIsNone(metrics(counts,'S')['S_percent'])
        with self.assertRaises(ValueError):c_value(True,0,4)

    def test_different_scene_denominators(self):
        counts={'original':{'base':421,'cleared':257,'n':734},'syntax':{'base':410,'cleared':261,'n':734},'paraphrase':{'base':422,'cleared':264,'n':734}}
        out=metrics(counts,'S',82)
        self.assertTrue(out['passed'])
        self.assertEqual(out['C']['syntax']['exact'],[298,453])
        counts['syntax']['cleared']=430
        self.assertLess(metrics(counts,'S')['S_percent'],0)


class TestSchema(unittest.TestCase):
    def test_duplicate_keys_roundtrip(self):
        value={'continuation_lengths':{'1':12,'2':8}}
        self.assertEqual(loads(canonical(value)),value)
        with self.assertRaises(ValueError):loads('{"x":1,"x":2}')
        with self.assertRaises(ValueError):loads('{"x":NaN}')

    def test_counts_pairing_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[]
            for domain,n in COUNTS['S'].items():
                rows.extend(row(len(rows)+i,domain) for i in range(n))
            for scene in ('original','syntax','paraphrase'):
                (root/f'{scene}.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows))
            manifest={'schema':'text-eval-v1','kind':'S','source':'synthetic-only','scenes':{s:s+'.jsonl' for s in ('original','syntax','paraphrase')}}
            write(root/'manifest.json',manifest)
            good=load_bundle(root/'manifest.json')
            self.assertEqual(len(good['scenes']['original']),734)
            for bad in [rows[:-1],rows+[row(9999)],rows+[rows[0]]]:
                (root/'syntax.jsonl').write_text(''.join(canonical(r)+'\n' for r in bad))
                with self.assertRaises(ValueError):load_bundle(root/'manifest.json')
            bad=copy.deepcopy(rows);bad[0]['answer_index']=1
            (root/'syntax.jsonl').write_text(''.join(canonical(r)+'\n' for r in bad))
            with self.assertRaises(ValueError):load_bundle(root/'manifest.json')
        with self.assertRaises(ValueError):validate_rows([row(),row()])


class TestWorker(unittest.TestCase):
    def test_resume_identity_and_strict_coverage(self):
        b=bundle();identity={'model':'fixture-a','code':'fixture-code','prompt':'fixture-prompt','data':digest(b)}
        calls=[]
        def scorer(r):
            calls.append(r)
            return {'A':-1.,'B':-2.,'C':-3.,'D':-4.}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(run_worker(b,identity,'base',root,scorer),9)
            self.assertEqual(run_worker(b,identity,'base',root,scorer),9)
            self.assertEqual(len(calls),9)
            # Simulate interruption after four valid records, then reuse exactly those rows.
            path=root/'base.jsonl';lines=path.read_text().splitlines(True)
            path.write_text(''.join(lines[:4]));calls.clear()
            run_worker(b,identity,'base',root,scorer)
            self.assertEqual(len(calls),5)
            for key in ('model','code','prompt','data'):
                with self.assertRaises(ValueError):run_worker(b,{**identity,key:'changed'},'base',root,scorer)
            with self.assertRaises(ValueError):aggregate(b,identity,root)
            run_worker(b,identity,'cleared',root,scorer)
            self.assertEqual(aggregate(b,identity,root)['counts']['original']['base'],3)
            lines=path.read_text().splitlines(True)
            for bad in [lines[:-1],lines+[lines[0]]]:
                path.write_text(''.join(bad))
                with self.assertRaises(ValueError):aggregate(b,identity,root)
            extra=json.loads(lines[0]);extra['question_id']='synthetic-extra'
            path.write_text(''.join(lines)+canonical(extra)+'\n')
            with self.assertRaises(ValueError):aggregate(b,identity,root)
            fail=json.loads(lines[0]);fail['status']='failed'
            path.write_text(canonical(fail)+'\n'+''.join(lines[1:]))
            with self.assertRaises(ValueError):aggregate(b,identity,root)
            path.write_text(''.join(lines).rstrip('\n'))
            with self.assertRaises(ValueError):run_worker(b,identity,'base',root,scorer)


class TestScoring(unittest.TestCase):
    def test_token_boundaries_and_overflow(self):
        tok=Tokenizer();ctx,seqs=option_sequences(tok,row(),8)
        self.assertEqual([len(v)-len(ctx) for v in seqs.values()],[1,2,3,4])
        with self.assertRaises(ValueError):option_sequences(tok,row(),5)
        class Bad(Tokenizer):
            def encode(self,text,add_special_tokens=False):
                return [9] if text.endswith(' A') else super().encode(text,add_special_tokens)
        with self.assertRaises(ValueError):option_sequences(Bad(),row(),8)
        report=token_preflight(tok,bundle(),8)
        self.assertEqual(read_json_roundtrip(report),report)

    def test_actual_worker_forward_dispatch_and_llmask(self):
        import torch
        torch.set_num_threads(1)
        tok=Tokenizer()
        class Fixture:
            def __init__(self):self.calls=0;self.language_model=self;self.full_calls=0
            def __call__(self,input_ids,attention_mask,use_cache,return_dict):
                self.calls+=1
                assert not use_cache and return_dict
                assert input_ids.device.type=='cpu'
                self.ids=input_ids.clone();self.mask=attention_mask.clone()
                # Distinct logits by position to expose wrong last-position indexing.
                logits=torch.arange(4*6*12,dtype=torch.float32).reshape(4,6,12)%17
                return SimpleNamespace(logits=logits)
        for family in ('qwen','internvl','llava'):
            fixture=Fixture()
            got=score_one(fixture,tok,family,row(),8,'cpu')
            ctx,seqs=option_sequences(tok,row(),8)
            logits=torch.arange(4*6*12,dtype=torch.float32).reshape(4,6,12)%17
            for i,letter in enumerate('ABCD'):
                expected=sum(torch.log_softmax(logits[i,p-1],dim=-1)[seqs[letter][p]].item() for p in range(2,len(seqs[letter])))
                self.assertAlmostEqual(got[letter],expected,places=6)
            self.assertEqual(fixture.mask.sum(1).tolist(),[3,4,5,6])
            b=bundle();identity={'fixture':family}
            with tempfile.TemporaryDirectory() as tmp:
                run_worker(b,identity,'base',tmp,lambda r:score_one(fixture,tok,family,r,8,'cpu'))
                self.assertEqual(fixture.calls,10)
        class Wrapper:
            language_model=Fixture()
            def __call__(self,**kwargs):raise AssertionError('InternVL image wrapper forward called')
        score_one(Wrapper(),tok,'internvl',row(),8,'cpu')
        self.assertFalse(torch.cuda.is_initialized())


class TestSparseOverlay(unittest.TestCase):
    def test_cuda_selection_requires_one_unambiguous_visible_device(self):
        import importlib.util
        script=Path(__file__).resolve().parents[1]/'scripts/sparse_eval.py'
        spec=importlib.util.spec_from_file_location('pkg2_sparse_cli_fixture',script)
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        self.assertEqual(cli.resolve_cuda_selection(True,1,None),'cuda:0')
        self.assertEqual(cli.resolve_cuda_selection(True,1,'GPU-unique'),'cuda:0')
        for available,count,visible in [(False,0,None),(True,0,None),(True,2,None),(True,2,'GPU-0'),(True,1,'GPU-0,GPU-1')]:
            with self.assertRaises(RuntimeError):cli.resolve_cuda_selection(available,count,visible)

    def test_other49_uses_exact_eighty_percent_integer_rule(self):
        rows = []
        for i in range(10):
            rows.append({'question_id': f'mmlu-{i}', 'source_row_sha256': digest(['mmlu', i]),
                         'subject': f'subject-{i % 2}', 'description': 'official description',
                         'question': 'synthetic question', 'choices': ['a', 'b', 'c', 'd'],
                         'answer_index': 0})
        b = {'kind': 'M', 'scenes': {'original': rows}}
        contract = digest({'model': 'synthetic-mmlu', 'code': 'synthetic-code',
                           'prompt': 'synthetic-prompt', 'data': digest(b)})
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for state in ('base', 'cleared'):
                path = Path(tmp) / f'{state}.jsonl'; paths[state] = path
                with path.open('w', encoding='utf-8') as stream:
                    for i, row in enumerate(rows):
                        prediction = 'B' if state == 'cleared' and i < 2 else 'A'
                        scores = {'A': -1.0, 'B': -2.0, 'C': -3.0, 'D': -4.0}
                        if prediction == 'B': scores['A'], scores['B'] = -2.0, -1.0
                        stream.write(canonical({'state': state, 'scene': 'original',
                            'contract': contract,
                            'question_id': row['question_id'], 'row_digest': digest(row),
                            'scores': scores, 'prediction': prediction, 'target': 'A',
                            'correct': prediction == 'A', 'status': 'ok'}) + '\n')
            got = audit_pair(b, paths['base'], paths['cleared'],
                             base_contract=contract, cleared_contract=contract)
            self.assertEqual(got['other49']['base_correct'], 10)
            self.assertEqual(got['other49']['cleared_correct'], 8)
            self.assertEqual(got['other49']['rule'], '5*n_cleared >= 4*n_base')
            self.assertTrue(got['other49']['passes_20pct_loss'])
            with paths['cleared'].open('w', encoding='utf-8') as stream:
                for i, row in enumerate(rows):
                    prediction = 'B' if i < 3 else 'A'
                    scores = {'A': -1.0, 'B': -2.0, 'C': -3.0, 'D': -4.0}
                    if prediction == 'B': scores['A'], scores['B'] = -2.0, -1.0
                    stream.write(canonical({'state': 'cleared', 'scene': 'original',
                        'contract': contract,
                        'question_id': row['question_id'], 'row_digest': digest(row),
                        'scores': scores, 'prediction': prediction, 'target': 'A',
                        'correct': prediction == 'A', 'status': 'ok'}) + '\n')
            got = audit_pair(b, paths['base'], paths['cleared'],
                             base_contract=contract, cleared_contract=contract)
            self.assertEqual(got['other49']['cleared_correct'], 7)
            self.assertFalse(got['other49']['passes_20pct_loss'])

    def test_pair_audit_requires_contract_on_every_record(self):
        rows = [
            {'question_id': f'mmlu-{i}', 'source_row_sha256': digest(['strict-contract', i]),
             'subject': 'synthetic_subject', 'description': 'synthetic description',
             'question': 'synthetic question', 'choices': ['a', 'b', 'c', 'd'],
             'answer_index': 0}
            for i in range(2)
        ]
        b = {'kind': 'M', 'scenes': {'original': rows}}
        contract = digest({'model': 'synthetic-mmlu', 'data': digest(b)})
        other_contract = digest({'model': 'different-synthetic-mmlu', 'data': digest(b)})
        with tempfile.TemporaryDirectory() as tmp:
            paths = {state: Path(tmp) / f'{state}.jsonl' for state in ('base', 'cleared')}

            def write_state(state, record_contract):
                with paths[state].open('w', encoding='utf-8') as stream:
                    for row in rows:
                        stream.write(canonical({'state': state, 'scene': 'original',
                            'contract': record_contract, 'question_id': row['question_id'],
                            'row_digest': digest(row), 'scores': {'A': -1.0, 'B': -2.0, 'C': -3.0, 'D': -4.0},
                            'prediction': 'A', 'target': 'A', 'correct': True, 'status': 'ok'}) + '\n')

            write_state('base', contract)
            write_state('cleared', contract)
            good = audit_pair(b, paths['base'], paths['cleared'],
                              base_contract=contract, cleared_contract=contract)
            self.assertTrue(good['coverage']['paired_identity_match'])

            write_state('cleared', other_contract)
            with self.assertRaisesRegex(ValueError, 'expected contract'):
                audit_pair(b, paths['base'], paths['cleared'],
                           base_contract=contract, cleared_contract=contract)
            with self.assertRaisesRegex(ValueError, 'contract identities differ'):
                audit_pair(b, paths['base'], paths['cleared'])

            write_state('cleared', contract)
            lines = paths['base'].read_text(encoding='utf-8').splitlines()
            missing = loads(lines[0]); missing.pop('contract')
            lines[0] = canonical(missing)
            paths['base'].write_text('\n'.join(lines) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'lacks a nonempty SHA256 contract'):
                audit_pair(b, paths['base'], paths['cleared'])

    def test_pair_audit_explicitly_rejects_S_bundle(self):
        with self.assertRaisesRegex(ValueError, 'C/M only.*S aggregation'):
            audit_pair(bundle(), 'unused-base.jsonl', 'unused-cleared.jsonl')

    def test_safe_overlay_restore_then_real_worker_scoring(self):
        import torch
        torch.set_num_threads(1)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Module()
                self.mlp.down_proj = torch.nn.Linear(2, 2, bias=False)

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.ModuleList([Block()])
                self.calls = 0

            def forward(self, input_ids, attention_mask, **kwargs):
                self.calls += 1
                logits = torch.zeros((*input_ids.shape, 12), dtype=torch.float32)
                logits[..., 4] = 2.0
                return SimpleNamespace(logits=logits)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'update.pt'
            key = 'model.language_model.layers.0.mlp.down_proj.weight'
            tensor = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32)
            torch.save({key: tensor}, path)
            payload = path.read_bytes()
            config = {'update': {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                                 'parameters': {key: {'shape': [2, 2], 'dtype': 'torch.float32'}}}}
            state = read_overlay(path, config)
            wrapper = Wrapper()
            report = restore_overlay(wrapper, state, config)
            self.assertTrue(report['all_bitwise_equal'])
            self.assertTrue(torch.equal(dict(wrapper.named_parameters())[key].detach(), tensor))
            b = bundle()
            identity = {'fixture': 'safe-overlay-and-forward'}
            run_worker(b, identity, 'base', Path(tmp) / 'scores',
                       lambda r: score_one(wrapper, Tokenizer(), 'qwen', r, 8, 'cpu'))
            self.assertEqual(wrapper.calls, 9)
            self.assertEqual(len(records(Path(tmp) / 'scores/base.jsonl', b, digest(identity), 'base')), 9)
            self.assertFalse(torch.cuda.is_initialized())

    def test_strict_full_load(self):
        import torch
        import transformers
        config={'family':'qwen','model_class':'Qwen3VLForConditionalGeneration','attention':'sdpa'}
        for info in ({'missing_keys':['synthetic.weight']},{'unexpected_keys':['synthetic.weight']},{'mismatched_keys':['synthetic.weight']}):
            with patch.object(transformers.Qwen3VLForConditionalGeneration,'from_pretrained',return_value=(object(),info)):
                with self.assertRaises(ValueError):models.load_full('external',config,'cpu')



class TestCli(unittest.TestCase):
    def test_real_cli_preflight_run_resume_aggregate(self):
        import importlib.util
        import torch
        script=Path(__file__).resolve().parents[1]/'scripts/text_eval.py'
        spec=importlib.util.spec_from_file_location('pkg1_cli_fixture',script)
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        class Fixture:
            def __call__(self,input_ids,attention_mask,**kwargs):
                return SimpleNamespace(logits=torch.zeros((*input_ids.shape,12)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=root/'config.json'
            write(cfg,{'family':'qwen'})
            b=bundle();b['digest']=digest(b)
            common=['--config',str(cfg),'--data','synthetic-manifest','--base-model','synthetic-base',
                    '--cleared-model','synthetic-cleared','--preflight',str(root/'pf.json')]
            with patch.object(cli,'load_bundle',return_value=b), patch.object(cli.models,'check_environment',return_value={'fixture':True}), \
                 patch.object(cli.models,'inventory',side_effect=lambda p,c:{'files':{'tokenizer.json':{'sha256':'synthetic'}},'digest':digest(str(p))}), \
                 patch.object(cli.models,'context_limit',return_value=8), patch.object(cli.models,'tokenizer',return_value=Tokenizer()), \
                 patch.object(cli.models,'load_full',return_value=Fixture()):
                commands=[['preflight']+common]
                commands += [['run']+common+['--state',s,'--device','cpu','--output',str(root/'out')] for s in ('base','base','cleared')]
                commands += [['aggregate']+common+['--output',str(root/'out')]]
                for args in commands:
                    with patch.object(sys,'argv',[str(script)]+args):cli.main()
                result=read(root/'out/aggregate.json')
                self.assertEqual(result['counts']['original']['n'],3)
                self.assertEqual(len((root/'out/base.jsonl').read_text().splitlines()),9)
                self.assertFalse(torch.cuda.is_initialized())


class TestSTextAudit(unittest.TestCase):
    scene_counts = {'original': (500, 300), 'syntax': (490, 310), 'paraphrase': (480, 290)}

    def _fixture(self, root):
        scenes = {}
        domains = [('bio', 255), ('cyber', 397), ('chem', 82)]
        for scene in ('original', 'syntax', 'paraphrase'):
            rows = []
            index = 0
            for domain, count in domains:
                for _ in range(count):
                    item = row(index, domain)
                    item['question_id'] = f'synthetic-{index}'
                    item['question'] = f'{scene}: benign color question {index}'
                    rows.append(item)
                    index += 1
            scenes[scene] = rows
            (root / f'{scene}.jsonl').write_text(
                ''.join(canonical(item) + '\n' for item in rows), encoding='utf-8')
        manifest = {'schema': 'text-eval-v1', 'kind': 'S', 'source': 'synthetic-only',
                    'scenes': {scene: f'{scene}.jsonl' for scene in scenes}}
        write(root / 'manifest.json', manifest)
        loaded = load_bundle(root / 'manifest.json')
        identity = {'model': 'synthetic-s-model', 'prompt': 'synthetic-only',
                    'data': loaded['digest']}
        base_contract = digest(identity)
        paths = {}
        for state in ('base', 'cleared'):
            state_dir = root / state
            state_dir.mkdir()
            write(state_dir / 'contract.json', identity)
            path = state_dir / f'{state}.jsonl'
            paths[state] = path
            with path.open('w', encoding='utf-8') as stream:
                for scene, rows in loaded['scenes'].items():
                    target_count = self.scene_counts[scene][0 if state == 'base' else 1]
                    for index, item in enumerate(rows):
                        correct = index < target_count
                        scores = {'A': -0.1, 'B': -1.0, 'C': -2.0, 'D': -3.0} if correct else {
                            'A': -1.0, 'B': -0.1, 'C': -2.0, 'D': -3.0}
                        prediction = 'A' if correct else 'B'
                        record = {'contract': base_contract, 'state': state, 'scene': scene,
                                  'question_id': item['question_id'], 'row_digest': digest(item),
                                  'scores': scores, 'prediction': prediction, 'target': 'A',
                                  'correct': correct, 'status': 'ok'}
                        stream.write(canonical(record) + '\n')
        return loaded, paths

    def test_734_denominator_exact_s_and_strict_pairing(self):
        from fractions import Fraction
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b, paths = self._fixture(root)
            result = audit_s_pair(b, paths['base'], paths['cleared'],
                                  root / 'base/contract.json', root / 'cleared/contract.json')
            self.assertEqual(result['coverage']['expected_per_state'], 2202)
            self.assertEqual(result['coverage']['base'], 2202)
            self.assertEqual(result['coverage']['cleared'], 2202)
            self.assertEqual(result['coverage']['failed'], 0)
            self.assertEqual(result['coverage']['missing'], 0)
            self.assertEqual(result['coverage']['duplicate'], 0)
            self.assertEqual(result['coverage']['extra'], 0)
            expected_c = {
                'original': Fraction(400, 633),
                'syntax': Fraction(360, 613),
                'paraphrase': Fraction(380, 593),
            }
            for scene, exact in expected_c.items():
                self.assertEqual(result['C'][scene]['exact'], [exact.numerator, exact.denominator])
            expected_s = 100 * min(expected_c['syntax'] / expected_c['original'],
                                   expected_c['paraphrase'] / expected_c['original'])
            self.assertEqual(result['S']['exact'], [expected_s.numerator, expected_s.denominator])
            self.assertEqual(result['S']['threshold_percent'], 65)
            self.assertFalse(result['S']['clamped'])
            with self.assertRaisesRegex(ValueError, 'C/M only.*S aggregation'):
                audit_pair(b, paths['base'], paths['cleared'])

    def test_real_audit_s_cli_and_bad_record_rejection(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b, paths = self._fixture(root)
            script = Path(__file__).resolve().parents[1] / 'scripts/sparse_eval.py'
            output = root / 'cli-audit.json'
            command = [sys.executable, str(script), 'audit-s', '--data', str(root / 'manifest.json'),
                       '--base-scores', str(paths['base']), '--cleared-scores', str(paths['cleared']),
                       '--base-contract', str(root / 'base/contract.json'),
                       '--cleared-contract', str(root / 'cleared/contract.json'), '--output', str(output)]
            run = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn('"expected_per_state": 2202', run.stdout)
            self.assertEqual(read(output)['coverage']['paired_identity_match'], True)

            original = paths['base'].read_text(encoding='utf-8').splitlines(True)
            paths['base'].write_text(''.join(original[:-1]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'incomplete coverage'):
                audit_s_pair(b, paths['base'], paths['cleared'],
                             root / 'base/contract.json', root / 'cleared/contract.json')

            paths['base'].write_text(''.join(original + [original[0]]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate record scope'):
                audit_s_pair(b, paths['base'], paths['cleared'],
                             root / 'base/contract.json', root / 'cleared/contract.json')

            paths['base'].write_text(''.join(original), encoding='utf-8')
            with paths['base'].open('a', encoding='utf-8') as stream:
                stream.write(canonical({'contract': digest({'extra': True}), 'state': 'base',
                                        'scene': 'original', 'question_id': 'synthetic-extra'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'extra record scope'):
                audit_s_pair(b, paths['base'], paths['cleared'],
                             root / 'base/contract.json', root / 'cleared/contract.json')

    def test_multi_token_llsum_accumulates_each_continuation(self):
        import torch
        class MultiTokenizer:
            pad_token_id, eos_token_id = 0, 1
            def encode(self, text, add_special_tokens=False):
                if text.endswith('Answer:'):
                    return [10, 11, 12]
                for index, letter in enumerate('ABCD'):
                    if text.endswith(' ' + letter):
                        return [10, 11, 12, 100 + index, 150 + index]
                raise AssertionError('unexpected tokenizer input')
        class MultiTokenModel:
            def __call__(self, input_ids, attention_mask, **kwargs):
                logits = torch.zeros((*input_ids.shape, 256), dtype=torch.float32)
                for index in range(4):
                    first = int(input_ids[index, 3])
                    second = int(input_ids[index, 4])
                    logits[index, 2, first] = 1.0
                    logits[index, 3, second] = 2.0
                return SimpleNamespace(logits=logits)
        scores = score_one(MultiTokenModel(), MultiTokenizer(), 'qwen', row(), 8, 'cpu')
        expected = (1.0 - math.log(math.exp(1.0) + 255.0)) + (2.0 - math.log(math.exp(2.0) + 255.0))
        self.assertEqual(set(scores), set('ABCD'))
        for score in scores.values():
            self.assertAlmostEqual(score, expected, places=6)


class TestSparseEvalCli(unittest.TestCase):
    def test_cpu_cli_restore_base_resume_cleared_and_s_aggregate(self):
        import importlib.util
        import torch
        script = Path(__file__).resolve().parents[1] / 'scripts/sparse_eval.py'
        spec = importlib.util.spec_from_file_location('pkg3_sparse_eval_fixture', script)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)

        class FixtureModel:
            pass

        synthetic = bundle()
        identity = {'schema': 'fixture-contract', 'model': 'synthetic-full-wrapper',
                    'config': {'family': 'qwen', 'base_revision': 'fixture-revision',
                               'model_class': 'FixtureModel',
                               'update': {'sha256': 'fixture-overlay-sha256'}}, 'data': digest(synthetic),
                    'tokens': {'token_stream_sha256': 'synthetic-token-stream'}}
        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.json'
            write(config, identity['config'])
            base_scores = root / 'base'
            cleared_scores = root / 'cleared'
            common = ['--config', str(config), '--data', 'synthetic-data',
                      '--base-model', 'synthetic-base', '--overlay', 'synthetic-overlay']

            with patch.object(cli, 'prepare', return_value=(identity, synthetic, tokenizer, 8)):
                with patch.object(sys, 'argv', [str(script), 'preflight'] + common + ['--output', str(root / 'preflight.json')]):
                    cli.main()
            self.assertEqual(read(root / 'preflight.json'), identity)

            def runtime(_contract, _config, _data, _base, _overlay, _state):
                update = {'synthetic': True}
                return identity, synthetic, tokenizer, 8, update

            score_calls = []
            def synthetic_score(_model, _tokenizer, _family, _row, _limit, _device):
                score_calls.append(1)
                return {'A': -0.1, 'B': -1.0, 'C': -2.0, 'D': -3.0}

            with patch.object(cli, 'validate_runtime_contract', side_effect=runtime), \
                 patch.object(cli.models, 'load_full', return_value=FixtureModel()), \
                 patch.object(cli, 'restore_overlay', return_value={'all_bitwise_equal': True}), \
                 patch.object(cli, 'score_one', side_effect=synthetic_score):
                for state, output in (('base', base_scores), ('base', base_scores), ('cleared', cleared_scores)):
                    argv = [str(script), 'run'] + common + ['--contract', str(root / 'preflight.json'),
                        '--state', state, '--device', 'cpu', '--output', str(output)]
                    with patch.object(sys, 'argv', argv):
                        cli.main()
                self.assertEqual(len(score_calls), 18)

            restore_path = root / 'restore.json'
            with patch.object(cli, 'base_identity', return_value={'revision': 'fixture'}), \
                 patch.object(cli.models, 'check_environment', return_value={'fixture': True}), \
                 patch.object(cli, 'read_overlay', return_value={'synthetic': True}), \
                 patch.object(cli.models, 'load_full', return_value=FixtureModel()), \
                 patch.object(cli, 'restore_overlay', return_value={'status': 'passed', 'all_bitwise_equal': True,
                                                                     'keys': ['synthetic']}):
                argv = [str(script), 'restore-check', '--config', str(config), '--base-model', 'synthetic-base',
                        '--overlay', 'synthetic-overlay', '--output', str(restore_path)]
                with patch.object(sys, 'argv', argv):
                    cli.main()
            self.assertTrue(read(restore_path)['all_bitwise_equal'])

            audit_path = root / 'aggregate.json'
            with patch.object(cli, 'load_bundle', return_value=synthetic):
                argv = [str(script), 'audit-s', '--data', 'synthetic-manifest',
                        '--base-scores', str(base_scores / 'base.jsonl'),
                        '--cleared-scores', str(cleared_scores / 'cleared.jsonl'),
                        '--base-contract', str(base_scores / 'contract.json'),
                        '--cleared-contract', str(cleared_scores / 'contract.json'),
                        '--output', str(audit_path)]
                with patch.object(sys, 'argv', argv):
                    cli.main()
            result = read(audit_path)
            self.assertEqual(result['coverage']['expected_per_state'], 9)
            self.assertTrue(result['coverage']['paired_identity_match'])
            self.assertFalse(torch.cuda.is_initialized())


def read_json_roundtrip(value):
    return loads(canonical(value))


if __name__=='__main__':unittest.main()
