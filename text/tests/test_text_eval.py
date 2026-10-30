"""Benign synthetic CPU fixtures; never research/model evaluation results."""
import copy
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


def read_json_roundtrip(value):
    return loads(canonical(value))


if __name__=='__main__':unittest.main()
