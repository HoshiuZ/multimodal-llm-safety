#!/usr/bin/env python3
"""Portable six-shard inference/review entry point for the frozen M1 + M3 v8 method."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.metadata, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import evaluate_baseline as EB
from xmodal import guard_v8 as GUARD

MODEL_CONFIG = ROOT / "configs" / "model_registry.json"
PROTOCOL_CONFIG = ROOT / "configs" / "eval_protocol.json"

def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def tree_fingerprint(root: Path) -> str:
    entries=[]
    for current,dirs,files in os.walk(root,followlinks=True):
        dirs[:]=sorted(d for d in dirs if d not in {'.cache','.git'})
        for name in sorted(files):
            p=Path(current)/name
            if not p.is_file(): continue
            rel=p.relative_to(root).as_posix()
            entries.append((rel,p.stat().st_size,sha_file(p)))
    if not entries: raise ValueError(f'no checkpoint files under {root}')
    h=hashlib.sha256()
    for rel,size,digest in sorted(entries):
        h.update(rel.encode());h.update(b'\0');h.update(str(size).encode());h.update(b'\0');h.update(digest.encode());h.update(b'\n')
    return h.hexdigest()

def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower()=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    else:
        rows=[]
        with path.open(encoding='utf-8') as f:
            for i,line in enumerate(f,1):
                if line.strip():
                    try: rows.append(json.loads(line))
                    except json.JSONDecodeError as e: raise ValueError(f'invalid JSONL {path}:{i}: {e}') from e
    ids=[str(r.get('sample_id') or r.get('eval_id') or '') for r in rows]
    if not rows or any(not x for x in ids) or len(ids)!=len(set(ids)): raise ValueError('manifest must have unique nonempty sample_id values')
    for r in rows:
        if not isinstance(r.get('query'),str) or not r['query'].strip(): raise ValueError(f"empty query: {r.get('sample_id')}")
        if not r.get('scenario'): r['scenario']=str(r.get('benchmark') or 'unknown')
        if not r.get('cell'): r['cell']=str(r.get('stratum') or r['scenario'])
        if not r.get('image_path') or len(str(r.get('img_sha256','')))!=64: raise ValueError(f"missing image path/hash: {r.get('sample_id')}")
    return rows

def write_json(path: Path,value:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8');tmp.replace(path)

def read_json(path: Path): return json.loads(path.read_text(encoding='utf-8'))

def write_row(path:Path,row:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')

def load_existing(path:Path,model_key:str,state:str,run_id:str):
    found={}
    if not path.exists(): return found
    with path.open(encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            try:r=json.loads(line)
            except json.JSONDecodeError as e:raise ValueError(f'torn/invalid output {path}:{n}; repair or use a new output directory') from e
            sid=r.get('sample_id')
            if not sid or sid in found or r.get('model_key')!=model_key or r.get('state')!=state or r.get('run_id')!=run_id:
                raise ValueError(f'foreign or duplicate output row at {path}:{n}')
            if r.get('response_sha256')!=hashlib.sha256(str(r.get('response','')).encode()).hexdigest(): raise ValueError(f'response hash mismatch at {path}:{n}')
            found[sid]=r
    return found

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate',help='Generate original or M1 answers. M1 adapters are loaded without training.')
    g.add_argument('--model-key',required=True,choices=['qwen3vl8b','llavanext8b','internvl3_8b']);g.add_argument('--state',required=True,choices=['orig','m1'])
    g.add_argument('--manifest',required=True,type=Path);g.add_argument('--data-root',required=True,type=Path);g.add_argument('--base-model',required=True,type=Path)
    g.add_argument('--adapter',type=Path);g.add_argument('--output-dir',required=True,type=Path);g.add_argument('--num-shards',type=int,default=int(os.environ.get('WORLD_SIZE','1')))
    g.add_argument('--rank',type=int,default=int(os.environ.get('LOCAL_RANK','0')));g.add_argument('--device-index',type=int,default=int(os.environ.get('LOCAL_RANK','0')))
    g.add_argument('--resume',action='store_true');g.add_argument('--config',type=Path,default=PROTOCOL_CONFIG)
    r=sub.add_parser('review',help='Apply the common frozen Qwen3-VL M3 v8 visual reviewer to M1 answers.')
    r.add_argument('--model-key',required=True,choices=['qwen3vl8b','llavanext8b','internvl3_8b']);r.add_argument('--manifest',required=True,type=Path);r.add_argument('--data-root',required=True,type=Path)
    r.add_argument('--m1-input-dir',required=True,type=Path);r.add_argument('--reviewer-model',required=True,type=Path);r.add_argument('--output-dir',required=True,type=Path)
    r.add_argument('--num-shards',type=int,default=int(os.environ.get('WORLD_SIZE','1')));r.add_argument('--rank',type=int,default=int(os.environ.get('LOCAL_RANK','0')))
    r.add_argument('--device-index',type=int,default=int(os.environ.get('LOCAL_RANK','0')));r.add_argument('--resume',action='store_true');r.add_argument('--config',type=Path,default=PROTOCOL_CONFIG)
    a=p.parse_args();cfg=read_json(a.config);models=read_json(MODEL_CONFIG)['models'];spec=models[a.model_key]
    if a.num_shards<1 or not 0<=a.rank<a.num_shards: raise ValueError('invalid shard settings')
    rows=read_rows(a.manifest);selected=rows[a.rank::a.num_shards]
    if not selected: raise ValueError('empty shard')
    data_root=a.data_root.expanduser().resolve();outdir=a.output_dir.expanduser().resolve();outdir.mkdir(parents=True,exist_ok=True)
    state=a.state if a.command=='generate' else 'm1_m3_v8';dataset=a.manifest.stem
    shard=outdir/f'{dataset}.{state}.shard-{a.rank:02d}-of-{a.num_shards:02d}.jsonl';marker=Path(str(shard)+'.complete.json')
    import torch, transformers
    required=spec['runtime']['transformers'] if a.command=='generate' else '5.16.1'
    if transformers.__version__!=required: raise RuntimeError(f"{a.model_key} requires transformers=={required}; found {transformers.__version__}")
    if not torch.cuda.is_available() or a.device_index>=torch.cuda.device_count(): raise RuntimeError('requested CUDA device is unavailable')
    torch.cuda.set_device(a.device_index)
    if a.command=='generate':
        if (a.state=='m1') != (a.adapter is not None): raise ValueError('orig requires no --adapter; m1 requires --adapter')
        base=a.base_model.expanduser().resolve();base_sha=tree_fingerprint(base)
        if base_sha!=spec['base_tree_sha256']: raise ValueError(f"base model tree SHA differs: {base_sha}")
        adapter_sha=adapter_cfg_sha=None
        if a.adapter:
            adapter=a.adapter.expanduser().resolve();w=adapter/'adapter_model.safetensors';c=adapter/'adapter_config.json'
            if not w.is_file() or not c.is_file(): raise FileNotFoundError('adapter requires adapter_model.safetensors and adapter_config.json')
            adapter_sha=sha_file(w);adapter_cfg_sha=sha_file(c)
            if adapter_sha!=spec['adapter_weight_sha256'] or adapter_cfg_sha!=spec['adapter_config_sha256']:
                raise ValueError('published adapter hashes differ from the frozen registry')
        context=dict(command='generate',model_key=a.model_key,state=a.state,manifest_sha256=sha_file(a.manifest),base_tree_sha256=base_sha,
                     base_revision=spec['base_revision'],adapter_weight_sha256=adapter_sha,adapter_config_sha256=adapter_cfg_sha,
                     decode=cfg['generation'],internvl_preprocess=cfg['internvl_preprocess'])
        run_id=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
        if marker.exists():
            old=read_json(marker)
            if old.get('run_id')==run_id and old.get('output_sha256')==sha_file(shard): print('already complete',shard);return
            if not a.resume: raise FileExistsError(f'output exists with different identity: {shard}')
        meta=Path(str(shard)+'.meta.json')
        meta_value=dict(context,run_id=run_id)
        if meta.exists() and read_json(meta)!=meta_value: raise ValueError('resume metadata differs')
        if not meta.exists(): write_json(meta,meta_value)
        prior=load_existing(shard,a.model_key,state,run_id)
        import evaluate_baseline as baseline_eval
        baseline_cfg=read_json(ROOT/'configs'/'baseline_runtime.json')
        bundle=baseline_eval.load_adapter(a.model_key,str(base),baseline_cfg,torch)
        if a.state=='m1':
            from peft import PeftModel
            if a.model_key=='internvl3_8b': bundle['model'].language_model=PeftModel.from_pretrained(bundle['model'].language_model,str(adapter),local_files_only=True,is_trainable=False).eval()
            else: bundle['model']=PeftModel.from_pretrained(bundle['model'],str(adapter),local_files_only=True,is_trainable=False).eval()
            bundle['model'].eval()
        from PIL import Image
        for i,source in enumerate(selected,1):
            sid=str(source.get('sample_id') or source.get('eval_id'))
            if sid in prior: continue
            image_path=(data_root/str(source['image_path'])).resolve()
            if not image_path.is_file(): raise FileNotFoundError(image_path)
            if sha_file(image_path)!=source['img_sha256']: raise ValueError(f'image hash mismatch: {sid}')
            prompt=source.get('generation_query') or source['query']
            with Image.open(image_path) as im: answer,patches=baseline_eval.generate_one(a.model_key,bundle,im.convert('RGB'),prompt,baseline_cfg,torch)
            if not isinstance(answer,str): raise TypeError('model generation must return text')
            row=dict(source,sample_id=sid,model_key=a.model_key,state=a.state,run_id=run_id,generation_query=prompt,response=answer,
                     response_sha256=hashlib.sha256(answer.encode()).hexdigest(),internvl_patch_count=patches)
            write_row(shard,row);prior[sid]=row
            if i%20==0 or i==len(selected): print(f'generate rank={a.rank} {i}/{len(selected)}',flush=True)
        if set(prior)!={str(x.get('sample_id') or x.get('eval_id')) for x in selected}: raise ValueError('generation shard coverage incomplete')
        write_json(marker,dict(generation_complete=True,model_key=a.model_key,state=a.state,rows=len(selected),run_id=run_id,rank=a.rank,output_sha256=sha_file(shard)))
        return
    # review phase
    if a.model_key!='internvl3_8b' and transformers.__version__!='5.16.1': raise RuntimeError('M3 reviewer requires transformers==5.16.1')
    m1=a.m1_input_dir.expanduser().resolve()/f'{dataset}.m1.shard-{a.rank:02d}-of-{a.num_shards:02d}.jsonl'
    m1rows=load_existing(m1,a.model_key,'m1',read_json(Path(str(m1)+'.meta.json'))['run_id'])
    expected={str(x.get('sample_id') or x.get('eval_id')) for x in selected}
    if set(m1rows)!=expected: raise ValueError('M1 shard does not cover the selected input rows')
    reviewer=a.reviewer_model.expanduser().resolve();reviewer_sha=tree_fingerprint(reviewer)
    m3=read_json(MODEL_CONFIG)['reviewer']
    if reviewer_sha!=m3['tree_sha256']: raise ValueError(f'M3 model tree SHA differs: {reviewer_sha}')
    context=dict(command='review',model_key=a.model_key,state=state,manifest_sha256=sha_file(a.manifest),m1_input_sha256=sha_file(m1),reviewer_tree_sha256=reviewer_sha,
                 m3_version=GUARD.VERSION,m3_settings=cfg['m3'],num_shards=a.num_shards)
    run_id=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    if marker.exists():
        old=read_json(marker)
        if old.get('run_id')==run_id and old.get('output_sha256')==sha_file(shard): print('already complete',shard);return
        if not a.resume: raise FileExistsError(f'output exists with different identity: {shard}')
    meta=Path(str(shard)+'.meta.json')
    meta_value=dict(context,run_id=run_id)
    if meta.exists() and read_json(meta)!=meta_value: raise ValueError('resume metadata differs')
    if not meta.exists(): write_json(meta,meta_value)
    prior=load_existing(shard,a.model_key,state,run_id)
    bundle=EB.load_adapter('qwen3vl8b',str(reviewer),read_json(ROOT/'configs'/'baseline_runtime.json'),torch)
    guard=GUARD.Reviewer(bundle,cfg['m3'],torch)
    controls=[]
    for case in GUARD.controls():
        result=guard.detect(case['query'],case['draft'])
        ok=result.get('status')=='resolved' and result['decision']['needs_revision']==case['needs_revision']
        controls.append(dict(control_id=case['control_id'],passed=ok,result=result))
    write_json(outdir/f'{dataset}.m3-controls.rank-{a.rank:02d}.json',dict(run_id=run_id,passed=all(x['passed'] for x in controls),outcomes=controls))
    if not all(x['passed'] for x in controls): raise RuntimeError('M3 control gate failed; no score should be reported')
    from PIL import Image
    for i,source in enumerate(selected,1):
        sid=str(source.get('sample_id') or source.get('eval_id'))
        if sid in prior: continue
        draft=m1rows[sid]
        if draft.get('query')!=source.get('query') or draft.get('img_sha256')!=source.get('img_sha256'): raise ValueError(f'M1/input pairing differs: {sid}')
        image_path=(data_root/str(source['image_path'])).resolve()
        if not image_path.is_file() or sha_file(image_path)!=source['img_sha256']: raise ValueError(f'image missing/hash mismatch: {sid}')
        prompt=source.get('generation_query') or source['query']
        with Image.open(image_path) as im: protected=guard.protect(prompt,draft['response'],'visual',im.convert('RGB'))
        if protected.get('guard_status')!='resolved' or not isinstance(protected.get('response'),str): raise RuntimeError(f'M3 unresolved for {sid}; repair before aggregation')
        row=dict(source,sample_id=sid,model_key=a.model_key,state='m1_m3_v8',run_id=run_id,generation_query=prompt,
                 response=protected['response'],response_sha256=protected['response_sha256'],m1_response_sha256=draft['response_sha256'],
                 guard_status=protected['guard_status'],guardian_version=GUARD.VERSION,review_trace=protected)
        write_row(shard,row);prior[sid]=row
        if i%20==0 or i==len(selected): print(f'review rank={a.rank} {i}/{len(selected)}',flush=True)
    if set(prior)!=expected: raise ValueError('review shard coverage incomplete')
    write_json(marker,dict(generation_complete=True,model_key=a.model_key,state=state,rows=len(selected),run_id=run_id,rank=a.rank,output_sha256=sha_file(shard)))

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise
