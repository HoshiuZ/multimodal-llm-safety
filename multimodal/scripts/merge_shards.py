#!/usr/bin/env python3
"""Merge six completed inference/review shards in frozen manifest order."""
import argparse,hashlib,json
from pathlib import Path
from mm_eval import read_rows,read_json,write_json,sha_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--model-key',required=True);p.add_argument('--state',required=True,choices=['orig','m1','m1_m3_v8']);p.add_argument('--num-shards',type=int,default=6);a=p.parse_args()
 rows=read_rows(a.manifest);ids=[str(x.get('sample_id') or x.get('eval_id')) for x in rows];name=a.manifest.stem;outdir=a.output_dir.resolve()
 records={};run_ids=[]
 for rank in range(a.num_shards):
  path=outdir/f'{name}.{a.state}.shard-{rank:02d}-of-{a.num_shards:02d}.jsonl';marker=Path(str(path)+'.complete.json')
  if not path.is_file() or not marker.is_file():raise FileNotFoundError(f'incomplete shard: {path}')
  meta=read_json(marker);chunk=read_rows(path);expected=ids[rank::a.num_shards]
  if meta.get('model_key')!=a.model_key or meta.get('state')!=a.state or meta.get('rows')!=len(expected) or meta.get('output_sha256')!=sha_file(path):raise ValueError(f'shard marker/hash mismatch: {path}')
  if [str(x.get('sample_id') or x.get('eval_id')) for x in chunk]!=expected:raise ValueError(f'shard IDs/order mismatch: {path}')
  run_ids.append(meta.get('run_id'))
  for row in chunk:
   sid=row['sample_id']
   if sid in records:raise ValueError(f'duplicate sample: {sid}')
   records[sid]=row
 if set(records)!=set(ids):raise ValueError('merged coverage differs from manifest')
 merged=[records[sid] for sid in ids];out=outdir/f'{name}.{a.state}.jsonl';out.parent.mkdir(parents=True,exist_ok=True)
 tmp=out.with_suffix(out.suffix+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:
  for row in merged:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
 tmp.replace(out)
 manifest_digest=hashlib.sha256('\n'.join(sorted(ids)).encode('utf-8')).hexdigest()
 write_json(Path(str(out)+'.complete.json'),dict(generation_complete=True,model_key=a.model_key,state=a.state,rows=len(merged),
  run_ids=run_ids,manifest_sha256=sha_file(a.manifest),manifest_digest=manifest_digest,output_sha256=sha_file(out)))
 print(out)
if __name__=='__main__':main()
