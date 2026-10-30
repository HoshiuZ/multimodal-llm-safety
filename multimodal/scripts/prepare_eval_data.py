#!/usr/bin/env python3
"""Build the fixed 1,008-row base-view manifest from the official MM-SafetyBench manifest."""
import argparse,csv,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--selection',type=Path,default=Path('data/sample_1008_seed42.txt'));p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 ids=[x.strip() for x in a.selection.read_text().splitlines() if x.strip()]
 if len(ids)!=1008 or len(set(ids))!=1008:raise ValueError('selection must contain exactly 1,008 unique sample IDs')
 with a.manifest.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 by={x.get('sample_id'):x for x in rows}
 if len(by)!=len(rows) or set(ids)-set(by):raise ValueError('official manifest does not cover the frozen selection')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',encoding='utf-8') as f:
  for sid in ids:f.write(json.dumps(by[sid],ensure_ascii=False)+'\n')
 print(f'wrote {len(ids)} rows to {a.output}')
if __name__=='__main__':main()
