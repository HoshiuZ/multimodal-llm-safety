#!/usr/bin/env python3
"""Verify a downloaded evaluation manifest and every referenced image SHA-256."""
import argparse,csv,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--data-root',required=True,type=Path);p.add_argument('--expected-rows',type=int,default=1008);a=p.parse_args()
 if a.manifest.suffix.lower()=='.csv':
  with a.manifest.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 else:rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]
 ids=[str(x.get('sample_id') or x.get('eval_id') or '') for x in rows]
 if len(rows)!=a.expected_rows or len(set(ids))!=len(ids) or any(not x for x in ids):raise ValueError('row count or sample IDs differ')
 root=a.data_root.resolve()
 for row in rows:
  path=(root/row['image_path']).resolve()
  if root not in path.parents:raise ValueError('image path escapes the data root')
  h=hashlib.sha256(path.read_bytes()).hexdigest()
  if h!=row['img_sha256']:raise ValueError(f'image SHA-256 mismatch: {row["sample_id"]}')
 print(f'PASS: {len(rows)} rows and all image hashes match')
if __name__=='__main__':main()
