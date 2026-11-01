"""Standalone C baseline scoring before the cleared model is available."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import score_common

ROOT = Path(__file__).resolve().parent


def plan(family):
    settings, info, _ = score_common.load_settings(family, 'C')
    out = info['baseline_output_dir']
    pre = info['baseline_preflight']
    base = info['base_dir']
    common = ['--config', info['config'], '--data', settings['datasets']['C'],
              '--base-model', base, '--cleared-model', base, '--preflight', pre]
    exe = [sys.executable, '-u', '-B', 'scripts/text_eval.py']
    return settings, info, out, pre, [exe + ['preflight'] + common,
        exe + ['run'] + common + ['--state', 'base', '--device', 'cuda:0', '--output', out]]


def main(family):
    parser = argparse.ArgumentParser(description=f'{family}: C baseline only')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--load-only', action='store_true', help='load the full base model without scoring')
    args = parser.parse_args()
    settings, info, output, pre, cmds = plan(family)
    if args.dry_run:
        print(json.dumps({'family':family,'dataset':'C','base_repo':info['base_repo'],
            'base_revision':info['base_revision'],'output':output,'load_only':args.load_only,'steps':([] if args.load_only else cmds),
            'note':'No model, GPU or network used.'},ensure_ascii=False,indent=2)); return
    base=ROOT/info['base_dir']; data=ROOT/settings['datasets']['C'];out=ROOT/output
    if args.load_only:
        if not base.is_dir():raise SystemExit(f'缺少完整基模: {base}')
        import torch
        from eval import models
        from eval.common import read
        cfg=read(ROOT/info['config']);models.check_environment(cfg)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError('须使用调度器分配的单张 GPU，逻辑设备 cuda:0')
        tok=models.tokenizer(base,family);model=models.load_full(base,cfg,'cuda:0')
        print(f'{family} 完整基准模型及分词器加载成功；未进行评分。',flush=True)
        del model,tok
        return
    if not base.is_dir() or not data.is_file():
        raise SystemExit(f'缺少基模或 C 清单；基模可按仓库说明从 {info["base_repo"]} 获取。')
    complete=out/'baseline-summary.json'
    if complete.exists():raise ValueError('基模评分已完成；重新测试请使用新的输出目录')
    if out.exists() or (ROOT/pre).exists():
        raise ValueError('基模输出或预检文件已存在，拒绝覆盖；请先指定新的输出目录')
    from eval.data import load_bundle
    bundle=load_bundle(data)
    if bundle['kind']!='C':raise ValueError('expected C dataset')
    cfg=json.loads((ROOT/info['config']).read_text(encoding='utf-8'))
    for name,expected in cfg.get('base_small_files',{}).items():
        path=base/name
        if not path.is_file() or path.stat().st_size!=expected['bytes'] or score_common.file_hash(path)!=expected['sha256']:
            raise ValueError(f'基模文件不匹配: {name}')
    print('基模独立预检与评分；两处模型参数均指向基模，仅运行 base 状态。',flush=True)
    for cmd in cmds:
        print('执行:', ' '.join(cmd),flush=True);subprocess.run(cmd,cwd=ROOT,check=True)
    from eval.common import read,digest
    from eval.worker import records
    identity=read(out/'contract.json');rows=records(out/'base.jsonl',bundle,digest(identity),'base',complete=True)
    n=len(rows);correct=sum(r['correct'] for r in rows.values())
    result={'dataset':'C','state':'base','n':n,'correct':correct,'accuracy':correct/n,'contract_sha256':digest(identity)}
    complete.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result,flush=True)

if __name__ == '__main__':
    raise SystemExit('Use baseline_qwen.py, baseline_llava.py or baseline_internvl.py')
