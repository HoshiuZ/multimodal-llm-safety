"""Optional historical Qwen GROM+NPO training example; no work during --dry-run."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    c=json.loads((ROOT/'train_qwen_config.json').read_text());project=(ROOT/c['training_project_root']).resolve();out=(ROOT/c['output_dir']).resolve();init=out/'initial_c50.pt'
    steps=[[sys.executable,'-u','-B',str(ROOT/'training_sources/qwen_materialize.py'),c['starting_overlay'],c['edit'],c['cyber_basis'],str(init)],
           [sys.executable,'-u','-B',str(ROOT/'training_sources/qwen_npo5.py'),c['tag'],'none',c['steps'],'base',c['historical_basis_arg'],c['learning_rate'],c['retain_weight'],c['max_length'],c['layers'],'seq',c['beta'],'1',str(init)]]
    if args.dry_run:
        print(json.dumps({'project_root':str(project),'output_dir':str(out),'steps':steps,'note':'No training or filesystem writes.'},indent=2));return
    if out.exists():raise ValueError('输出目录已存在，拒绝覆盖；请修改 train_qwen_config.json。')
    needed=[project/'models/qwen3vl8b/config.json',project/c['starting_overlay'],project/c['cyber_basis'],project/c['historical_basis_arg'],project/'configs/three_domain_rmu_mmlu285/qwen3vl8b.json',project/'data/eval/bio_rmu_effect_20261023/lists/mmlu_dev285.jsonl',project/'data/eval/raw/hails_mmlu_no_train_b2e1ec9']
    train=project/'data/processed/qwen3vl8b/three_domain_rmu_20261024'
    needed += [train/name for name in ('existing_bio_forget_train_seed42.jsonl','cyber_forget_train_seed42.jsonl','chem_forget_train_seed42.jsonl','existing_wikitext_retain_train_seed42_v2.jsonl')]
    missing=[str(x) for x in needed if not x.exists()]
    if missing:raise FileNotFoundError('训练输入缺失：'+', '.join(missing))
    for k in ('starting_overlay','cyber_basis'):
        if sha(project/c[k])!=c[k+'_sha256']:raise ValueError(f'{k} 与记录身份不一致')
    source=json.loads((ROOT/'training_sources/source_identity.json').read_text())
    for name,meta in source.items():
        if sha(ROOT/'training_sources'/name)!=meta['adapted_sha256']:raise ValueError('训练脚本身份不一致：'+name)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('须使用调度器分配的单张 GPU')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,TRAIN_PROJECT_ROOT=str(project),TRAIN_OUTPUT_DIR=str(out/'checkpoints'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    print('Qwen 清除训练示例：GROM 参数构建和 NPO 训练；输出不会覆盖原实验。',flush=True)
    for cmd in steps:
        print('执行:', ' '.join(cmd),flush=True);subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    print('训练阶段完成；结果在',out/'checkpoints','中。它是参数更新文件，正式评测前仍需恢复为完整模型并核验。',flush=True)
if __name__=='__main__':main()
