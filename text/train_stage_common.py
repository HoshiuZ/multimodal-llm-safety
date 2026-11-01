"""Recorded LLaVA/InternVL NPO stage; not a complete released-model recipe."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def main(family):
    parser=argparse.ArgumentParser(description=f'{family} historical NPO stage')
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    settings=json.loads((ROOT/'train_stages.json').read_text());info=settings['models'][family]
    project=(ROOT/settings['training_project_root']).resolve();out=(ROOT/info['output_dir']).resolve()
    arguments=[x.replace('{input}',info['input_file']) for x in info['argv']]
    cmd=[sys.executable,'-u','-B',str(ROOT/'training_sources/dsh_npo_generic.py'),*arguments]
    if args.dry_run:
        print(json.dumps({'family':family,'project_root':str(project),'output_dir':str(out),'command':cmd,
            'note':'Recorded NPO stage only; not a complete released-model reproduction. No training performed.'},indent=2));return
    if out.exists():raise ValueError('输出目录已存在，拒绝覆盖；请修改 train_stages.json。')
    train=project/info['data_dir']
    needed=[project/info['model_dir']/'config.json',project/info['input_file'],project/info['mmlu_config'],
            project/'data/eval/bio_rmu_effect_20261023/lists/mmlu_dev285.jsonl',project/'data/eval/raw/hails_mmlu_no_train_b2e1ec9']
    needed += [train/name for name in ('existing_bio_forget_train_seed42.jsonl','cyber_forget_train_seed42.jsonl','chem_forget_train_seed42.jsonl','existing_wikitext_retain_train_seed42_v2.jsonl')]
    missing=[str(x) for x in needed if not x.exists()]
    if missing:raise FileNotFoundError('训练输入缺失：'+', '.join(missing))
    if sha(project/info['input_file'])!=info['input_sha256']:raise ValueError('中间参数文件身份不一致')
    source=ROOT/'training_sources/dsh_npo_generic.py';record=json.loads((ROOT/'training_sources/source_identity.json').read_text())
    if sha(source)!=record[source.name]['adapted_sha256']:raise ValueError('训练脚本身份不一致')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('须使用调度器分配的单张 GPU')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,TRAIN_PROJECT_ROOT=str(project),TRAIN_OUTPUT_DIR=str(out/'checkpoints'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    print(f'{family} 历史 NPO 阶段；输出为参数更新，不是完整发布模型。',flush=True)
    if family=='internvl':print('注意：历史配置与最终参数来源存在未确认差异，不应视为完整发布模型复现。',flush=True)
    print('执行:', ' '.join(cmd),flush=True);subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    print('训练阶段完成；参数更新位于',out/'checkpoints',flush=True)
