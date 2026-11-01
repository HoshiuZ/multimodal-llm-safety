"""Restore a selected sparse update into its full multimodal base checkpoint."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def main(family):
    parser=argparse.ArgumentParser(description=f'restore selected {family} checkpoint')
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    cfg=json.loads((ROOT/'configs'/f'{family}.json').read_text())
    paths=json.loads((ROOT/'restore_paths.json').read_text())['models'][family]
    score_paths=json.loads((ROOT/'score_paths.json').read_text())['models'][family]
    if paths['identity_mode']!='selected':raise ValueError('仅支持恢复项目冻结的选定参数')
    if cfg['repository']!=score_paths['base_repo'] or cfg['base_revision']!=score_paths['base_revision']:
        raise ValueError('基模身份配置不一致')
    plan={'family':family,'base':paths['base_dir'],'overlay':paths['overlay'],
          'output':paths['output_dir'],'overlay_sha256':cfg['update']['sha256'],
          'steps':['verify base and selected overlay','restore complete wrapper','save full model',
                   'reload and compare every state tensor','verify released weight-file identity']}
    if args.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    base=ROOT/paths['base_dir'];overlay=ROOT/paths['overlay'];output=ROOT/paths['output_dir']
    if output.exists() or output.is_symlink():raise ValueError('输出目录已存在，拒绝覆盖')
    if not base.is_dir() or not overlay.is_file():raise FileNotFoundError('缺少完整基模或冻结清除参数')
    for name,expect in cfg.get('base_small_files',{}).items():
        p=base/name
        if not p.is_file() or p.stat().st_size!=expect['bytes'] or sha(p)!=expect['sha256']:
            raise ValueError('基模文件身份不一致：'+name)
    from eval import models
    from eval.overlay import read_overlay,restore_overlay
    models.check_environment(cfg)
    selected=read_overlay(overlay,cfg)
    print('已核对冻结清除参数；加载完整基模。',flush=True)
    model=models.load_full(base,cfg,'cpu')
    proof=restore_overlay(model,selected,cfg);del selected
    model.config._name_or_path=cfg['repository']
    output.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(str(output),safe_serialization=True,max_shard_size='5GB')
    generation=getattr(model,'generation_config',None)
    if generation is not None:generation.save_pretrained(str(output))
    # Retain the original tokenizer, processor metadata and custom model code.
    for p in base.iterdir():
        if not p.is_file() or p.suffix not in ('.json','.txt','.model','.tiktoken','.py','.jinja'):continue
        target=output/p.name
        if not target.exists():shutil.copy2(p,target)
    licenses=ROOT/'restoration_licenses'
    if family=='qwen':shutil.copyfile(licenses/'LICENSE-APACHE-2.0.txt',output/'LICENSE')
    elif family=='llava':
        shutil.copyfile(licenses/'LICENSE-LLAMA3',output/'LICENSE')
        shutil.copyfile(licenses/'USE_POLICY-LLAMA3.md',output/'USE_POLICY.md')
    else:
        (output/'LICENSES.md').write_text('# Third-party licenses\n\n## InternVL MIT\n\n'+
            (licenses/'LICENSE-INTERNVL-MIT').read_text()+'\n\n## Qwen Apache-2.0\n\n'+
            (licenses/'LICENSE-APACHE-2.0.txt').read_text(),encoding='utf-8')
    print('完整模型已保存，重新加载并逐张量核验。',flush=True)
    reloaded=models.load_full(output,cfg,'cpu');left=model.state_dict();right=reloaded.state_dict()
    if set(left)!=set(right):raise ValueError('恢复模型的参数键不一致')
    import torch
    for name,value in left.items():
        other=right[name]
        if value.shape!=other.shape or value.dtype!=other.dtype or not torch.equal(value,other):
            raise ValueError('恢复后张量不一致：'+name)
    expected=json.loads((ROOT/'score_model_identity.json').read_text())[family]['files']
    actual={}
    for name,ref in expected.items():
        p=output/name
        if not p.is_file():raise ValueError('恢复模型缺少文件：'+name)
        actual[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
        if actual[name]!=ref:raise ValueError('恢复权重文件与已发布版本身份不一致：'+name)
    record={'family':family,'base_repository':cfg['repository'],'base_revision':cfg['base_revision'],
            'selected_overlay_sha256':cfg['update']['sha256'],'overlay_restore':proof,
            'full_state_bitwise_equal':True,'state_tensor_count':len(left),
            'released_weight_files_match':True,'files':actual}
    (output/'restoration_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('恢复完成：完整模型及已发布权重身份核验通过。',flush=True)
    print('模型目录:',output,flush=True)
