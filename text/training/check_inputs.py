"""Check local text-training inputs without loading model weights."""
import argparse
import json
import os
from pathlib import Path

MODELS = {"qwen":"qwen3vl8b","llava":"llava_next8b","internvl":"internvl3_8b"}
FILES = ["existing_bio_forget_train_seed42.jsonl","cyber_forget_train_seed42.jsonl",
         "chem_forget_train_seed42.jsonl","existing_wikitext_retain_train_seed42_v2.jsonl"]
BASES = {
 "qwen":[("TRAIN_QWEN_GROM_BASIS","qwen_grom_basis.pt"),("TRAIN_QWEN_CYBER_GROM_BASIS","qwen_grom_basis_cyber.pt")],
 "llava":[("TRAIN_LLAVA_GROM_BASIS","llava_grom_basis.pt")],
 "internvl":[("TRAIN_INTERNVL_GROM_BASIS","internvl_grom_basis_multi.pt")]}

def check(family, root):
    root=Path(root)
    model=root/"text/models"/(family+"-base")
    data=root/"data/processed"/MODELS[family]/"three_domain_rmu_20261024"
    required=[model/"config.json"]+[data/n for n in FILES]
    for variable,name in BASES[family]:
        p=Path(os.environ.get(variable,"text/runs/bases/"+name))
        required.append(p if p.is_absolute() else root/p)
    missing=[str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError("缺少训练输入（C/S评测数据不能代替训练数据）：\n"+"\n".join(missing))
    for name in FILES:
        rows=0
        with (data/name).open(encoding="utf-8") as stream:
            for line in stream:
                row=json.loads(line);ids=row.get("token_ids")
                if not isinstance(ids,list) or len(ids)<2 or any(type(x) is not int or x<0 for x in ids):
                    raise ValueError("训练token_ids格式错误："+name)
                rows+=1
        if not rows:raise ValueError("训练文件为空："+name)
    print(f"{family}: 训练输入路径和token_ids格式检查完成。",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("model",choices=list(MODELS))
    args=parser.parse_args()
    check(args.model,Path(os.environ.get("TRAIN_ROOT",Path(__file__).resolve().parents[2])))
