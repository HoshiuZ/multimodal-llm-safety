#!/usr/bin/env python3
"""Generic NPO stage (llava / qwen / internvl).

argv[1] model_key
argv[2] tag
argv[3] edits  "7:30,9:15,11:10" (applied before training) or "none"
argv[4] nodes  "8,12,16,18"
argv[5] ref_mode: "base" or a checkpoint path for the reference model
argv[6] basis path (or "none")

Loss: NPO(beta=0.1) on forget + 1.0 * CE_retain, lr=5e-6, 4+4 sequences/step, 512 tokens,
trainable = layers 4-7 down_proj (fp32 masters, bf16 autocast).
"""
import json, pathlib, sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForImageTextToText

import os
ROOT = pathlib.Path(os.environ["TRAIN_PROJECT_ROOT"]).resolve()
SPECS = {
    "llava": {"cls": "llava", "dir": "models/llava_next8b", "fmt": "model.language_model.layers.{i}.mlp.down_proj",
              "data": "data/processed/llava_next8b/three_domain_rmu_20261024",
              "mmlu_cfg": "configs/three_domain_rmu_mmlu285/llava_next8b.json"},
    "qwen": {"dir": "models/qwen3vl8b", "fmt": "model.language_model.layers.{i}.mlp.down_proj",
             "data": "data/processed/qwen3vl8b/three_domain_rmu_20261024",
             "mmlu_cfg": "configs/three_domain_rmu_mmlu285/qwen3vl8b.json"},
    "internvl": {"dir": "models/internvl3_8b", "fmt": "language_model.model.layers.{i}.mlp.down_proj",
                 "data": "data/processed/internvl3_8b/three_domain_rmu_20261024",
                 "mmlu_cfg": "configs/three_domain_rmu_mmlu285/internvl3_8b.json"},
}
FILES = {"bio": "existing_bio_forget_train_seed42.jsonl",
         "cyber": "cyber_forget_train_seed42.jsonl",
         "chem": "chem_forget_train_seed42.jsonl",
         "retain": "existing_wikitext_retain_train_seed42_v2.jsonl"}
MLIST = ROOT / "data/eval/bio_rmu_effect_20261023/lists/mmlu_dev285.jsonl"
MPARQ = ROOT / "data/eval/raw/hails_mmlu_no_train_b2e1ec9"
LETTERS = ["A", "B", "C", "D"]
BETA, LR_DEFAULT, MAXLEN, NL, LAYERS = 0.1, 5e-6, 512, 4, (4, 5, 6, 7)


def load_model(spec, ckpt=None, edits=None, basis=None):
    kw = dict(torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True)
    if spec.get("cls") == "llava":
        from transformers import LlavaNextForConditionalGeneration
        m = LlavaNextForConditionalGeneration.from_pretrained(str(ROOT / spec["dir"]), **kw)
    else:
        try:
            m = AutoModelForImageTextToText.from_pretrained(str(ROOT / spec["dir"]), **kw)
        except Exception:
            from transformers import AutoModel
            m = AutoModel.from_pretrained(str(ROOT / spec["dir"]), **kw)
    m = m.to("cuda")
    modules = dict(m.named_modules())
    if edits and edits != "none":
        _obj = torch.load(ROOT / basis, map_location="cpu", weights_only=True)
        _multi = not isinstance(_obj.get("1000"), torch.Tensor)
        for item in edits.split(","):
            li, c = item.split(":")
            b = _obj[li]["1000"] if _multi else _obj["1000"]
            mod = modules[spec["fmt"].format(i=int(li))]
            base = mod.weight.detach()
            d = (b.to(device=base.device, dtype=torch.float64) * float(c))
            with torch.no_grad():
                mod.weight.copy_((base.to(torch.float64) + d).to(base.dtype))
    if ckpt and ckpt != "base":
        sd = torch.load(ROOT / ckpt, map_location="cpu", weights_only=True)
        named = dict(m.named_parameters())
        with torch.no_grad():
            for n, t in sd.items():
                named[n].copy_(t.to(device=named[n].device, dtype=torch.float32))
    # causal LM module (has lm_head and .model.layers)
    lm = None
    for name, mod in m.named_modules():
        inner = getattr(mod, "model", None)
        if hasattr(mod, "lm_head") and inner is not None and hasattr(inner, "layers"):
            lm = mod
            print("lm module:", name or "<root>", flush=True)
            break
    if lm is None and hasattr(m, "lm_head"):
        lm = m
        print("lm module: <root> (has lm_head)", flush=True)
    if lm is None:
        raise SystemExit("no causal LM module found")
    return m, lm


def seq_logp(lm, ids):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    lg = out.logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    return F.log_softmax(lg, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()


def main():
    key, tag, edits, nodes_s, refmode, basis = sys.argv[1:7]
    LR = float(sys.argv[7]) if len(sys.argv) > 7 else LR_DEFAULT
    INIT = sys.argv[9] if len(sys.argv) > 9 else "none"
    REPAIR = int(sys.argv[10]) if len(sys.argv) > 10 else 0
    REPAIR_LR = float(sys.argv[11]) if len(sys.argv) > 11 else 5e-7
    MODE = sys.argv[12] if len(sys.argv) > 12 else "npo"
    RL = tuple(int(x) for x in sys.argv[13].split(",")) if len(sys.argv) > 13 and sys.argv[13] != "none" else None
    TRAIN_LAYERS = tuple(int(x) for x in sys.argv[14].split(",")) if len(sys.argv) > 14 and sys.argv[14] != "none" else LAYERS
    print(f"trainable layers for NPO = {TRAIN_LAYERS}", flush=True)
    print(f"repair layers = {RL if RL else LAYERS}", flush=True)
    print(f"loss mode = {MODE}", flush=True)
    RW = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    print(f"lr={LR} retain_weight={RW}", flush=True)
    nodes = [int(x) for x in nodes_s.split(",")]
    spec = SPECS[key]
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(str(ROOT / spec["dir"]), local_files_only=True,
                                        trust_remote_code=True)
    letter_ids = {L: tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS}
    rows_f = [json.loads(l) for f in ("bio", "cyber", "chem")
              for l in open(ROOT / spec["data"] / FILES[f])]
    rows_r = [json.loads(l) for l in open(ROOT / spec["data"] / FILES["retain"])]
    print("forget", len(rows_f), "retain", len(rows_r), flush=True)

    # --- reference logp
    rm, rlm = load_model(spec, ckpt=(None if refmode == "base" else refmode))
    ref = []
    with torch.no_grad():
        for r in rows_f:
            ids = torch.tensor([r["token_ids"][:MAXLEN]], dtype=torch.long, device="cuda")
            ref.append(float(seq_logp(rlm, ids)))
    print("ref logp mean", sum(ref) / len(ref), flush=True)
    del rm, rlm
    torch.cuda.empty_cache()

    # --- student
    model, lm = load_model(spec, edits=edits, basis=basis)
    if INIT != "none":
        _sd = torch.load(ROOT / INIT, map_location="cpu", weights_only=True)
        _named = dict(model.named_parameters())
        with torch.no_grad():
            for _n, _t in _sd.items():
                _named[_n].copy_(_t.to(device=_named[_n].device, dtype=torch.float32))
        print(f"student initialised from {INIT}", flush=True)
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(lm, "enable_input_require_grads"):
        lm.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad_(False)
    named = dict(model.named_parameters())
    params, names = [], []
    for li in LAYERS:
        name = spec["fmt"].format(i=li) + ".weight"
        master = torch.nn.Parameter(named[name].detach().float().clone())
        mod_name = name.rsplit(".", 1)[0]
        module = dict(model.named_modules())[mod_name]
        module.weight = master
        params.append(master)
        names.append(name)
    opt = torch.optim.Adam(params, lr=LR)
    lm.train()

    mcfg = ROOT / spec["mmlu_cfg"]
    pre = set(json.loads(mcfg.read_text()).get("predeclared_subject_group") or []) if mcfg.is_file() else set()
    import pyarrow.parquet as pq
    mq, cache = [], {}
    for line in MLIST.open():
        r = json.loads(line)
        s = r["subject"]
        if s not in cache:
            cache[s] = pq.read_table(MPARQ / s / "test-00000-of-00001.parquet").to_pylist()
        src = cache[s][int(r["source_row_index"])]
        opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, src["choices"]))
        mq.append((s, f"The following are multiple choice questions (with answers) about "
                      f"{s.replace('_',' ')}.\n\n{src['question'].strip()}\n{opts}\nAnswer: ", int(src["answer"])))

    @torch.no_grad()
    def mmlu():
        lm.eval()
        m = o = n = 0
        for s, ctx, ans in mq:
            ids = torch.tensor([tok.encode(ctx, add_special_tokens=False)], dtype=torch.long, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            lg = out.logits[0, -1, :].float()
            pick = max(LETTERS, key=lambda L: float(lg[letter_ids[L]]))
            ok = int(LETTERS.index(pick) == ans)
            m += ok
            if s not in pre:
                o += ok; n += 1
        lm.train()
        return m, o, n

    m0, o0, _ = mmlu()
    print(f"node 0: mmlu285={m0} other245={o0}", flush=True)
    out = pathlib.Path(os.environ["TRAIN_OUTPUT_DIR"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    traj = {0: {"mmlu285": m0, "other245": o0}}

    def retain_ce(step):
        ce = torch.zeros((), device="cuda")
        for j in range(NL):
            r = rows_r[(NL * (step - 1) + j) % len(rows_r)]
            ids = torch.tensor([r["token_ids"][:MAXLEN]], dtype=torch.long, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            lg = o.logits[:, :-1, :].float()
            tgt = ids[:, 1:]
            ce = ce + F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="mean") / NL
        return ce

    if REPAIR > 0:
        if RL:
            named2 = dict(model.named_parameters())
            params, names = [], []
            for li in RL:
                nm = spec["fmt"].format(i=li) + ".weight"
                master = torch.nn.Parameter(named2[nm].detach().float().clone())
                modu = dict(model.named_modules())[nm.rsplit(".", 1)[0]]
                modu.weight = master
                params.append(master)
                names.append(nm)
            print("repair trains layers", RL, flush=True)
        opt_r = torch.optim.Adam(params, lr=REPAIR_LR)
        for rs in range(1, REPAIR + 1):
            opt_r.zero_grad(set_to_none=True)
            l_r = retain_ce(rs)
            l_r.backward()
            opt_r.step()
            del l_r
            torch.cuda.empty_cache()
        mr, orr, _ = mmlu()
        print(f"after repair ({REPAIR} steps): mmlu285={mr} other245={orr}", flush=True)
        traj[-1] = {"mmlu285": mr, "other245": orr}
    for step in range(1, max(nodes) + 1):
        q = (step - 1) // 3
        opt.zero_grad(set_to_none=True)
        npo = torch.zeros((), device="cuda")
        for j in range(NL):
            idx = (q * NL + j) % len(rows_f)
            ids = torch.tensor([rows_f[idx]["token_ids"][:MAXLEN]], dtype=torch.long, device="cuda")
            lp = seq_logp(lm, ids)
            if MODE == "ga":
                npo = npo - (2.0 / BETA) * F.logsigmoid(-BETA * (lp - ref[idx])) / NL if False else -(lp) / NL
            else:
                npo = npo - (2.0 / BETA) * F.logsigmoid(-BETA * (lp - ref[idx])) / NL
        loss = npo + RW * retain_ce(step)
        loss.backward()
        opt.step()
        del loss, npo
        torch.cuda.empty_cache()
        if step in nodes:
            m, o, _ = mmlu()
            traj[step] = {"mmlu285": m, "other245": o}
            print(f"node {step}: mmlu285={m} other245={o}", flush=True)
            torch.save({n: p.detach().to(torch.bfloat16).cpu() for n, p in zip(names, params)},
                       out / f"step{step}_updated_parameters.pt")
    (out / "trajectory.json").write_text(json.dumps(traj, indent=2))
    print("json " + json.dumps(traj, sort_keys=True))
    print("saved to", out, flush=True)


if __name__ == "__main__":
    main()
