#!/usr/bin/env python3
"""Qwen3-VL-8B NPO stage (same math as scripts/dsh_npo_generic.py).

Only structural difference from the generic script: Qwen3VLForConditionalGeneration
holds lm_head at the top level and its text tower has no lm_head, so the generic
"module with lm_head whose .model has .layers" search cannot fire.  We therefore use
the root model for the forward pass (text-only input is supported: get_rope_index
falls back to plain positions when image_grid_thw is None).

argv[1] tag
argv[2] edits  "7:30,9:15,11:10" (applied before training) or "none"
argv[3] nodes  "8,12,16,18"
argv[4] ref_mode: "base" or a checkpoint path for the reference model
argv[5] basis path (or "none")
argv[6] lr (default 5e-6)
argv[7] retain_weight (default 1.0)
argv[8] maxlen (default 512)
argv[9] layers  (default "4,5,6,7")

Loss: NPO(beta=0.1) on forget + RW * CE_retain, 4+4 sequences/step,
trainable = the given layers' down_proj (fp32 masters, bf16 autocast).
"""
import json
import pathlib
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForImageTextToText

import os
ROOT = pathlib.Path(os.environ["TRAIN_PROJECT_ROOT"]).resolve()
MODEL_DIR = ROOT / "models/qwen3vl8b"
FMT = "model.language_model.layers.{i}.mlp.down_proj"
DATA = ROOT / "data/processed/qwen3vl8b/three_domain_rmu_20261024"
MMLU_CFG = ROOT / "configs/three_domain_rmu_mmlu285/qwen3vl8b.json"
FILES = {"bio": "existing_bio_forget_train_seed42.jsonl",
         "cyber": "cyber_forget_train_seed42.jsonl",
         "chem": "chem_forget_train_seed42.jsonl",
         "retain": "existing_wikitext_retain_train_seed42_v2.jsonl"}
MLIST = ROOT / "data/eval/bio_rmu_effect_20261023/lists/mmlu_dev285.jsonl"
MPARQ = ROOT / "data/eval/raw/hails_mmlu_no_train_b2e1ec9"
LETTERS = ["A", "B", "C", "D"]
BETA, NL = 0.1, 4


def load_model(edits=None, basis=None, ckpt=None):
    kw = dict(torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True)
    try:
        m = AutoModelForImageTextToText.from_pretrained(str(MODEL_DIR), **kw)
    except Exception:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(str(MODEL_DIR), **kw)
    m = m.to("cuda")
    modules = dict(m.named_modules())
    if edits and edits != "none":
        b = torch.load(ROOT / basis, map_location="cpu", weights_only=True)["1000"]
        for item in edits.split(","):
            li, c = item.split(":")
            mod = modules[FMT.format(i=int(li))]
            base = mod.weight.detach()
            d = b.to(device=base.device, dtype=torch.float64) * float(c)
            with torch.no_grad():
                mod.weight.copy_((base.to(torch.float64) + d).to(base.dtype))
            print(f"edit {FMT.format(i=int(li))} c={c}", flush=True)
    if ckpt and ckpt != "base":
        sd = torch.load(ROOT / ckpt, map_location="cpu", weights_only=True)
        named = dict(m.named_parameters())
        with torch.no_grad():
            for n, t in sd.items():
                named[n].copy_(t.to(device=named[n].device, dtype=named[n].dtype))
        print(f"loaded ckpt {ckpt} ({len(sd)} tensors)", flush=True)
    print("top-level class:", type(m).__name__, flush=True)
    return m


def seq_logp(lm, ids):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    lg = out.logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    return F.log_softmax(lg, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()


def main():
    tag, edits, nodes_s, refmode, basis = sys.argv[1:6]
    LR = float(sys.argv[6]) if len(sys.argv) > 6 else 5e-6
    RW = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
    MAXLEN = int(sys.argv[8]) if len(sys.argv) > 8 else 512
    LAYERS = tuple(int(x) for x in (sys.argv[9].split(",") if len(sys.argv) > 9
                                    else ["4", "5", "6", "7"]))
    ORDER = sys.argv[10] if len(sys.argv) > 10 else "seq"
    global BETA
    if len(sys.argv) > 11:
        BETA = float(sys.argv[11])
    RSTRIDE = int(sys.argv[12]) if len(sys.argv) > 12 else 1
    SRC = sys.argv[13] if len(sys.argv) > 13 else "none"
    print(f"lr={LR} retain_weight={RW} maxlen={MAXLEN} layers={LAYERS} beta={BETA} "
          f"order={ORDER} retain_stride={RSTRIDE} src={SRC}", flush=True)
    nodes = [int(x) for x in nodes_s.split(",")]
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True,
                                        trust_remote_code=True)
    letter_ids = {L: tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS}
    by_domain = {d: [json.loads(l) for l in open(DATA / FILES[d])]
                 for d in ("bio", "cyber", "chem")}
    rows_r = [json.loads(l) for l in open(DATA / FILES["retain"])]
    if ORDER == "rotate":
        # LLaVA recipe: forget samples rotate bio/cyber/chem inside every window of 4,
        # so a short run touches all three domains instead of spending all steps on bio.
        rows_f = []
        for i in range(max(len(v) for v in by_domain.values()) * 3):
            dom = ("bio", "cyber", "chem")[i % 3]
            pos = i // 3
            if pos < len(by_domain[dom]):
                rows_f.append(by_domain[dom][pos])
    else:
        rows_f = [r for d in ("bio", "cyber", "chem") for r in by_domain[d]]
    print("forget", len(rows_f), "retain", len(rows_r), flush=True)

    # --- reference logp
    rm = load_model(ckpt=(None if refmode == "base" else refmode))
    ref = []
    with torch.no_grad():
        for r in rows_f:
            ids = torch.tensor([r["token_ids"][:MAXLEN]], dtype=torch.long, device="cuda")
            ref.append(float(seq_logp(rm, ids)))
    print("ref logp mean", sum(ref) / len(ref), flush=True)
    del rm
    torch.cuda.empty_cache()

    # --- student
    model = load_model(edits=edits, basis=basis)
    if SRC != "none":
        sd = torch.load(ROOT / SRC, map_location="cpu", weights_only=True)
        named_s = dict(model.named_parameters())
        with torch.no_grad():
            for n, t in sd.items():
                named_s[n].copy_(t.to(device=named_s[n].device, dtype=named_s[n].dtype))
        print(f"student init from {SRC} ({len(sd)} tensors)", flush=True)
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        ckpt_target = model
    except Exception as e:
        print("root gc enable failed:", type(e).__name__, e, flush=True)
        ckpt_target = model.model
        ckpt_target.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    print("gradient checkpointing on:", type(ckpt_target).__name__, flush=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad_(False)
    named = dict(model.named_parameters())
    modules = dict(model.named_modules())
    params, names = [], []
    for li in LAYERS:
        name = FMT.format(i=li) + ".weight"
        master = torch.nn.Parameter(named[name].detach().float().clone())
        module = modules[name.rsplit(".", 1)[0]]
        module.weight = master
        params.append(master)
        names.append(name)
    opt = torch.optim.Adam(params, lr=LR)
    model.train()

    mcfg = MMLU_CFG
    pre = set(json.loads(mcfg.read_text()).get("predeclared_subject_group") or [])
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
                      f"{s.replace('_', ' ')}.\n\n{src['question'].strip()}\n{opts}\nAnswer: ",
                   int(src["answer"])))

    @torch.no_grad()
    def mmlu():
        model.eval()
        m = o = n = 0
        for s, ctx, ans in mq:
            ids = torch.tensor([tok.encode(ctx, add_special_tokens=False)],
                               dtype=torch.long, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            lg = out.logits[0, -1, :].float()
            pick = max(LETTERS, key=lambda L: float(lg[letter_ids[L]]))
            ok = int(LETTERS.index(pick) == ans)
            m += ok
            if s not in pre:
                o += ok
                n += 1
        model.train()
        return m, o, n

    def retain_ce(step):
        ce = torch.zeros((), device="cuda")
        for j in range(NL):
            # RSTRIDE>1 walks the whole 320-row retain corpus instead of its first
            # 4*steps rows, which keeps capability protection broader for short runs.
            r = rows_r[(NL * (step - 1) + j) * RSTRIDE % len(rows_r)]
            ids = torch.tensor([r["token_ids"][:MAXLEN]], dtype=torch.long, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            lg = o.logits[:, :-1, :].float()
            tgt = ids[:, 1:]
            ce = ce + F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                                      reduction="mean") / NL
        return ce

    m0, o0, _ = mmlu()
    print(f"node 0: mmlu285={m0} other245={o0}", flush=True)
    out = pathlib.Path(os.environ["TRAIN_OUTPUT_DIR"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    traj = {0: {"mmlu285": m0, "other245": o0}}

    def save(step):
        torch.save({n: p.detach().to(torch.bfloat16).cpu() for n, p in zip(names, params)},
                   out / f"step{step}_updated_parameters.pt")

    save(0)
    for step in range(1, max(nodes) + 1):
        q = (step - 1) // 3
        opt.zero_grad(set_to_none=True)
        npo = torch.zeros((), device="cuda")
        for j in range(NL):
            idx = (q * NL + j) % len(rows_f)
            ids = torch.tensor([rows_f[idx]["token_ids"][:MAXLEN]], dtype=torch.long,
                               device="cuda")
            lp = seq_logp(model, ids)
            npo = npo - (2.0 / BETA) * F.logsigmoid(-BETA * (lp - ref[idx])) / NL
        loss = npo + RW * retain_ce(step)
        loss.backward()
        opt.step()
        print(f"step {step} loss={float(loss):.4f} npo={float(npo):.4f}", flush=True)
        del loss, npo
        torch.cuda.empty_cache()
        if step in nodes:
            m, o, _ = mmlu()
            traj[step] = {"mmlu285": m, "other245": o}
            print(f"node {step}: mmlu285={m} other245={o}", flush=True)
            save(step)
    (out / "trajectory.json").write_text(json.dumps(traj, indent=2))
    print("json " + json.dumps(traj, sort_keys=True))
    print("saved to", out, flush=True)


if __name__ == "__main__":
    main()

