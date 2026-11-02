#!/usr/bin/env python3
"""Shared training primitives for the three unlearning checkpoints.

Every numerical path mirrors the code used for the associated experiment checkpoints:

  * fp32 master weights with bf16 autocast (an lr of 5e-7 does not move a pure-bf16 weight);
  * Adam, 4 forget + 4 retain sequences per optimizer step, 512-token truncation;
  * forget rows consumed in corpus order as ``(q*4 + j) % len(rows)`` with ``q = (step-1)//3``;
  * retain rows as ``(4*(step-1) + j) % len(retain)``;
  * NPO loss ``-(2/beta) * logsigmoid(-beta * (logp_theta - logp_ref))`` plus ``rw * CE_retain``;
  * a bf16 save/load boundary between pipeline stages (this rounding is part of the recipe).

Set ``TRAIN_ROOT`` to point at the data/checkpoint root (defaults to the repository root).
"""
from __future__ import annotations

import json
import os
import pathlib

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = pathlib.Path(os.environ.get("TRAIN_ROOT", pathlib.Path(__file__).resolve().parents[2])).resolve()

SPECS = {
    "llava": {
        "loader": "llava",
        "dir": "text/models/llava-base",
        "fmt": "model.language_model.layers.{i}.mlp.down_proj",
        "data": "data/processed/llava_next8b/three_domain_rmu_20261024",
    },
    "qwen": {
        "loader": "auto",
        "dir": "text/models/qwen-base",
        "fmt": "model.language_model.layers.{i}.mlp.down_proj",
        "data": "data/processed/qwen3vl8b/three_domain_rmu_20261024",
    },
    "internvl": {
        "loader": "auto",
        "dir": "text/models/internvl-base",
        "fmt": "language_model.model.layers.{i}.mlp.down_proj",
        "data": "data/processed/internvl3_8b/three_domain_rmu_20261024",
    },
}

FILES = {
    "bio": "existing_bio_forget_train_seed42.jsonl",
    "cyber": "cyber_forget_train_seed42.jsonl",
    "chem": "chem_forget_train_seed42.jsonl",
    "retain": "existing_wikitext_retain_train_seed42_v2.jsonl",
}

BETA = 0.1
MAXLEN = 512
NL = 4          # 4 forget + 4 retain sequences per optimizer step
DEV = "cuda"


# ---------------------------------------------------------------- model loading
def load_model(key: str):
    spec = SPECS[key]
    kw = dict(torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True)
    if spec["loader"] == "llava":
        from transformers import LlavaNextForConditionalGeneration
        model = LlavaNextForConditionalGeneration.from_pretrained(str(ROOT / spec["dir"]), **kw)
    else:
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(str(ROOT / spec["dir"]), **kw)
        except Exception:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(str(ROOT / spec["dir"]), **kw)
    model = model.to(DEV)
    return model, causal_lm(model)


def causal_lm(model):
    """Return the module that owns ``lm_head`` and accepts ``input_ids``."""
    for name, mod in model.named_modules():
        inner = getattr(mod, "model", None)
        if hasattr(mod, "lm_head") and inner is not None and hasattr(inner, "layers"):
            print(f"[lib] causal LM module: {name or '<root>'}", flush=True)
            return mod
    if hasattr(model, "lm_head"):
        print("[lib] causal LM module: <root> (has lm_head)", flush=True)
        return model
    raise SystemExit("no causal LM module found")


def load_state(model, path) -> None:
    full = pathlib.Path(path)
    if not full.is_absolute():
        full = ROOT / full
    state = torch.load(full, map_location="cpu", weights_only=True)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            named[name].copy_(tensor.to(device=named[name].device, dtype=torch.float32))
    print(f"[lib] loaded state from {path}", flush=True)


# ------------------------------------------------------------------ GROM edits
def _pick_basis(obj, layer: str):
    """Basis dicts come in two shapes: {'1000': T} or {'5': {'1000': T}, ...} (per-layer)."""
    if isinstance(obj.get("1000"), torch.Tensor):
        return obj["1000"]
    return obj[layer]["1000"]


def apply_grom(model, key: str, basis_path: str, edits) -> None:
    """Apply ``W[layer] += c * P1`` (float64 accumulation, bf16 storage) — the closed-form edit."""
    obj = torch.load(ROOT / basis_path, map_location="cpu", weights_only=True)
    fmt = SPECS[key]["fmt"]
    modules = dict(model.named_modules())
    for layer, c in edits:
        b = _pick_basis(obj, str(layer))
        module = modules[fmt.format(i=layer)]
        base = module.weight.detach()
        if tuple(base.shape) != tuple(b.shape):
            raise SystemExit(f"basis/shape mismatch at layer {layer}: {tuple(base.shape)} vs {tuple(b.shape)}")
        delta = b.to(device=base.device, dtype=torch.float64) * float(c)
        with torch.no_grad():
            module.weight.copy_((base.to(torch.float64) + delta).to(base.dtype))
        print(f"[lib] grom edit layer {layer} c={c}  |P1|_2={float(b.float().norm()):.4f}", flush=True)


# ---------------------------------------------------------------- data + logp
def load_rows(key: str):
    from check_inputs import check
    check(key, ROOT)
    data = ROOT / SPECS[key]["data"]
    forget = [json.loads(line) for name in ("bio", "cyber", "chem")
              for line in open(data / FILES[name])]
    retain = [json.loads(line) for line in open(data / FILES["retain"])]
    print(f"[lib] forget rows={len(forget)} retain rows={len(retain)}", flush=True)
    return forget, retain


def seq_logp(lm, ids):
    with torch.autocast(DEV, dtype=torch.bfloat16):
        out = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    logits = out.logits[:, :-1, :].float()
    target = ids[:, 1:]
    return F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1).mean()


@torch.no_grad()
def reference_logps(lm, forget, tag="base"):
    ref = []
    for row in forget:
        ids = torch.tensor([row["token_ids"][:MAXLEN]], dtype=torch.long, device=DEV)
        ref.append(float(seq_logp(lm, ids)))
    print(f"[lib] reference ({tag}) logp mean = {sum(ref) / len(ref):.4f}", flush=True)
    return ref


# ------------------------------------------------------------------- training
def make_masters(model, key: str, layers):
    """Replace bf16 module weights with fp32 trainable masters."""
    fmt = SPECS[key]["fmt"]
    named = dict(model.named_parameters())
    modules = dict(model.named_modules())
    masters, names = [], []
    for layer in layers:
        name = fmt.format(i=layer) + ".weight"
        master = torch.nn.Parameter(named[name].detach().float().clone())
        modules[name.rsplit(".", 1)[0]].weight = master
        masters.append(master)
        names.append(name)
    return masters, names


def retain_ce(lm, retain, step: int):
    ce = torch.zeros((), device=DEV)
    for j in range(NL):
        row = retain[(NL * (step - 1) + j) % len(retain)]
        ids = torch.tensor([row["token_ids"][:MAXLEN]], dtype=torch.long, device=DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            out = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        logits = out.logits[:, :-1, :].float()
        target = ids[:, 1:]
        ce = ce + F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)) / NL
    return ce


def npo_step(lm, forget, ref, step: int, beta: float = BETA):
    q = (step - 1) // 3
    loss = torch.zeros((), device=DEV)
    for j in range(NL):
        idx = (q * NL + j) % len(forget)
        ids = torch.tensor([forget[idx]["token_ids"][:MAXLEN]], dtype=torch.long, device=DEV)
        lp = seq_logp(lm, ids)
        loss = loss - (2.0 / beta) * F.logsigmoid(-beta * (lp - ref[idx])) / NL
    return loss


def run_npo(model, lm, key, forget, retain, ref, *, layers, lr, rw, steps,
            tag, out_dir, save_at=(), repair=0, repair_lr=5e-7, repair_layers=None):
    """NPO steps (optionally preceded by a retain-only repair phase)."""
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(lm, "enable_input_require_grads"):
        lm.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad_(False)

    masters, names = make_masters(model, key, layers)
    opt = torch.optim.Adam(masters, lr=lr)
    lm.train()
    print(f"[{tag}] NPO lr={lr} rw={rw} steps={steps} layers={tuple(layers)}", flush=True)

    if repair > 0:
        rlayers = repair_layers or layers
        rm, _ = (masters, names) if tuple(rlayers) == tuple(layers) else make_masters(model, key, rlayers)
        opt_r = torch.optim.Adam(rm, lr=repair_lr)
        for s in range(1, repair + 1):
            opt_r.zero_grad(set_to_none=True)
            loss = retain_ce(lm, retain, s)
            loss.backward()
            opt_r.step()
            del loss
            torch.cuda.empty_cache()
        print(f"[{tag}] repair {repair} steps lr={repair_lr} layers={tuple(rlayers)}", flush=True)

    saved = {}
    out_path = ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = npo_step(lm, forget, ref, step) + rw * retain_ce(lm, retain, step)
        loss.backward()
        opt.step()
        del loss
        torch.cuda.empty_cache()
        if step in save_at:
            snapshot = {n: p.detach().to(torch.bfloat16).cpu() for n, p in zip(names, masters)}
            torch.save(snapshot, out_path / f"step{step}_updated_parameters.pt")
            print(f"[{tag}] saved step{step}", flush=True)
            saved[step] = snapshot
    return saved


# ------------------------------------------------------------------- assembling
def current_state(model, key: str, layers) -> dict:
    """Snapshot the given layers as bf16 cpu tensors (same layout as a checkpoint file)."""
    fmt = SPECS[key]["fmt"]
    named = dict(model.named_parameters())
    return {fmt.format(i=layer) + ".weight": named[fmt.format(i=layer) + ".weight"]
            .detach().to(torch.bfloat16).cpu() for layer in layers}


def save_delivery(model, key, layers, out_path, extra_note=""):
    """Write a checkpoint dict holding the named layers' current weights (bf16)."""
    payload = current_state(model, key, layers)
    out = ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"[lib] wrote {out} ({len(payload)} tensors) {extra_note}", flush=True)
    return payload


def bf16_roundtrip(model, key: str, layers) -> None:
    """Simulate the save/load boundary between pipeline stages (this rounding matters)."""
    fmt = SPECS[key]["fmt"]
    named = dict(model.named_parameters())
    modules = dict(model.named_modules())
    with torch.no_grad():
        for layer in layers:
            name = fmt.format(i=layer) + ".weight"
            modules[name.rsplit(".", 1)[0]].weight = torch.nn.Parameter(
                named[name].detach().to(torch.bfloat16).float())
    print("[lib] bf16 round-trip applied (stage boundary)", flush=True)


def verify_against(payload, reference_path) -> bool:
    """Compare a produced payload with a reference checkpoint file (bit-exact + max abs diff)."""
    ref = torch.load(ROOT / reference_path, map_location="cpu", weights_only=True)
    if set(ref) != set(payload):
        print(f"[verify] KEY MISMATCH: produced {sorted(payload)} vs reference {sorted(ref)}", flush=True)
        return False
    ok = True
    for name in sorted(ref):
        diff = (payload[name].float() - ref[name].float()).abs()
        same = torch.equal(payload[name], ref[name])
        rel = float(diff.max()) / max(float(ref[name].float().abs().max()), 1e-9)
        print(f"[verify] {name}: bitexact={same} max|d|={float(diff.max()):.3e} rel={rel:.3e}", flush=True)
        ok = ok and same
    print(f"[verify] BIT-EXACT: {ok}", flush=True)
    return ok


def tokenizer(key: str):
    return AutoTokenizer.from_pretrained(str(ROOT / SPECS[key]["dir"]),
                                        local_files_only=True, trust_remote_code=True)
