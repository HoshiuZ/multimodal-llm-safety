#!/usr/bin/env python3
"""Materialise a self-contained Qwen checkpoint from an existing checkpoint plus a
post-hoc GROM closed-form edit.

  W_layer <- (W_layer_fp64_from_ckpt + c * P1).to(bf16)

Writes the 4 down_proj tensors (layers 4-7) as bf16 so the result is a single file
that reproduces exactly what the scorer computes in memory.

argv: src_ckpt edits "7:45" basis out_path
"""
import json
import pathlib
import sys

import torch
from transformers import AutoModelForImageTextToText

import os
ROOT = pathlib.Path(os.environ["TRAIN_PROJECT_ROOT"]).resolve()
MODEL_DIR = ROOT / "models/qwen3vl8b"
FMT = "model.language_model.layers.{i}.mlp.down_proj"


def main():
    src, edits, basis_s, out_path = sys.argv[1:5]
    kw = dict(torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(MODEL_DIR), **kw)
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(MODEL_DIR), **kw)
    model = model.to("cuda").eval()
    modules = dict(model.named_modules())
    names = dict(model.named_parameters())

    sd = torch.load(ROOT / src, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for n, t in sd.items():
            names[n].copy_(t.to(device=names[n].device, dtype=names[n].dtype))
    print(f"loaded {src} ({len(sd)} tensors)", flush=True)

    basis = torch.load(ROOT / basis_s, map_location="cpu", weights_only=True)["1000"]
    for item in edits.split(","):
        li, c = item.split(":")
        mod = modules[FMT.format(i=int(li))]
        b = mod.weight.detach().to(torch.float64)
        d = basis.to(device=b.device, dtype=torch.float64) * float(c)
        with torch.no_grad():
            mod.weight.copy_((b + d).to(mod.weight.dtype))
        print(f"post edit {FMT.format(i=int(li))} c={c}", flush=True)

    out = {}
    for li in (4, 5, 6, 7):
        n = FMT.format(i=li) + ".weight"
        out[n] = names[n].detach().to(torch.bfloat16).cpu().contiguous()
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    import hashlib
    h = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(json.dumps({"out": str(out_path), "sha256": h,
                      "bytes": out_path.stat().st_size}, indent=2), flush=True)


if __name__ == "__main__":
    main()

