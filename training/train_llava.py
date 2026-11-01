#!/usr/bin/env python3
"""LLaVA-NeXT-8B — GROM edit + NPO + repair, starting from the base model.

Reference model: the un-edited base model itself. No external reference checkpoint.

Pipeline
--------
The 2026-11-01 base-reference experiment output is an assembly of GROM closed-form edits
on layers 8/10/12 (not trained by NPO) and NPO/repair-trained weights on layers 4-7.
It is separate from the existing published HF J8 LLaVA model; this experiment's checkpoint
is not included in this code increment.

  1. GROM edit      layer 8 += 45 * P1                       three-domain basis
  2. NPO 5 steps    lr 1e-5, retain weight 1.0, layers 5-7   reference = base
  3. repair         60 steps retain-CE, lr 5e-7, layers 5-7
  4. NPO 2 steps    lr 5e-6, retain weight 1.0, layers 5-7
  5. assemble       base layer 4 + trained 5-7 + edits (8:30, 10:15, 12:10)
  6. NPO 16 steps   lr 5e-6, retain weight 1.0, layers 4-7, reference = base
  7. assemble       trained 4-7 + edits (8:30, 10:15, 12:10)

Recorded result for this separate base-reference experiment (WMDP three-domain, 3668 questions, 0-shot option likelihood):
C = 80.4348 %, other245 = 131 / 245, MMLU285 = 145.

Usage
-----
    python train_llava.py [--out <path>] [--verify <checkpoint to compare against>]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dsh_train_lib as L

KEY = "llava"
GROM_BASIS = os.environ.get("DSH_LLAVA_GROM_BASIS", "runs/bases/llava_grom_basis.pt")

EARLY_EDIT = (8, 45.0)                              # edit present while the early stages train
FINAL_EDITS = [(8, 30.0), (10, 15.0), (12, 10.0)]   # edits baked into the assembled checkpoint
EARLY_LAYERS = (5, 6, 7)                            # layers trained by stages 2-4
LATE_LAYERS = (4, 5, 6, 7)                          # layers trained by stage 6


def assemble(trained_state: dict, layers, out_path: str) -> dict:
    """Build a checkpoint file from a clean base model.

    The edited layers are recomputed from the *base* weights: the training model already
    carries the layer-8 c=45 edit, so adding the final c=30 edit on top of it would double
    the edit (base + 75*P1 instead of base + 30*P1).
    """
    model, _ = L.load_model(KEY)
    L.apply_grom(model, KEY, GROM_BASIS, FINAL_EDITS)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in trained_state.items():
            named[name].copy_(tensor.to(device=named[name].device, dtype=torch.float32))
    return L.save_delivery(model, KEY, sorted(set(layers) | {layer for layer, _ in FINAL_EDITS}), out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/output/llava/updated_parameters.pt")
    ap.add_argument("--verify", default=None, help="reference checkpoint to compare against")
    args = ap.parse_args()

    torch.manual_seed(0)
    forget, retain = L.load_rows(KEY)

    print("== stage 0: reference log-probs from the un-edited base ==", flush=True)
    ref_model, ref_lm = L.load_model(KEY)
    ref = L.reference_logps(ref_lm, forget, tag="base")
    del ref_model, ref_lm
    torch.cuda.empty_cache()

    print("== stage 1: GROM edit layer 8 c=45 ==", flush=True)
    model, lm = L.load_model(KEY)
    L.apply_grom(model, KEY, GROM_BASIS, [EARLY_EDIT])

    print("== stage 2: NPO 5 steps (lr 1e-5) on layers 5-7 ==", flush=True)
    L.run_npo(model, lm, KEY, forget, retain, ref, layers=EARLY_LAYERS,
              lr=1e-5, rw=1.0, steps=5, tag="llava-npo5", out_dir="runs/output/llava/tmp")
    L.bf16_roundtrip(model, KEY, EARLY_LAYERS)

    print("== stage 3: retain repair 60 steps (lr 5e-7) ==", flush=True)
    L.run_npo(model, lm, KEY, forget, retain, ref, layers=EARLY_LAYERS,
              lr=1e-5, rw=1.0, steps=0, tag="llava-repair", out_dir="runs/output/llava/tmp",
              repair=60, repair_lr=5e-7, repair_layers=EARLY_LAYERS)
    L.bf16_roundtrip(model, KEY, EARLY_LAYERS)

    print("== stage 4: NPO 2 steps (lr 5e-6) ==", flush=True)
    L.run_npo(model, lm, KEY, forget, retain, ref, layers=EARLY_LAYERS,
              lr=5e-6, rw=1.0, steps=2, tag="llava-npo2", out_dir="runs/output/llava/tmp")
    L.bf16_roundtrip(model, KEY, EARLY_LAYERS)

    print("== stage 5: assemble the early checkpoint ==", flush=True)
    assemble(L.current_state(model, KEY, EARLY_LAYERS), EARLY_LAYERS,
             "runs/output/llava/delivery_early.pt")

    print("== stage 6: NPO 16 steps (lr 5e-6) on layers 4-7 ==", flush=True)
    model, lm = L.load_model(KEY)
    L.load_state(model, "runs/output/llava/delivery_early.pt")
    L.run_npo(model, lm, KEY, forget, retain, ref, layers=LATE_LAYERS,
              lr=5e-6, rw=1.0, steps=16, tag="llava-deepen", out_dir="runs/output/llava/tmp")

    print("== stage 7: assemble the final checkpoint ==", flush=True)
    payload = assemble(L.current_state(model, KEY, LATE_LAYERS), LATE_LAYERS, args.out)
    if args.verify:
        L.verify_against(payload, args.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
