#!/usr/bin/env python3
"""Qwen3-VL-8B — GROM edit + NPO + repair + a cyber-targeted second edit.

Pipeline
--------
  1. GROM edit       layer 7 += 45 * P1                        three-domain basis
  2. NPO 20 steps    lr 1e-5, retain weight 0.5, layers 4-7    reference = base
  3. repair          10 steps retain-CE, lr 5e-7, layers 4-7
  4. NPO 2 steps     lr 1e-5, retain weight 1.0                -> 10%-budget checkpoint
  5. GROM edit       layer 7 += 50 * P1_cyber                  cyber-domain-only basis
  6. NPO 2 steps     lr 1e-5, retain weight 1.0                -> final checkpoint

Recorded result associated with this experiment (WMDP three-domain, 3668 questions, 0-shot option likelihood; not the current repository C/S summary):
C = 89.8168 % (89.0110 % from this script), other245 = 155 / 245, MMLU285 = 170.

Internal reproduction note: every tensor lands within one bf16 unit of its reference file
(max |difference| = 2.4e-4, i.e. <= 2 ULP), but the run is not byte-identical because
34 optimizer steps accumulate GPU-level floating-point non-determinism. Score delta: 1067
vs 1056 correct answers (0.8 C points) at identical capability (other245 = 155).

Usage
-----
    python train_qwen.py [--out <path>] [--verify <checkpoint to compare against>]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dsh_train_lib as L

KEY = "qwen"
EDIT_BASIS = os.environ.get("DSH_QWEN_GROM_BASIS", "runs/bases/qwen_grom_basis.pt")
CYBER_BASIS = os.environ.get("DSH_QWEN_CYBER_GROM_BASIS", "runs/bases/qwen_grom_basis_cyber.pt")
TRAIN_LAYERS = (4, 5, 6, 7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/output/qwen/updated_parameters.pt")
    ap.add_argument("--verify", default=None, help="reference checkpoint to compare against")
    args = ap.parse_args()

    torch.manual_seed(0)
    forget, retain = L.load_rows(KEY)

    print("== stage 0: reference log-probs from the un-edited base ==", flush=True)
    ref_model, ref_lm = L.load_model(KEY)
    ref = L.reference_logps(ref_lm, forget, tag="base")
    del ref_model, ref_lm
    torch.cuda.empty_cache()

    print("== stage 1: GROM edit (three-domain basis, layer 7, c=45) ==", flush=True)
    model, lm = L.load_model(KEY)
    L.apply_grom(model, KEY, EDIT_BASIS, [(7, 45.0)])

    print("== stage 2: NPO 20 steps (lr 1e-5, retain weight 0.5) ==", flush=True)
    L.run_npo(model, lm, KEY, forget, retain, ref, layers=TRAIN_LAYERS,
              lr=1e-5, rw=0.5, steps=20, tag="qwen-npo20", out_dir="runs/output/qwen/tmp")
    L.bf16_roundtrip(model, KEY, TRAIN_LAYERS)

    print("== stage 3: retain repair 10 steps (lr 5e-7) ==", flush=True)
    L.run_npo(model, lm, KEY, forget, retain, ref, layers=TRAIN_LAYERS,
              lr=1e-5, rw=0.5, steps=0, tag="qwen-repair", out_dir="runs/output/qwen/tmp",
              repair=10, repair_lr=5e-7, repair_layers=TRAIN_LAYERS)
    L.bf16_roundtrip(model, KEY, TRAIN_LAYERS)

    print("== stage 4: NPO 2 steps (lr 1e-5, retain weight 1.0) ==", flush=True)
    L.run_npo(model, lm, KEY, forget, retain, ref, layers=TRAIN_LAYERS,
              lr=1e-5, rw=1.0, steps=2, tag="qwen-npo2", out_dir="runs/output/qwen/tmp")
    L.bf16_roundtrip(model, KEY, TRAIN_LAYERS)

    print("== stage 5: cyber-only GROM edit (layer 7, c=50) ==", flush=True)
    L.apply_grom(model, KEY, CYBER_BASIS, [(7, 50.0)])

    print("== stage 6: NPO 2 steps (lr 1e-5, retain weight 1.0) ==", flush=True)
    L.run_npo(model, lm, KEY, forget, retain, ref, layers=TRAIN_LAYERS,
              lr=1e-5, rw=1.0, steps=2, tag="qwen-npo2b", out_dir="runs/output/qwen/tmp")

    print("== save checkpoint ==", flush=True)
    payload = L.save_delivery(model, KEY, TRAIN_LAYERS, args.out)
    if args.verify:
        L.verify_against(payload, args.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
