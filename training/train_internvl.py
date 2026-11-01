#!/usr/bin/env python3
"""InternVL3-8B — GROM edit + one NPO step, starting from the base model.

Pipeline
--------
The experiment output is an assembly of two things: GROM closed-form edits on layers
5/7/9 (never trained as such) and one NPO step applied to layers 4-7.

  1. GROM edits   layer 5 += 46 * P1_5 ,  layer 7 += 23 * P1_7 ,  layer 9 += 46 * P1_9
                  (per-layer bases: every layer gets its own closed-form solution)
  2. NPO 1 step   beta 0.1, lr 1e-6, retain weight 1.0, layers 4-7, reference = base
  3. save         layers 4,5,6,7 (trained; 5 and 7 carry their edit) + layer 9 (edit only)

Recorded result associated with this experiment (WMDP three-domain, 3668 questions, 0-shot option likelihood; not the current repository C/S summary):
C = 76.1026 %, other245 = 148 / 245, MMLU285 = 162.

Internal reproduction check recorded for this experiment: running this script reproduced its reference tensor file bit-for-bit
(max |difference| = 0.0 on all five tensors).

Usage
-----
    python train_internvl.py [--out <path>] [--verify <checkpoint to compare against>]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dsh_train_lib as L

KEY = "internvl"
EDIT_BASIS = os.environ.get("DSH_INTERNVL_GROM_BASIS", "runs/bases/internvl_grom_basis_multi.pt")
EDITS = [(5, 46.0), (7, 23.0), (9, 46.0)]
TRAIN_LAYERS = (4, 5, 6, 7)
SAVE_LAYERS = (4, 5, 6, 7, 9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/output/internvl/updated_parameters.pt")
    ap.add_argument("--verify", default=None, help="reference checkpoint to compare against")
    args = ap.parse_args()

    torch.manual_seed(0)
    forget, retain = L.load_rows(KEY)

    print("== stage 0: reference log-probs from the un-edited base ==", flush=True)
    ref_model, ref_lm = L.load_model(KEY)
    ref = L.reference_logps(ref_lm, forget, tag="base")
    del ref_model, ref_lm
    torch.cuda.empty_cache()

    print("== stage 1: GROM closed-form edits ==", flush=True)
    model, lm = L.load_model(KEY)
    L.apply_grom(model, KEY, EDIT_BASIS, EDITS)

    print("== stage 2: NPO 1 step (lr 1e-6) on layers 4-7 ==", flush=True)
    L.run_npo(model, lm, KEY, forget, retain, ref, layers=TRAIN_LAYERS,
              lr=1e-6, rw=1.0, steps=1, tag="internvl-npo", out_dir="runs/output/internvl/tmp")

    print("== stage 3: save checkpoint ==", flush=True)
    payload = L.save_delivery(model, KEY, SAVE_LAYERS, args.out)
    if args.verify:
        L.verify_against(payload, args.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
