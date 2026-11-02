#!/usr/bin/env python3
"""Stable CLI for the pinned three-domain RMU data preparation pipeline.

This is a repo-relative entry point for the bundled pinned selection and
tokenization implementation. No training or evaluation is performed here.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import runpy
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMPLEMENTATION = pathlib.Path(__file__).resolve().parent / "_pipeline" / "prepare_three_domain_rmu_data.py"
PHASES = ("prepare-all", "select", "tokenize", "link-reuse", "verify", "freeze")
MODEL_KEYS = ("qwen3vl8b", "llava_next8b", "internvl3_8b")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="isolated project-layout output/input root (default: repository root)")
    parser.add_argument("--source-dir", help="authorized local source tree with data/source_import/... layout")
    parser.add_argument("phase", choices=PHASES,
                        help="pipeline phase; see text/training/README.md")
    parser.add_argument("--model-key", choices=MODEL_KEYS, help="required for link-reuse")
    parser.add_argument("--exclude-hashes", help="optional initial exact-overlap exclusion manifest")
    parser.add_argument("--output-name", help="frozen manifest filename for prepare-all")
    args, remainder = parser.parse_known_args()
    if args.data_root:
        os.environ["RMU_DATA_ROOT"] = str(pathlib.Path(args.data_root).resolve())
    if args.source_dir:
        os.environ["RMU_SOURCE_DIR"] = str(pathlib.Path(args.source_dir).resolve())
    if args.phase == "prepare-all":
        if not args.data_root:
            parser.error("prepare-all requires --data-root")
        root = pathlib.Path(args.data_root).resolve()
        source_root = pathlib.Path(args.source_dir).resolve() if args.source_dir else root
        stage_base = pathlib.Path("data/processed/three_domain_rmu_20261024/work/prepare_all")
        audit_impl = pathlib.Path(__file__).resolve().parent / "_pipeline" / "audit_three_domain_candidates_against_wmdp_stems.py"
        excluded = args.exclude_hashes
        selected_stage = None
        for attempt in range(10):
            selected_stage = stage_base / f"selection_{attempt:02d}"
            command = [sys.executable, str(IMPLEMENTATION), "select", "--stage-dir", str(selected_stage)]
            if excluded:
                command.extend(["--exclude-hashes", str(excluded)])
            subprocess.run(command, env=os.environ.copy(), check=True)
            candidate_args = []
            for domain in ("cyber", "chem"):
                candidate_args.extend(["--candidates", str(root / selected_stage / f"selected_{domain}.jsonl")])
            audit_path = selected_stage / "wmdp_stem_audit.json"
            subprocess.run([sys.executable, str(audit_impl), *candidate_args,
                            "--output", str(root / audit_path)], env=os.environ.copy(), check=True)
            audit = json.loads((root / audit_path).read_text())
            if audit["matched_candidate_count"] == 0:
                break
            excluded = str(root / audit_path)
        else:
            raise RuntimeError("Exact C-stem overlap remained after 10 deterministic exclusion passes")
        for model_key in MODEL_KEYS:
            subprocess.run([sys.executable, str(IMPLEMENTATION), "tokenize",
                            "--selection-dir", str(selected_stage), "--model-key", model_key],
                           env=os.environ.copy(), check=True)
            subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "link-reuse",
                            "--data-root", str(root), "--source-dir", str(source_root),
                            "--model-key", model_key], env=os.environ.copy(), check=True)
            subprocess.run([sys.executable, str(IMPLEMENTATION), "verify",
                            "--selection-dir", str(selected_stage), "--model-key", model_key],
                           env=os.environ.copy(), check=True)
        freeze = [sys.executable, str(IMPLEMENTATION), "freeze", "--selection-dir",
                  str(selected_stage), "--audit-report", str(audit_path)]
        if args.output_name:
            freeze.extend(["--output-name", args.output_name])
        subprocess.run(freeze, env=os.environ.copy(), check=True)
        return 0
    if args.phase == "link-reuse":
        if not args.data_root or not args.model_key:
            parser.error("link-reuse requires --data-root and --model-key")
        root = pathlib.Path(args.data_root).resolve()
        source_root = pathlib.Path(args.source_dir).resolve() if args.source_dir else root
        config = json.loads((pathlib.Path(__file__).resolve().parent / "configs" / "bio_rmu_effect.json").read_text())
        model = config["models"][args.model_key]
        model_dir = root / model["data_dir"]
        source_model_dir = source_root / model["data_dir"]
        output_dir = model_dir / "three_domain_rmu_20261024"
        output_dir.mkdir(parents=True, exist_ok=True)
        pairs = (
            ("bio_rmu_effect_forget_train_seed42.jsonl", "existing_bio_forget_train_seed42.jsonl"),
            ("bio_rmu_effect_retain_train_seed42_v2.jsonl", "existing_wikitext_retain_train_seed42_v2.jsonl"),
        )
        for original_name, output_name in pairs:
            original, destination = source_model_dir / original_name, output_dir / output_name
            if not original.is_file():
                raise FileNotFoundError(f"Authorized reusable source file missing: {original_name}")
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite reusable output: {output_name}")
            os.link(original, destination)
        print(json.dumps({"model_key": args.model_key, "linked_reuse_files": 2}, ensure_ascii=False))
        return 0
    sys.argv = [str(IMPLEMENTATION), args.phase, *remainder]
    if args.model_key:
        sys.argv.extend(["--model-key", args.model_key])
    if args.exclude_hashes:
        sys.argv.extend(["--exclude-hashes", args.exclude_hashes])
    if args.output_name:
        sys.argv.extend(["--output-name", args.output_name])
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
