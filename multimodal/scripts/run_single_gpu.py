#!/usr/bin/env python3
"""Run the released multimodal method and its original-model baseline on one GPU.

Put each evaluation view in multimodal/data/input/<view>/ with a manifest.jsonl
and its referenced images. Model and adapter paths follow models/README.md.
The output contains responses for independent scoring; this entry point does
not call the project's local judge.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MODEL_DIRS = {
    "qwen3vl8b": ("Qwen3-VL-8B-Instruct", "qwen3vl8b-m1-vlguard-v1"),
    "llavanext8b": ("llama3-llava-next-8b-hf", "llavanext8b-m1-vlguard-v1"),
    "internvl3_8b": ("InternVL3-8B", "internvl3-8b-m1-vlguard-v1"),
}


def command(args: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)


def device_description() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    device = torch.cuda.get_device_properties(0)
    return f"{device.name} ({device.total_memory / 1024**3:.1f} GiB)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODEL_DIRS, help="target model")
    args = parser.parse_args()
    models = Path(os.environ.get("MM_MODELS_ROOT", str(Path.home() / "models"))).expanduser().resolve()
    input_root = Path(os.environ.get("MM_INPUT_ROOT", str(ROOT / "data" / "input"))).expanduser().resolve()
    output_root = Path(os.environ.get("MM_OUTPUT_ROOT", str(ROOT / "outputs"))).expanduser().resolve()
    # Keep the virtualenv executable path: resolving its symlink can escape the environment.
    common_python = Path(sys.executable).absolute()
    target_python = common_python
    if args.model == "internvl3_8b":
        target_python = Path(os.environ.get("MM_INTERNVL_PYTHON", str(ROOT.parent / ".venv-mm-internvl" / "bin" / "python"))).expanduser().absolute()
    base_name, adapter_name = MODEL_DIRS[args.model]
    base = models / base_name
    adapter = models / "adapters" / adapter_name
    reviewer = models / "Qwen3-VL-8B-Instruct"
    for label, path in (("base model", base), ("M1 adapter", adapter), ("M3 reviewer", reviewer), ("target Python", target_python)):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    views = sorted(input_root.glob("*/manifest.jsonl"))
    if not views:
        raise FileNotFoundError(f"no <view>/manifest.jsonl under {input_root}; see multimodal/data/README.md")
    print(f"GPU: {device_description()}", flush=True)
    for manifest in views:
        view = manifest.parent.name
        out = output_root / args.model / view
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "run.log"
        print(f"model={args.model} view={view} device=CUDA:0", flush=True)
        common = ["--model-key", args.model, "--manifest", str(manifest), "--data-root", str(manifest.parent), "--output-dir", str(out), "--num-shards", "1", "--rank", "0", "--device-index", "0"]
        for state in ("orig", "m1"):
            print(f"[{view}] generate: {state}", flush=True)
            extra = ["--base-model", str(base), "--state", state]
            if state == "m1":
                extra += ["--adapter", str(adapter)]
            command([str(target_python), "-B", str(SCRIPTS / "mm_eval.py"), "generate", *common, *extra], log_path)
        print(f"[{view}] run M3 visual review", flush=True)
        command([str(common_python), "-B", str(SCRIPTS / "mm_eval.py"), "review", *common, "--m1-input-dir", str(out), "--reviewer-model", str(reviewer)], log_path)
        for state in ("orig", "m1_m3_v8"):
            command([str(common_python), "-B", str(SCRIPTS / "merge_shards.py"), "--manifest", str(manifest), "--output-dir", str(out), "--model-key", args.model, "--state", state, "--num-shards", "1"], log_path)
        print(f"[{view}] done: baseline and method responses saved", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}; see the view output run.log for details", file=sys.stderr, flush=True)
        raise SystemExit(1)
