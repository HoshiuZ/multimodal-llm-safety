#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run one original MLLM over the complete MM-SafetyBench manifest.

This writes raw generations only. The separate judge step produces ASR labels.
The script is resumable: completed sample_ids are skipped only with --resume.
Multiple workers can process disjoint manifest shards with --num-shards and
--shard-index; merge their outputs before running the separate judge step.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from xmodal import manifest as MF  # noqa: E402
from xmodal import models as MD  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint_model_tree(model_path: str) -> tuple[str, list[dict]]:
    """Hash every checkpoint file so the baseline is pinned even without HF refs/main."""
    entries: list[dict] = []
    for current, dirs, filenames in os.walk(model_path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != ".cache")
        for filename in sorted(filenames):
            path = os.path.join(current, filename)
            if not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, model_path).replace(os.sep, "/")
            size = os.path.getsize(path)
            digest = sha256_file(path)
            entries.append({"path": rel, "bytes": size, "sha256": digest})
            print(f"checkpoint hash: {rel} ({size / (1024**3):.2f} GiB)", flush=True)
    if not entries:
        raise RuntimeError(f"no regular checkpoint files found under {model_path}")
    tree = hashlib.sha256()
    for item in sorted(entries, key=lambda x: x["path"]):
        tree.update(item["path"].encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(item["bytes"]).encode("ascii"))
        tree.update(b"\0")
        tree.update(item["sha256"].encode("ascii"))
        tree.update(b"\n")
    return tree.hexdigest(), entries


def expand_path(value: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(value)))


def config_path(value: str) -> str:
    return expand_path(value if os.path.isabs(value) else os.path.join(ROOT, value))


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_inputs(cfg: dict, model_key: str, manifest_path: str, data_root: str) -> tuple[list[dict], str]:
    if model_key not in cfg["models"]:
        raise ValueError(f"model key is not enabled in config: {model_key}")
    rows = MF.read_csv(manifest_path)
    check = MF.verify(rows, data_root)
    if not check["ok"]:
        raise RuntimeError(f"manifest/data validation failed: {check['problems']}")
    expected_n = int(cfg["split"]["expected_rows"])
    expected_digest = cfg["split"]["expected_manifest_digest"]
    if len(rows) != expected_n:
        raise RuntimeError(f"expected {expected_n} rows, found {len(rows)}")
    if check["digest"] != expected_digest:
        raise RuntimeError(
            f"manifest digest mismatch: expected {expected_digest}, got {check['digest']}"
        )
    for index, row in enumerate(rows):
        if not row.get("query", "").strip():
            raise RuntimeError(f"empty query at manifest row {index + 2}")
        if os.path.isabs(row["image_path"]):
            raise RuntimeError(f"image_path must be relative to data_root: {row['image_path']}")
        image_hash = row.get("img_sha256", "")
        if len(image_hash) != 64 or any(ch not in "0123456789abcdef" for ch in image_hash.lower()):
            raise RuntimeError(f"missing/invalid img_sha256 at manifest row {index + 2}")
        if row.get("cell") != f"{row['scenario']}|{row['img_type']}":
            raise RuntimeError(f"inconsistent cell at manifest row {index + 2}")
    return rows, check["digest"]


def checkpoint_revision_hint(model_path: str, hf_id: str, models_root: str) -> str | None:
    cache_id = "models--" + hf_id.replace("/", "--")
    candidates = [
        os.path.join(model_path, "refs", "main"),
        os.path.join(models_root, cache_id, "refs", "main"),
        os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub", cache_id, "refs", "main")),
    ]
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                value = f.read().strip()
            if value:
                return value
        except OSError:
            pass
    return None


def prepare_internvl_image(image: Image.Image, cfg: dict, torch):
    """Official-style InternVL dynamic tiling with fixed config values."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    options = cfg["internvl_preprocess"]
    image_size = int(options["image_size"])
    min_num = int(options["min_num"])
    max_num = int(options["max_num"])
    if min_num < 1 or max_num < min_num:
        raise ValueError("invalid InternVL min_num/max_num")

    aspect = image.width / image.height
    ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    candidates = sorted(ratios, key=lambda pair: pair[0] * pair[1])
    best_diff = float("inf")
    target = (1, 1)
    image_area = image.width * image.height
    for ratio in candidates:
        ratio_diff = abs(aspect - ratio[0] / ratio[1])
        if ratio_diff < best_diff:
            best_diff = ratio_diff
            target = ratio
        elif ratio_diff == best_diff and image_area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            target = ratio
    cols, rows = target
    resized = image.resize((image_size * cols, image_size * rows), Image.Resampling.BICUBIC)
    blocks = []
    for idx in range(cols * rows):
        left = (idx % cols) * image_size
        top = (idx // cols) * image_size
        blocks.append(resized.crop((left, top, left + image_size, top + image_size)))
    if options.get("use_thumbnail", True) and len(blocks) > 1:
        blocks.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return torch.stack([transform(block) for block in blocks]), len(blocks)


def load_adapter(model_key: str, model_path: str, cfg: dict, torch):
    spec = MD.get(model_key)
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(f"model config.json not found: {model_path}")

    if spec.family == MD.FAMILY_QWEN:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()},
            local_files_only=True,
        ).eval()
        return {"processor": processor, "model": model}

    if spec.family == MD.FAMILY_LLAVA:
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()},
            local_files_only=True,
        ).eval()
        return {"processor": processor, "model": model}

    if spec.family == MD.FAMILY_INTERNVL:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().to(f"cuda:{torch.cuda.current_device()}")
        return {"tokenizer": tokenizer, "model": model}

    raise ValueError(f"unsupported model family: {spec.family}")


def generate_one(model_key: str, adapter: dict, image: Image.Image, query: str, cfg: dict, torch):
    spec = MD.get(model_key)
    model = adapter["model"]
    device = f"cuda:{torch.cuda.current_device()}"
    max_new_tokens = int(cfg["decode"]["max_new_tokens"])

    if spec.family == MD.FAMILY_QWEN:
        processor = adapter["processor"]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query},
        ]}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(device)
        prompt_len = inputs.input_ids.shape[-1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        answer_ids = output[0][prompt_len:]
        return processor.decode(answer_ids, skip_special_tokens=True), None

    if spec.family == MD.FAMILY_LLAVA:
        processor = adapter["processor"]
        messages = [{"role": "user", "content": [
            {"type": "text", "text": query},
            {"type": "image"},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        answer_ids = output[0][prompt_len:]
        return processor.decode(answer_ids, skip_special_tokens=True), None

    if spec.family == MD.FAMILY_INTERNVL:
        pixels, patch_count = prepare_internvl_image(image, cfg, torch)
        pixels = pixels.to(device=device, dtype=torch.bfloat16)
        question = "<image>\n" + query
        generation = {"max_new_tokens": max_new_tokens, "do_sample": False}
        with torch.inference_mode():
            answer = model.chat(adapter["tokenizer"], pixels, question, generation)
        return str(answer), patch_count

    raise ValueError(f"unsupported model family: {spec.family}")


def load_completed_ids(output_path: str, model_key: str) -> set[str]:
    done: set[str] = set()
    if not os.path.isfile(output_path):
        return done
    with open(output_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {output_path}:{line_no}") from exc
            if record.get("model_key") != model_key:
                raise RuntimeError(f"output contains another model at line {line_no}")
            sid = record.get("sample_id")
            if not sid or sid in done:
                raise RuntimeError(f"missing/duplicate sample_id in output line {line_no}")
            done.add(sid)
    return done


def save_or_check_metadata(path: str, metadata: dict, resume: bool) -> None:
    if os.path.exists(path):
        if not resume:
            raise FileExistsError(f"metadata exists; pass --resume to continue: {path}")
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        stable_keys = (
            "model_key", "hf_id", "model_revision_hint", "model_path", "model_config_sha256",
            "model_tree_sha256", "manifest_digest", "manifest_rows", "shard_count",
            "shard_index", "shard_rows", "device_index", "decode", "internvl_preprocess",
            "software", "script_sha256",
        )
        changes = [k for k in stable_keys if existing.get(k) != metadata.get(k)]
        if changes:
            raise RuntimeError(f"resume metadata differs for fields: {changes}")
        return
    if resume and os.path.exists(path):
        raise RuntimeError("resume requested but metadata could not be read")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval_baseline.json")
    ap.add_argument("--model-key", required=True, choices=MD.MODEL_ORDER)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--models-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", help="continue a partial JSONL output")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="number of disjoint manifest shards for data-parallel inference")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="zero-based manifest shard index")
    ap.add_argument("--device-index", type=int, default=0,
                    help="CUDA device index visible to this worker")
    args = ap.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards == 1 and args.shard_index != 0:
        raise ValueError("--shard-index must be 0 when --num-shards is 1")
    if args.device_index < 0:
        raise ValueError("--device-index must be non-negative")
    if args.out and args.num_shards > 1:
        raise ValueError("do not set --out for sharded runs; each shard uses a distinct output path")

    cfg_path = config_path(args.config)
    cfg = load_config(cfg_path)
    manifest_path = config_path(args.manifest or cfg["split"]["manifest"])
    data_root = expand_path(args.data_root or os.environ.get("DATA_ROOT") or cfg["paths"]["data_root"])
    models_root = expand_path(args.models_root or os.environ.get("MODELS_ROOT") or cfg["paths"]["models_root"])
    out_dir = expand_path(cfg["paths"]["output_dir"])
    if args.out:
        output_path = config_path(args.out)
    elif args.num_shards == 1:
        output_path = os.path.join(out_dir, args.model_key + ".jsonl")
    else:
        output_path = os.path.join(
            out_dir,
            f"{args.model_key}.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl",
        )
    metadata_path = output_path + ".meta.json"

    all_rows, digest = validate_inputs(cfg, args.model_key, manifest_path, data_root)
    rows = [row for index, row in enumerate(all_rows) if index % args.num_shards == args.shard_index]
    if not rows:
        raise RuntimeError(f"shard {args.shard_index}/{args.num_shards} is empty")
    spec = MD.get(args.model_key)
    model_path = os.path.join(models_root, spec.hf_id.split("/")[-1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if (os.path.exists(output_path) or os.path.exists(metadata_path)) and not args.resume:
        raise FileExistsError(f"output or metadata exists; pass --resume to continue: {output_path}")
    if args.resume and os.path.isfile(output_path) and not os.path.isfile(metadata_path):
        raise RuntimeError("cannot resume output without its metadata sidecar")
    done = load_completed_ids(output_path, args.model_key)
    known_ids = {r["sample_id"] for r in rows}
    unknown = done - known_ids
    if unknown:
        raise RuntimeError(f"output contains IDs absent from manifest: {sorted(unknown)[:5]}")

    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script inside a GPU job")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no visible CUDA device")
    if args.device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {args.device_index} requested, but only "
            f"{torch.cuda.device_count()} device(s) are visible"
        )
    torch.cuda.set_device(args.device_index)

    config_json = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_json):
        raise FileNotFoundError(f"model config.json not found: {config_json}")
    revision_hint = checkpoint_revision_hint(model_path, spec.hf_id, models_root)
    model_tree_digest, model_files = fingerprint_model_tree(model_path)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": args.model_key,
        "model_name": spec.name,
        "hf_id": spec.hf_id,
        "model_revision_hint": revision_hint,
        "model_path": os.path.abspath(model_path),
        "model_config_sha256": sha256_file(config_json),
        "model_tree_sha256": model_tree_digest,
        "model_files": model_files,
        "manifest_path": os.path.abspath(manifest_path),
        "data_root": os.path.abspath(data_root),
        "manifest_digest": digest,
        "manifest_rows": len(all_rows),
        "shard_count": args.num_shards,
        "shard_index": args.shard_index,
        "shard_rows": len(rows),
        "device_index": args.device_index,
        "decode": cfg["decode"],
        "internvl_preprocess": cfg["internvl_preprocess"] if spec.family == MD.FAMILY_INTERNVL else None,
        "judge_status": cfg["judge"]["status"],
        "judge_method": cfg["judge"]["method"],
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device_index),
        },
        "script_sha256": sha256_file(os.path.abspath(__file__)),
    }

    save_or_check_metadata(metadata_path, metadata, args.resume)

    print(f"模型: {spec.name} ({args.model_key})")
    print(f"权重目录: {model_path}")
    print(f"revision hint: {revision_hint or 'not found; see the full checkpoint tree hash below'}")
    print(f"checkpoint tree sha256: {model_tree_digest}")
    print(f"manifest: {len(all_rows)} rows, digest={digest}")
    print(f"shard: {args.shard_index + 1}/{args.num_shards}, {len(rows)} rows, cuda:{args.device_index}")
    print(f"resume: {len(done)} already generated, {len(rows) - len(done)} remaining")
    print(f"output: {output_path}")

    adapter = load_adapter(args.model_key, model_path, cfg, torch)
    start = time.monotonic()
    wrote = 0
    with open(output_path, "a", encoding="utf-8") as out:
        for row in rows:
            sid = row["sample_id"]
            if sid in done:
                continue
            image_path = os.path.join(data_root, row["image_path"])
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            sample_start = time.monotonic()
            answer, patch_count = generate_one(
                args.model_key, adapter, image, row["query"], cfg, torch
            )
            elapsed = time.monotonic() - sample_start
            record = {
                "sample_id": sid,
                "model_key": args.model_key,
                "model_name": spec.name,
                "state": "orig",
                "scenario": row["scenario"],
                "task": row["task"],
                "qidx": int(row["qidx"]),
                "img_type": row["img_type"],
                "cell": row["cell"],
                "query": row["query"],
                "image_path": row["image_path"],
                "img_sha256": row["img_sha256"],
                "response": answer,
                "generation_seconds": round(elapsed, 4),
                "internvl_patch_count": patch_count,
                "judge_status": "pending",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            done.add(sid)
            wrote += 1
            if wrote % 25 == 0 or len(done) == len(rows):
                elapsed_all = time.monotonic() - start
                print(f"progress {len(done)}/{len(rows)}; latest={sid}; "
                      f"sample={elapsed:.2f}s; session={elapsed_all / 60:.1f}min", flush=True)
            if wrote % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    if done != known_ids:
        missing = sorted(known_ids - done)
        raise RuntimeError(f"incomplete generation: {len(missing)} missing; first={missing[:5]}")
    summary = {
        "model_key": args.model_key,
        "rows": len(done),
        "manifest_digest": digest,
        "shard_count": args.num_shards,
        "shard_index": args.shard_index,
        "generation_complete": True,
        "judge_complete": False,
        "output": os.path.abspath(output_path),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path + ".complete.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("[完成] raw generations complete; ASR still requires the separate judge step")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
