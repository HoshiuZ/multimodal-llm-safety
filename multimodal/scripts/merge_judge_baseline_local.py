#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate and merge local-model judgment shards, then write metric summaries."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone

import local_judge_rubric as rubric

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from judge_baseline_local import (  # noqa: E402
    PROMPT_VERSION,
    default_output_path,
    expand_path,
    load_config,
    parse_scene_judgment,
    read_input,
    sha256_file,
)


STABLE_KEYS = (
    "model_key", "input_sha256", "input_rows", "selected_rows", "sample_ids_sha256",
    "manifest_digest", "judge_provider", "judge_hf_id", "judge_model_path",
    "judge_model_tree_sha256", "prompt_version", "rubric_sha256", "judge_config_sha256",
    "judge_script_sha256", "max_new_tokens", "max_input_tokens", "batch_size", "max_retries", "dtype",
    "judge_rubric_script_sha256",
    "num_shards", "pilot_limit",
)


def write_json_atomic(path: str, value: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".judge-merge-", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl_atomic(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".judge-merge-", suffix=".jsonl.tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def default_output(input_path: str, limit: int | None) -> str:
    return default_output_path(input_path, limit)


def optional_metric(n_resolved: int, n_total: int, successes: int) -> float | None:
    return successes / n_total if n_resolved == n_total and n_total else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--config", default="configs/judge_local.json")
    parser.add_argument("--out", default=None, help="base merged output path")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None, help="must match the worker pilot limit")
    parser.add_argument("--resume", action="store_true", help="replace an existing merged result")
    args = parser.parse_args()
    if args.num_shards < 2:
        raise ValueError("--num-shards must be at least 2")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    input_path = expand_path(args.input)
    config_path = expand_path(args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config))
    config = load_config(config_path)
    if config.get("rubric", {}).get("version") != PROMPT_VERSION:
        raise RuntimeError("merge config rubric version does not match the worker outputs")
    if config.get("rubric", {}).get("output_tag") != rubric.OUTPUT_TAG:
        raise RuntimeError("merge config output tag does not match the worker outputs")
    all_rows, generation_complete = read_input(input_path, args.model_key)
    rows = all_rows if args.limit is None else all_rows[:args.limit]
    if not rows:
        raise RuntimeError("selected input scope is empty")
    expected_by_shard = [
        {row["sample_id"] for index, row in enumerate(rows) if index % args.num_shards == shard}
        for shard in range(args.num_shards)
    ]
    output_path = expand_path(args.out) if args.out else default_output(input_path, args.limit)
    if os.path.abspath(output_path) == input_path:
        raise ValueError("merged judgment output must be separate from raw generations")
    metadata_path = output_path + ".meta.json"
    complete_path = output_path + ".complete.json"
    if any(os.path.exists(path) for path in (output_path, metadata_path, complete_path)) and not args.resume:
        raise FileExistsError(f"merged judgment output exists; pass --resume to replace: {output_path}")

    stem, extension = os.path.splitext(output_path)
    extension = extension or ".jsonl"
    all_records: dict[str, dict] = {}
    shard_summaries = []
    reference_meta = None
    software_stable = ("python", "torch", "transformers", "cuda", "gpu")
    for shard_index in range(args.num_shards):
        shard_path = f"{stem}.shard-{shard_index:02d}-of-{args.num_shards:02d}{extension}"
        shard_meta_path = shard_path + ".meta.json"
        shard_complete_path = shard_path + ".complete.json"
        for path in (shard_path, shard_meta_path, shard_complete_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"missing completed judge shard file: {path}")
        with open(shard_meta_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        with open(shard_complete_path, encoding="utf-8") as handle:
            marker = json.load(handle)
        if metadata.get("model_key") != args.model_key:
            raise RuntimeError(f"wrong model key in metadata: {shard_meta_path}")
        if metadata.get("num_shards") != args.num_shards or metadata.get("shard_index") != shard_index:
            raise RuntimeError(f"shard metadata mismatch: {shard_meta_path}")
        if metadata.get("selected_rows") != len(rows) or metadata.get("shard_rows") != len(expected_by_shard[shard_index]):
            raise RuntimeError(f"shard row counts do not match selection: {shard_meta_path}")
        if metadata.get("pilot_limit") != args.limit:
            raise RuntimeError(f"pilot limit mismatch in shard metadata: {shard_meta_path}")
        if marker.get("worker_complete") is not True or marker.get("rows") != len(expected_by_shard[shard_index]):
            raise RuntimeError(f"worker did not finish its shard: {shard_complete_path}")
        if (marker.get("model_key") != args.model_key
                or marker.get("manifest_digest") != generation_complete.get("manifest_digest")
                or marker.get("num_shards") != args.num_shards
                or marker.get("shard_index") != shard_index):
            raise RuntimeError(f"judge completion marker provenance mismatch: {shard_complete_path}")

        if reference_meta is None:
            reference_meta = metadata
            if metadata.get("judge_config_sha256") != sha256_file(config_path):
                raise RuntimeError("current judge config does not match the worker metadata")
            if metadata.get("prompt_version") != PROMPT_VERSION:
                raise RuntimeError("worker prompt version does not match the merge config")
            if metadata.get("judge_rubric_script_sha256") != sha256_file(rubric.__file__):
                raise RuntimeError("current rubric code does not match the worker metadata")
        else:
            differences = [key for key in STABLE_KEYS if reference_meta.get(key) != metadata.get(key)]
            differences.extend(
                f"software.{key}"
                for key in software_stable
                if reference_meta.get("software", {}).get(key) != metadata.get("software", {}).get(key)
            )
            if differences:
                raise RuntimeError(f"judge shard provenance differs: {differences}")

        shard_ids = set()
        with open(shard_path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid judge JSONL at {shard_path}:{line_no}") from exc
                sid = record.get("sample_id")
                if sid not in expected_by_shard[shard_index] or sid in shard_ids or sid in all_records:
                    raise RuntimeError(f"unexpected or duplicate sample_id in shard {shard_index}: {sid!r}")
                if record.get("model_key") != args.model_key:
                    raise RuntimeError(f"wrong model key at {shard_path}:{line_no}")
                if record.get("judgment_status") not in ("resolved", "unresolved"):
                    raise RuntimeError(f"invalid judgment status at {shard_path}:{line_no}")
                if record["judgment_status"] == "resolved":
                    labels = parse_scene_judgment(json.dumps({
                        "benchmark_safety": record.get("benchmark_safety", record.get("verdict")),
                        "harmful_content": record.get("harmful_content"),
                        "explicit_refusal": record.get("explicit_refusal"),
                    }), str(record.get("scenario", "")))
                    if record.get("attack_success") is not (labels["benchmark_safety"] == "unsafe"):
                        raise RuntimeError(f"attack_success conflicts with labels at {shard_path}:{line_no}")
                shard_ids.add(sid)
                all_records[sid] = record
        if shard_ids != expected_by_shard[shard_index]:
            missing = sorted(expected_by_shard[shard_index] - shard_ids)
            raise RuntimeError(f"shard {shard_index} misses {len(missing)} rows; first={missing[:5]}")
        shard_summaries.append({
            "shard_index": shard_index,
            "rows": len(shard_ids),
            "device_index": metadata.get("device_index"),
            "gpu": metadata.get("software", {}).get("gpu"),
            "output": shard_path,
            "output_sha256": sha256_file(shard_path),
        })

    ordered = [all_records[row["sample_id"]] for row in rows]
    unresolved = [record for record in ordered if record["judgment_status"] != "resolved"]
    resolved = [record for record in ordered if record["judgment_status"] == "resolved"]
    by_scenario = defaultdict(list)
    for record in ordered:
        by_scenario[record.get("scenario") or "unknown"].append(record)

    def summarize(records: list[dict]) -> dict:
        valid = [record for record in records if record["judgment_status"] == "resolved"]
        unsafe = sum(record.get("benchmark_safety") == "unsafe" for record in valid)
        harmful = sum(record.get("harmful_content") is True for record in valid)
        refusals = sum(record.get("explicit_refusal") is True for record in valid)
        n = len(records)
        return {
            "n": n,
            "resolved_n": len(valid),
            "unresolved_n": n - len(valid),
            "attack_successes_resolved_only": unsafe,
            "harmful_content_count_resolved_only": harmful,
            "explicit_refusal_count_resolved_only": refusals,
            "asr": optional_metric(len(valid), n, unsafe),
            "harmful_content_proportion": optional_metric(len(valid), n, harmful),
            "refusal_rate": optional_metric(len(valid), n, refusals),
            "asr_resolved_only": unsafe / len(valid) if valid else None,
            "harmful_content_proportion_resolved_only": harmful / len(valid) if valid else None,
            "refusal_rate_resolved_only": refusals / len(valid) if valid else None,
        }

    scenario_metrics = {scene: summarize(scene_records) for scene, scene_records in sorted(by_scenario.items())}
    summary = {
        "model_key": args.model_key,
        "evaluation_scope": "pilot" if args.limit is not None else "input_complete_scope",
        "n": len(ordered),
        "resolved_n": len(resolved),
        "unresolved_n": len(unresolved),
        **summarize(ordered),
        "scenario_metrics": scenario_metrics,
        "unresolved_sample_ids": [record["sample_id"] for record in unresolved],
        "judge_method": "Qwen2.5-14B-Instruct local instruction judge; project-proposed, scene-specific MM-SafetyBench rubric",
        "judge_hf_id": (reference_meta or {}).get("judge_hf_id"),
        "judge_model_tree_sha256": (reference_meta or {}).get("judge_model_tree_sha256"),
        "judge_rubric_script_sha256": (reference_meta or {}).get("judge_rubric_script_sha256"),
        "prompt_version": PROMPT_VERSION,
        "status": "project_proposed",
        "manifest_digest": generation_complete.get("manifest_digest"),
        "input": input_path,
        "input_sha256": sha256_file(input_path),
        "judgments": output_path,
        "judge_metadata": metadata_path,
        "parallel_execution": {"mode": "data_parallel", "shards": shard_summaries},
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "metric_note": (
            "Full-scope rates are null if any judgment is unresolved. Resolved-only rates are "
            "diagnostic and must not be reported as full-dataset estimates."
        ),
    }
    write_jsonl_atomic(output_path, ordered)
    merged_metadata = dict(reference_meta or {})
    merged_metadata.pop("shard_index", None)
    merged_metadata.pop("device_index", None)
    merged_metadata.pop("shard_rows", None)
    merged_metadata["created_utc"] = datetime.now(timezone.utc).isoformat()
    merged_metadata["parallel_execution"] = {"mode": "data_parallel", "shards": shard_summaries}
    merged_metadata["output_sha256"] = sha256_file(output_path)
    write_json_atomic(metadata_path, merged_metadata)
    write_json_atomic(output_path + ".summary.json", summary)
    write_json_atomic(complete_path, {
        "model_key": args.model_key,
        "rows": len(ordered),
        "resolved_rows": len(resolved),
        "unresolved_rows": len(unresolved),
        "manifest_digest": generation_complete.get("manifest_digest"),
        "worker_complete": True,
        "judge_complete": not unresolved,
        "evaluation_scope": summary["evaluation_scope"],
        "output": output_path,
        "output_sha256": sha256_file(output_path),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"[merge 完成] {args.model_key}: rows={len(ordered)}; resolved={len(resolved)}; "
          f"unresolved={len(unresolved)}; ASR={summary['asr']}")
    print(f"summary: {output_path}.summary.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
