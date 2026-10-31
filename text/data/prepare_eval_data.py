#!/usr/bin/env python3
"""Rebuild the pinned WMDP C and frozen S bundles from published inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

REPOSITORY = "cais/wmdp"
REVISION = "7125571f22f032c56415e7980f48d877dd830ff8"
SREV1_PACKAGE_MANIFEST_SHA256 = "b89785624ffbaf9e427851704715e481508f8a2e1d3cec42cc7c3a9c9a77788c"
DEFAULT_DATASET_VERSION = "srev1_partial147_20261031_v1"
HISTORICAL_DATASET_VERSION = "pkg3_s2_v1"
SPLIT = "test"
DOMAINS = (
    ("bio", "wmdp-bio", 1273, 255),
    ("cyber", "wmdp-cyber", 1987, 397),
    ("chem", "wmdp-chem", 408, 82),
)
EXPECTED_SHA256 = {
    "wmdp-bio/test-00000-of-00001.parquet": "738cee28e11e4dcf9a0efc6b2c22b425bddda4b2b69477a11f38539c86cf00b7",
    "wmdp-cyber/test-00000-of-00001.parquet": "ec65f7ba4a9cd1ef368618ebd2b90b6d930f9b9f053f67a926b82fd71d87b4fe",
    "wmdp-chem/test-00000-of-00001.parquet": "b9e5f2be6c7526931ae8aeb2eea246baa4144d2ba8d0de1de09905aaa84912ce",
}
SCENE_FILES = {
    "original": "s_original.jsonl",
    "syntax": "s_syntax.jsonl",
    "paraphrase": "s_paraphrase.jsonl",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ValueError(f"truncated JSONL line at {path.name}:{line_number}")
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> str:
    h = hashlib.sha256()
    with path.open("xb") as stream:
        for row in rows:
            line = canonical(row) + b"\n"
            stream.write(line)
            h.update(line)
    return h.hexdigest()


def rows_sha256(rows: list[dict]) -> str:
    """Hash the exact canonical JSONL bytes before writing reconstructed output."""
    h = hashlib.sha256()
    for row in rows:
        h.update(canonical(row) + b"\n")
    return h.hexdigest()


def verify_source_manifest(path: Path, source_files: dict[str, Path], c_rows: list[dict],
                           scenes: dict[str, list[dict]], selection_path: Path,
                           syntax_path: Path, paraphrase_path: Path, frozen_meta_path: Path,
                           revision_selection_path: Path | None) -> str:
    """Bind a minimal release's source files and reconstructed JSONL to its pinned hashes."""
    expected = json.loads(path.read_text(encoding="utf-8"))
    if (expected.get("repository") != REPOSITORY or expected.get("revision") != REVISION
            or expected.get("license") != "MIT"):
        raise ValueError("expected source manifest repository/revision/license mismatch")
    if expected.get("dataset_version") not in (None, DEFAULT_DATASET_VERSION, HISTORICAL_DATASET_VERSION):
        raise ValueError("expected source manifest dataset version mismatch")
    actual_sources = {
        name: {"sha256": sha256_file(source_files[name]), "bytes": source_files[name].stat().st_size}
        for name in EXPECTED_SHA256
    }
    if expected.get("source_files") != actual_sources:
        raise ValueError("downloaded/local source files differ from the pinned source manifest")
    actual = {
        "C_rows": len(c_rows),
        "S_rows_per_scene": 734,
        "C_jsonl_sha256": rows_sha256(c_rows),
        "S_jsonl_sha256": {scene: rows_sha256(rows) for scene, rows in scenes.items()},
        "S_selection_sha256": sha256_file(selection_path),
        "S_syntax_patch_sha256": sha256_file(syntax_path),
        "S_paraphrase_patch_sha256": sha256_file(paraphrase_path),
        "S_frozen_manifest_sha256": sha256_file(frozen_meta_path),
    }
    if revision_selection_path is not None:
        actual["S_revision_selection_sha256"] = sha256_file(revision_selection_path)
    expected_reconstruction = expected.get("reconstruction", {})
    for key, value in actual.items():
        if expected_reconstruction.get(key) != value:
            raise ValueError(f"reconstructed output differs from pinned source manifest: {key}")
    return sha256_file(path)


def fetch_source_files(cache_dir: Path) -> dict[str, Path]:
    """Download only the three pinned public Parquet files into the Hub cache."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("--download requires huggingface_hub; install the model environment requirements") from exc
    found = {}
    for relative in EXPECTED_SHA256:
        found[relative] = Path(hf_hub_download(
            repo_id=REPOSITORY,
            repo_type="dataset",
            revision=REVISION,
            filename=relative,
            cache_dir=str(cache_dir),
        ))
    return found


def local_source_files(cache_dir: Path) -> dict[str, Path]:
    return {relative: cache_dir / relative for relative in EXPECTED_SHA256}


def read_upstream_rows(source_files: dict[str, Path]) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet reconstruction requires pyarrow") from exc

    output = []
    for domain, config, expected_rows, _sample_size in DOMAINS:
        relative = f"{config}/{SPLIT}-00000-of-00001.parquet"
        path = source_files[relative]
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned source: {relative}")
        actual_sha = sha256_file(path)
        if actual_sha != EXPECTED_SHA256[relative]:
            raise ValueError(f"fixed-revision source hash mismatch: {relative}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != expected_rows:
            raise ValueError(f"fixed-revision row count mismatch: {relative}")
        rows = parquet.read().to_pylist()
        if len(rows) != expected_rows:
            raise ValueError(f"decoded row count mismatch: {relative}")
        for index, source_row in enumerate(rows):
            question = source_row.get("question")
            choices = source_row.get("choices")
            answer = source_row.get("answer")
            if (not isinstance(question, str) or not question.strip()
                    or not isinstance(choices, (list, tuple)) or len(choices) != 4
                    or any(not isinstance(item, str) or not item.strip() for item in choices)
                    or type(answer) is not int or answer not in range(4)):
                raise ValueError(f"invalid four-choice row at {domain}:{index}")
            stable_id = source_row.get("id", source_row.get("question_id"))
            if stable_id is None:
                stable_id = f"{config}/{SPLIT}/{index:04d}"
            output.append({
                "question_id": str(stable_id),
                "source_row_sha256": sha256_bytes(canonical(source_row)),
                "domain": domain,
                "question": question,
                "choices": list(choices),
                "answer_index": answer,
            })
    if len(output) != 3668:
        raise ValueError("complete WMDP C reconstruction must contain 3668 rows")
    return output


def assemble_s(c_rows: list[dict], selection_path: Path, syntax_path: Path,
               paraphrase_path: Path) -> dict[str, list[dict]]:
    selection = read_jsonl(selection_path)
    syntax = read_jsonl(syntax_path)
    paraphrase = read_jsonl(paraphrase_path)
    if len(selection) != 734 or len(syntax) != 734 or len(paraphrase) != 734:
        raise ValueError("frozen S input must contain one 734-row selection and two 734-row variants")
    c_by_id = {row["question_id"]: row for row in c_rows}
    selected = {}
    for row in selection:
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid or qid in selected or qid not in c_by_id:
            raise ValueError("empty, duplicate, or unknown S selection ID")
        source = c_by_id[qid]
        if any(row.get(key) != source[key] for key in ("source_row_sha256", "domain")):
            raise ValueError("selection source hash/domain differs from the pinned C row")
        selected[qid] = source
    domain_counts = {domain: sum(row["domain"] == domain for row in selected.values())
                     for domain in ("bio", "cyber", "chem")}
    if domain_counts != {"bio": 255, "cyber": 397, "chem": 82}:
        raise ValueError("fixed S selection must retain the 255/397/82 domain allocation")

    scenes = {"original": [], "syntax": [], "paraphrase": []}
    for pick in selection:
        source = selected[pick["question_id"]]
        scenes["original"].append(dict(source))
    for name, variant_rows, expected_type in (
            ("syntax", syntax, "syntax_adjustment"),
            ("paraphrase", paraphrase, "question_restatement")):
        by_id = {}
        for variant in variant_rows:
            qid = variant.get("question_id")
            text = variant.get("question")
            if (not isinstance(qid, str) or qid not in selected or qid in by_id
                    or not isinstance(text, str) or not text.strip()
                    or variant.get("variant_type") != expected_type):
                raise ValueError(f"invalid or duplicate {name} variant row")
            source = selected[qid]
            content_sha = sha256_bytes(text.encode("utf-8"))
            if (variant.get("source_row_sha256") != source["source_row_sha256"]
                    or variant.get("domain") != source["domain"]
                    or variant.get("question_sha256") != content_sha):
                raise ValueError(f"{name} variant source/content binding mismatch")
            by_id[qid] = text
        if set(by_id) != set(selected):
            raise ValueError(f"{name} variants do not cover the exact frozen sample")
        for pick in selection:
            source = selected[pick["question_id"]]
            row = dict(source)
            row["question"] = by_id[pick["question_id"]]
            scenes[name].append(row)
    if any(len(rows) != 734 for rows in scenes.values()):
        raise ValueError("S scenario must contain exactly 734 rows")
    return scenes


def validate_partial147_inputs(freeze_root: Path, freeze_dir: Path, frozen_meta: dict,
                               base_selection_path: Path, revision_selection_path: Path,
                               syntax_path: Path, paraphrase_path: Path) -> None:
    if frozen_meta.get("schema") != "pkg3-srev1-partial147-frozen-inputs-v1":
        raise ValueError("unsupported partial147 frozen manifest schema")
    if frozen_meta.get("dataset_version") != DEFAULT_DATASET_VERSION:
        raise ValueError("partial147 dataset version mismatch")
    if frozen_meta.get("base_selection", {}).get("sha256") != sha256_file(base_selection_path):
        raise ValueError("partial147 parent selection hash mismatch")
    if frozen_meta.get("revision_selection", {}).get("sha256") != sha256_file(revision_selection_path):
        raise ValueError("partial147 selection hash mismatch")
    if frozen_meta.get("patches", {}).get("syntax", {}).get("sha256") != sha256_file(syntax_path):
        raise ValueError("partial147 syntax patch hash mismatch")
    if frozen_meta.get("patches", {}).get("paraphrase", {}).get("sha256") != sha256_file(paraphrase_path):
        raise ValueError("partial147 paraphrase patch hash mismatch")
    base_rows = read_jsonl(base_selection_path)
    selected_rows = read_jsonl(revision_selection_path)
    if len(base_rows) != 734 or len(selected_rows) != 147:
        raise ValueError("partial147 parent/revision selection row count mismatch")
    base_by_id = {row.get("question_id"): row for row in base_rows}
    selected_by_id = {row.get("question_id"): row for row in selected_rows}
    if (None in base_by_id or None in selected_by_id or len(base_by_id) != 734
            or len(selected_by_id) != 147 or not set(selected_by_id) <= set(base_by_id)):
        raise ValueError("partial147 selection has duplicate or unknown IDs")
    for qid, row in selected_by_id.items():
        parent = base_by_id[qid]
        if any(row.get(field) != parent.get(field) for field in ("domain", "source_row_sha256")):
            raise ValueError("partial147 selection changed parent source binding")
    domains = {domain: sum(row.get("domain") == domain for row in selected_rows)
               for domain in ("bio", "cyber", "chem")}
    if domains != {"bio": 51, "cyber": 80, "chem": 16}:
        raise ValueError("partial147 selection domain counts mismatch")

    old_and_new = (
        (freeze_root / "s2_syntax_questions.jsonl", syntax_path),
        (freeze_root / "s2_paraphrase_questions.jsonl", paraphrase_path),
    )
    for old_path, new_path in old_and_new:
        old_lines = old_path.read_bytes().splitlines(keepends=True)
        new_lines = new_path.read_bytes().splitlines(keepends=True)
        if len(old_lines) != 734 or len(new_lines) != 734:
            raise ValueError("partial147 variant patches must each contain 734 JSONL rows")
        old_by = {json.loads(line)["question_id"]: line for line in old_lines}
        new_by = {json.loads(line)["question_id"]: line for line in new_lines}
        if set(old_by) != set(new_by) or len(new_by) != 734:
            raise ValueError("partial147 variant IDs/order coverage differ from historical patch")
        changed = {qid for qid in old_by if old_by[qid] != new_by[qid]}
        unchanged = set(old_by) - changed
        if changed != set(selected_by_id) or len(unchanged) != 587:
            raise ValueError("partial147 patches must replace exactly selected 147 rows")
        for qid in unchanged:
            if old_by[qid] != new_by[qid]:
                raise ValueError("an unselected historical variant line changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-cache", type=Path,
                        help="offline root containing wmdp-{bio,cyber,chem}/test-00000-of-00001.parquet")
    source.add_argument("--download", action="store_true",
                        help="download those exact files from cais/wmdp at the pinned revision")
    parser.add_argument("--hub-cache", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--dataset-version", choices=(DEFAULT_DATASET_VERSION, HISTORICAL_DATASET_VERSION),
                        default=DEFAULT_DATASET_VERSION,
                        help="S dataset; defaults to the revised 147-item variant set")
    parser.add_argument("--freeze-root", type=Path, default=Path(__file__).resolve().parent / "frozen",
                        help="root holding the historical PKG3 selection and versioned S freezes")
    parser.add_argument("--freeze-dir", type=Path,
                        help="optional direct override for the chosen version's frozen input directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for the rebuilt C and S bundles; it must not already exist")
    parser.add_argument("--expected-c-jsonl", type=Path,
                        help="optional local row-for-row comparison against an existing frozen C input")
    parser.add_argument("--expected-s-dir", type=Path,
                        help="optional local comparison against s_original.jsonl/s_syntax.jsonl/s_paraphrase.jsonl")
    parser.add_argument("--expected-source-manifest", type=Path,
                        help="compare source and reconstructed C/S hashes with this compact manifest; defaults to freeze-dir/prepared/source_manifest.json when present")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    source_files = fetch_source_files(args.hub_cache) if args.download else local_source_files(args.source_cache)
    base_selection_path = args.freeze_root / "s2_selection_734.jsonl"
    if args.dataset_version == DEFAULT_DATASET_VERSION:
        freeze_dir = args.freeze_dir or args.freeze_root / "srev1_partial147_v1"
        selection_path = base_selection_path
        revision_selection_path = freeze_dir / "selection_147.jsonl"
        syntax_path = freeze_dir / "syntax_v2_questions_734.jsonl"
        paraphrase_path = freeze_dir / "paraphrase_v2_questions_734.jsonl"
        frozen_meta_path = freeze_dir / "manifest.json"
    else:
        freeze_dir = args.freeze_dir or args.freeze_root
        selection_path = base_selection_path
        revision_selection_path = None
        syntax_path = freeze_dir / "s2_syntax_questions.jsonl"
        paraphrase_path = freeze_dir / "s2_paraphrase_questions.jsonl"
        frozen_meta_path = freeze_dir / "s2_frozen_manifest.json"
    frozen_meta = json.loads(frozen_meta_path.read_text(encoding="utf-8"))
    if args.dataset_version == DEFAULT_DATASET_VERSION:
        validate_partial147_inputs(args.freeze_root, freeze_dir, frozen_meta, base_selection_path,
                                   revision_selection_path, syntax_path, paraphrase_path)
    elif (frozen_meta.get("schema") != "pkg3-s2-frozen-inputs-v1"
          or frozen_meta.get("selection_sha256") != sha256_file(selection_path)
          or frozen_meta.get("variant_patch_sha256", {}).get(syntax_path.name) != sha256_file(syntax_path)
          or frozen_meta.get("variant_patch_sha256", {}).get(paraphrase_path.name) != sha256_file(paraphrase_path)):
        raise ValueError("historical S selection/patch identity differs from its manifest")
    c_rows = read_upstream_rows(source_files)
    scenes = assemble_s(c_rows, selection_path, syntax_path, paraphrase_path)
    expected_manifest = args.expected_source_manifest
    if expected_manifest is None:
        candidate = (freeze_dir / "source_manifest.json" if args.dataset_version == DEFAULT_DATASET_VERSION
                     else freeze_dir / "prepared" / "source_manifest.json")
        if candidate.is_file():
            expected_manifest = candidate
    expected_manifest_sha = None
    if expected_manifest is not None:
        if not expected_manifest.is_file():
            raise FileNotFoundError(f"expected source manifest missing: {expected_manifest}")
        expected_manifest_sha = verify_source_manifest(
            expected_manifest, source_files, c_rows, scenes, selection_path,
            syntax_path, paraphrase_path, frozen_meta_path, revision_selection_path)
    if args.expected_c_jsonl:
        expected_c = read_jsonl(args.expected_c_jsonl)
        fields = ("question_id", "source_row_sha256", "domain", "question", "choices", "answer_index")
        actual = [{key: row[key] for key in fields} for row in c_rows]
        expected = [{key: row[key] for key in fields} for row in expected_c]
        if actual != expected:
            raise ValueError("reconstructed C differs from the existing frozen input by field or order")
    if args.expected_s_dir:
        fields = ("question_id", "source_row_sha256", "domain", "question", "choices", "answer_index")
        for scene, filename in SCENE_FILES.items():
            expected = read_jsonl(args.expected_s_dir / filename)
            actual = [{key: row[key] for key in fields} for row in scenes[scene]]
            expected = [{key: row[key] for key in fields} for row in expected]
            if actual != expected:
                raise ValueError(f"reconstructed {scene} differs from the existing frozen S rows")

    out = args.output_dir
    (out / "C").mkdir(parents=True)
    (out / "S").mkdir(parents=True)
    c_path = out / "C" / "original.jsonl"
    c_digest = write_jsonl(c_path, c_rows)
    write_json(out / "C" / "manifest.json", {
        "schema": "text-eval-v1", "kind": "C",
        "source": f"{REPOSITORY}@{REVISION}; official Parquet SHA256 pinned in source_manifest.json",
        "dataset_version": args.dataset_version,
        "scenes": {"original": "original.jsonl"},
        "source_manifest": "../source_manifest.json",
    })
    s_files = {}
    s_digests = {}
    for scene, filename in SCENE_FILES.items():
        path = out / "S" / filename
        s_digests[scene] = write_jsonl(path, scenes[scene])
        s_files[scene] = filename
    write_json(out / "S" / "manifest.json", {
        "schema": "text-eval-v1", "kind": "S",
        "source": (f"{REPOSITORY}@{REVISION}; dataset_version={DEFAULT_DATASET_VERSION}; "
                   f"srev1_package_manifest_sha256={SREV1_PACKAGE_MANIFEST_SHA256}"
                   if args.dataset_version == DEFAULT_DATASET_VERSION else
                   f"{REPOSITORY}@{REVISION}; frozen S2 selection and variants listed in the package data README"),
        "dataset_version": args.dataset_version,
        "scenes": s_files,
        "frozen_identity": {
            "sample_selection_sha256": (frozen_meta["base_selection"]["sha256"]
                                         if revision_selection_path is not None else
                                         frozen_meta["selection_sha256"]),
            "revision_selection_sha256": (None if revision_selection_path is None
                                           else sha256_file(revision_selection_path)),
            "syntax_patch_sha256": (frozen_meta["patches"]["syntax"]["sha256"]
                                     if revision_selection_path is not None else
                                     frozen_meta["variant_patch_sha256"][syntax_path.name]),
            "paraphrase_patch_sha256": (frozen_meta["patches"]["paraphrase"]["sha256"]
                                         if revision_selection_path is not None else
                                         frozen_meta["variant_patch_sha256"][paraphrase_path.name]),
            "frozen_manifest_sha256": sha256_file(frozen_meta_path),
            "scene_input_sha256": {scene: rows_sha256(rows) for scene, rows in scenes.items()},
        },
    })
    comparison = "not requested"
    if expected_manifest_sha:
        comparison = "passed against pinned source and reconstructed C/S JSONL hashes"
    elif args.expected_c_jsonl or args.expected_s_dir:
        comparison = "passed by requested field-and-order comparison"
    manifest = {
        "repository": REPOSITORY, "revision": REVISION, "license": "MIT",
        "source_files": {name: {"sha256": EXPECTED_SHA256[name], "bytes": source_files[name].stat().st_size}
                         for name in EXPECTED_SHA256},
        "reconstruction": {
            "C_rows": len(c_rows), "S_rows_per_scene": 734,
            "C_jsonl_sha256": c_digest, "S_jsonl_sha256": s_digests,
            "S_selection_sha256": sha256_file(selection_path),
            "S_revision_selection_sha256": (None if revision_selection_path is None
                                             else sha256_file(revision_selection_path)),
            "S_syntax_patch_sha256": sha256_file(syntax_path),
            "S_paraphrase_patch_sha256": sha256_file(paraphrase_path),
            "S_frozen_manifest_sha256": sha256_file(frozen_meta_path),
            "field_and_order_comparison": comparison,
            "expected_source_manifest_sha256": expected_manifest_sha,
        },
    }
    write_json(out / "source_manifest.json", manifest)
    print(json.dumps({"status": "passed", "C_rows": len(c_rows),
                      "S_rows_per_scene": 734, "field_and_order_comparison": manifest["reconstruction"]["field_and_order_comparison"],
                      "expected_source_manifest_sha256": expected_manifest_sha,
                      "C_sha256": c_digest, "S_sha256": s_digests}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
