#!/usr/bin/env python3
"""Prepare restricted Cyber/Chem RMU snippets using the existing Bio helper.

Selection is CPU-only. Token IDs are produced by invoking this entry separately
inside each model's existing environment. This script does not train or score.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import re
import stat
import sys
import unicodedata
import urllib.parse

import pyarrow.parquet as pq
import torch


ROOT = pathlib.Path(__import__("os").environ.get("RMU_DATA_ROOT", pathlib.Path(__file__).resolve().parents[3])).resolve()
CONFIG = pathlib.Path(__file__).resolve().parents[1] / "configs/three_domain_rmu_data_20261024.json"
RUN_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "configs/bio_rmu_effect.json"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import prepare_bio_rmu_effect_data as bio  # vendored pinned helper

SOURCE_ROOT = pathlib.Path(__import__("os").environ.get("RMU_SOURCE_DIR", pathlib.Path(__import__("os").environ.get("RMU_DATA_ROOT", pathlib.Path(__file__).resolve().parents[3])))).resolve()
MODEL_KEYS = ("qwen3vl8b", "llava_next8b", "internvl3_8b")
CHEM_CATEGORIES = frozenset({
    "Organic Chemistry",
    "Physical and Theoretical Chemistry Textbook Maps",
    "General Chemistry",
    "Introductory Chemistry",
    "Inorganic Chemistry",
    "Analytical Chemistry",
    "Environmental Chemistry",
    "Biological Chemistry",
})
MODELS = {
    "qwen3vl8b": {"use_fast": True, "trust_remote_code": False},
    "llava_next8b": {"use_fast": True, "trust_remote_code": False},
    "internvl3_8b": {"use_fast": False, "trust_remote_code": True},
}
CHEM_EXCLUDED_TITLE = re.compile(
    r"(?i)\b(?:exercises?|practice\s+problems?|problem\s+sets?|review\s+questions?|"
    r"answers?|solutions?|table\s+of\s+contents|contents|navigation|index)\b"
)
EXPLICIT_NONCHEM_TITLE = re.compile(
    r"(?i)\b(?:economics|sociology|political\s+science|world\s+history|literature|"
    r"computer\s+programming)\b"
)
BAD_SECTION = re.compile(
    r"(?i)\b(?:exercises?|practice\s+problems?|problem\s+sets?|review\s+questions?|"
    r"answers?|solutions?|table\s+of\s+contents|contents|navigation)\b"
)
NUMBERED_QUESTION = re.compile(r"^\s*(?:\d+[\.\)]|\([a-z0-9]+\))\s+.+\?\s*$", re.I)
ANSWER_LINE = re.compile(r"^\s*(?:answer|solution)\s*[:.\-]", re.I)
SPACE = re.compile(r"\s+", re.UNICODE)


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def duplicate_key(text: str) -> str:
    return SPACE.sub(" ", normalize_text(text)).strip()


def make_private_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    private_root = ROOT / "data/processed/three_domain_rmu_20261024"
    current = path
    while current != ROOT:
        if current.exists() and current.is_dir() and current.is_relative_to(private_root):
            os.chmod(current, 0o700)
        current = current.parent
        if current == private_root.parent:
            break


def write_private_jsonl(path: pathlib.Path, rows):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    created = []
    bio.write_jsonl(path, rows, created)
    os.chmod(path, 0o600)


def write_private_json(path: pathlib.Path, value):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    created = []
    bio.write_json(path, value, created)
    os.chmod(path, 0o600)


def tokenizer_bundle(run_config):
    bundle = {}
    for key in MODEL_KEYS:
        model_cfg = run_config["models"][key]
        prior_name = "manifest.json" if key == "qwen3vl8b" else "rmu_data_manifest.json"
        prior_path = SOURCE_ROOT / model_cfg["data_dir"] / prior_name
        prior = read_json(prior_path)
        tokenizer, meta = bio.make_tokenizer(key, model_cfg, prior)
        meta["model_key"] = key
        bundle[key] = (tokenizer, meta)
    return bundle


def decode_cyber(path, source_cfg):
    actual = sha256_file(path)
    if path.stat().st_size != source_cfg["size_bytes"] or actual != source_cfg["upstream_lfs_sha256"]:
        raise ValueError("Cyber train shard size or upstream LFS SHA-256 mismatch")
    pf = pq.ParquetFile(path)
    if pf.schema_arrow.names != ["text"]:
        raise ValueError("Cyber train shard schema differs from the pinned source")
    docs = []
    decoded = 0
    rg_counts = []
    empty_text = 0
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=["text"])
        rg_counts.append(table.num_rows)
        vals = table.column("text").to_pylist()
        for offset, value in enumerate(vals):
            row = decoded + offset
            text = normalize_text(value) if isinstance(value, str) else ""
            if not text:
                empty_text += 1
                continue
            row_id = f"cyber-forget-corpus/train-00000-of-00001.parquet:row{row:06d}"
            docs.append({
                "source_record_id": row_id,
                "source_document_id": f"cyber-row:{row:06d}",
                "source_url": None,
                "source_row_index": row,
                "source_row_group": rg,
                "text": text,
                "source_text_sha256": sha256_bytes(text.encode("utf-8")),
                "normalized_duplicate_group_sha256": sha256_bytes(duplicate_key(text).encode("utf-8")),
                "source_text_characters": len(text),
            })
        decoded += table.num_rows
    if decoded != pf.metadata.num_rows:
        raise ValueError("Cyber source decode did not cover every Parquet row")
    return docs, {
        "source_file": source_cfg["file"],
        "local_path": str(path.relative_to(SOURCE_ROOT)),
        "size_bytes": path.stat().st_size,
        "local_sha256": actual,
        "upstream_lfs_sha256": source_cfg["upstream_lfs_sha256"],
        "upstream_lfs_sha256_match": True,
        "metadata_rows": pf.metadata.num_rows,
        "decoded_rows": decoded,
        "row_group_count": pf.metadata.num_row_groups,
        "row_group_rows": rg_counts,
        "schema": [{"name": f.name, "type": str(f.type)} for f in pf.schema_arrow],
        "empty_text_rows": empty_text,
        "all_row_groups_decoded": decoded == pf.metadata.num_rows,
    }


def chemistry_excerpt(text: str, url: str):
    normalized = normalize_text(text)
    parsed = urllib.parse.urlparse(url)
    parts = [urllib.parse.unquote(x) for x in parsed.path.split("/") if x]
    if not parts or parts[0].casefold() != "bookshelves":
        return None, "not_bookshelves", {}
    if len(parts) < 3:
        return None, "bookshelf_path_incomplete", {}
    category, book_title = parts[1].replace("_", " "), parts[2]
    if category not in CHEM_CATEGORIES:
        return None, "unclear_or_nonchem_category", {"category": category}
    title_path = " ".join(part.replace("_", " ") for part in parts[2:])
    if EXPLICIT_NONCHEM_TITLE.search(title_path):
        return None, "explicit_nonchem_title", {"category": category}
    page_title = parts[-1].replace("_", " ")
    if CHEM_EXCLUDED_TITLE.search(page_title):
        return None, "excluded_exercise_answer_or_navigation_title", {"category": category}

    lines = normalized.split("\n")
    boundary = None
    for index, line in enumerate(lines):
        candidate = re.sub(r"^\s*(?:#{1,6}\s*|={2,}\s*)", "", line).strip()
        candidate = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", candidate)
        heading_like = len(candidate) <= 120 and (
            line.lstrip().startswith("#")
            or line.strip().startswith("==")
            or (candidate and not re.search(r"[.!?]\s*$", candidate))
        )
        if heading_like and BAD_SECTION.search(candidate):
            boundary = index
            break
    bounded = "\n".join(lines[:boundary]).strip() if boundary is not None else normalized
    if not bounded:
        return None, "no_expository_text_before_excluded_section", {"category": category}

    # Strong exercise structure is a signal only when no explicit section boundary
    # isolates it. The words "question" or "problem" alone never reject a page.
    blocks = re.split(r"\n\s*\n", bounded)
    numbered_question_lines = sum(
        1 for block in blocks for line in block.splitlines() if NUMBERED_QUESTION.match(line)
    )
    answer_lines = sum(1 for block in blocks for line in block.splitlines() if ANSWER_LINE.match(line))
    if boundary is None and (numbered_question_lines >= 2 or answer_lines >= 2):
        return None, "exercise_or_answer_structure_without_clear_boundary", {
            "category": category,
            "question_word_signal": bool(re.search(r"(?i)\b(?:question|problem)\b", normalized)),
            "numbered_question_lines": numbered_question_lines,
            "answer_lines": answer_lines,
        }
    return bounded, "eligible", {
        "category": category,
        "book_title": book_title,
        "explicit_excluded_section_boundary": boundary is not None,
        "question_word_signal": bool(re.search(r"(?i)\b(?:question|problem)\b", normalized)),
        "numbered_question_lines": numbered_question_lines,
        "answer_lines": answer_lines,
    }


def decode_chem(path, source_cfg):
    actual = sha256_file(path)
    if path.stat().st_size != source_cfg["size_bytes"] or actual != source_cfg["sha256"]:
        raise ValueError("Chem local Parquet does not match the fixed source SHA-256")
    pf = pq.ParquetFile(path)
    if pf.schema_arrow.names != ["url", "text"]:
        raise ValueError("Chem source schema differs from the audited fixed source")
    docs = []
    decoded = 0
    rg_counts = []
    stage = collections.Counter()
    diagnostics = collections.Counter()
    seen_urls = set()
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=["url", "text"])
        rg_counts.append(table.num_rows)
        urls = table.column("url").to_pylist()
        texts = table.column("text").to_pylist()
        for offset, (url, text) in enumerate(zip(urls, texts)):
            row = decoded + offset
            if not isinstance(url, str) or not url.strip():
                stage["empty_url"] += 1
                continue
            if not isinstance(text, str) or not text.strip():
                stage["empty_text"] += 1
                continue
            excerpt, reason, flags = chemistry_excerpt(text, url)
            stage[reason] += 1
            diagnostics["question_problem_word_signal_rows"] += int(flags.get("question_word_signal", False))
            diagnostics["explicit_exercise_heading_boundaries"] += int(flags.get("explicit_excluded_section_boundary", False))
            diagnostics["exercise_structure_ambiguous_rows"] += int(reason == "exercise_or_answer_structure_without_clear_boundary")
            diagnostics["numbered_question_signal_lines"] += int(flags.get("numbered_question_lines", 0))
            diagnostics["answer_signal_lines"] += int(flags.get("answer_lines", 0))
            if reason != "eligible":
                continue
            url_key = normalize_text(url)
            if url_key in seen_urls:
                stage["duplicate_source_url_skipped"] += 1
                continue
            seen_urls.add(url_key)
            normalized = normalize_text(text)
            url_hash = sha256_bytes(url_key.encode("utf-8"))
            docs.append({
                "source_record_id": f"LibreText_Chemistry-default/train-00000-of-00001.parquet:row{row:06d}",
                "source_document_id": f"chem-url-sha256:{url_hash}",
                "source_url": url,
                "source_row_index": row,
                "source_row_group": rg,
                "text": excerpt,
                "source_text_sha256": sha256_bytes(normalized.encode("utf-8")),
                "normalized_duplicate_group_sha256": sha256_bytes(duplicate_key(normalized).encode("utf-8")),
                "source_text_characters": len(normalized),
                "eligible_expository_characters": len(excerpt),
                "chem_category": flags.get("category"),
                "chem_book_title": flags.get("book_title"),
                "expository_end_char": len(excerpt),
                "explicit_excluded_section_boundary": flags.get("explicit_excluded_section_boundary", False),
            })
        decoded += table.num_rows
    if decoded != pf.metadata.num_rows:
        raise ValueError("Chem source decode did not cover every Parquet row")
    return docs, {
        "source_file": source_cfg["file"],
        "local_path": str(path.relative_to(SOURCE_ROOT)),
        "size_bytes": path.stat().st_size,
        "local_sha256": actual,
        "expected_sha256": source_cfg["sha256"],
        "expected_sha256_match": True,
        "metadata_rows": pf.metadata.num_rows,
        "decoded_rows": decoded,
        "row_group_count": pf.metadata.num_row_groups,
        "row_group_rows": rg_counts,
        "schema": [{"name": f.name, "type": str(f.type)} for f in pf.schema_arrow],
        "selection_stage_counts": dict(stage),
        "selection_signal_counts": dict(diagnostics),
        "eligible_documents_before_normalized_dedup": len(docs),
        "all_row_groups_decoded": decoded == pf.metadata.num_rows,
        "source_audit_reused": "reports/source_audits/chempile_education_candidate_20261024.json",
    }


def ordered_deduplicate(docs):
    seen_keys = set()
    unique = []
    duplicates = 0
    for doc in docs:
        key = doc.get("normalized_duplicate_group_sha256")
        if not key:
            key = sha256_bytes(duplicate_key(doc["text"]).encode("utf-8"))
        if key in seen_keys:
            duplicates += 1
            continue
        seen_keys.add(key)
        item = dict(doc)
        item["normalized_duplicate_group_sha256"] = key
        unique.append(item)
    return unique, duplicates


def select_domain(docs, tokenizers, seed, excluded_hashes):
    unique, normalized_duplicates = ordered_deduplicate(docs)
    rng = __import__("random").Random(seed)
    rng.shuffle(unique)
    selected = []
    examined = 0
    no_span = 0
    contamination_excluded = 0
    duplicate_snippets = 0
    seen_snippets = set()
    for doc in unique:
        if len(selected) >= 320:
            break
        examined += 1
        candidate = dict(doc)
        try:
            window = bio.common_window(candidate, tokenizers, 384, 512, seed)
        except Exception as exc:
            raise RuntimeError(f"Tokenizer span selection failed ({type(exc).__name__})") from None
        if window is None:
            no_span += 1
            continue
        snippet_hash = sha256_bytes(window["text"].encode("utf-8"))
        if snippet_hash in excluded_hashes:
            contamination_excluded += 1
            continue
        if snippet_hash in seen_snippets:
            duplicate_snippets += 1
            continue
        seen_snippets.add(snippet_hash)
        selected.append({
            **doc,
            "text": window["text"],
            "char_start": window["char_start"],
            "char_end_exclusive": window["char_end_exclusive"],
            "chunk_index": 0,
            "text_sha256": snippet_hash,
            "shared_preselection_token_lengths": {
                key: window["token_lengths"][MODEL_KEYS.index(key)] for key in MODEL_KEYS
            },
            "sample_index": len(selected),
        })
    stats = {
        "source_documents_before_normalized_dedup": len(docs),
        "duplicate_normalized_documents_skipped": normalized_duplicates,
        "unique_documents": len(unique),
        "seed": seed,
        "selection_order": "candidate list in source row order; then independent random.Random(42) permutation per forget domain",
        "documents_examined": examined,
        "no_common_384_512_span": no_span,
        "exact_test_stem_matches_excluded_by_hash": contamination_excluded,
        "duplicate_selected_snippets_skipped": duplicate_snippets,
        "selected_samples": len(selected),
        "unexamined_after_quota": max(0, len(unique) - examined),
        "selected_source_documents_unique": len({x["source_document_id"] for x in selected}) == len(selected),
        "selected_duplicate_groups_unique": len({x["normalized_duplicate_group_sha256"] for x in selected}) == len(selected),
        "selected_snippet_hashes_unique": len({x["text_sha256"] for x in selected}) == len(selected),
    }
    if selected and "chem_category" in selected[0]:
        stats["selected_chemistry_category_counts"] = dict(sorted(collections.Counter(
            row["chem_category"] for row in selected
        ).items()))
    return selected, stats


def read_excluded(path):
    if path is None:
        return set()
    path = pathlib.Path(path)
    if not path.is_absolute():
        path = ROOT / path
    doc = read_json(path)
    return set(doc.get("excluded_candidate_text_sha256", []))


def phase_select(args):
    cuda_initialized_before = torch.cuda.is_initialized()
    if cuda_initialized_before:
        raise RuntimeError("CUDA was already initialized before selection; refusing to continue")
    config = read_json(CONFIG)
    run_config = read_json(RUN_CONFIG)
    bundle = tokenizer_bundle(run_config)
    tokenizers = [bundle[key][0] for key in MODEL_KEYS]
    cyber_cfg = config["new_sources"]["cyber"]
    chem_cfg = config["new_sources"]["chem"]
    cyber_path = SOURCE_ROOT / cyber_cfg["local_dir"] / cyber_cfg["file"]
    chem_path = SOURCE_ROOT / chem_cfg["local_dir"] / chem_cfg["file"]
    cyber_docs, cyber_meta = decode_cyber(cyber_path, cyber_cfg)
    chem_docs, chem_meta = decode_chem(chem_path, chem_cfg)
    excluded = read_excluded(args.exclude_hashes)
    root = pathlib.Path(args.stage_dir)
    if not root.is_absolute():
        root = ROOT / root
    make_private_dir(root)
    summaries = {}
    source_meta = {"cyber": cyber_meta, "chem": chem_meta}
    for domain, docs in (("cyber", cyber_docs), ("chem", chem_docs)):
        selected, stats = select_domain(docs, tokenizers, config["seed"], excluded)
        summaries[domain] = {**stats, "source": source_meta[domain]}
        out = root / f"selected_{domain}.jsonl"
        write_private_jsonl(out, selected)
        summaries[domain]["selected_path"] = str(out.relative_to(ROOT))
        summaries[domain]["selected_file_sha256"] = sha256_file(out)
    summaries["selection_config_sha256"] = sha256_file(CONFIG)
    summaries["selection_code_sha256"] = sha256_file(pathlib.Path(__file__).resolve())
    summaries["reused_bio_preparation_helper_sha256"] = sha256_file(pathlib.Path(__file__).resolve().parent / "prepare_bio_rmu_effect_data.py")
    summaries["cuda_initialized_before"] = cuda_initialized_before
    summaries["cuda_initialized_after"] = torch.cuda.is_initialized()
    if summaries["cuda_initialized_after"]:
        raise RuntimeError("CUDA initialized during CPU selection")
    summaries["tokenizers"] = {
        key: {
            "model_revision": bundle[key][1]["revision"],
            "tokenizer_fingerprint_sha256": bundle[key][1]["fingerprint_sha256"],
            "length": bundle[key][1]["length"],
        }
        for key in MODEL_KEYS
    }
    summaries["excluded_hash_list_path"] = str(pathlib.Path(args.exclude_hashes).resolve().relative_to(ROOT)) if args.exclude_hashes else None
    write_private_json(root / "selection_summary.json", summaries)
    print(json.dumps({
        "stage_dir": str(root.relative_to(ROOT)),
        "domains": {d: {
            "selected_samples": summaries[d]["selected_samples"],
            "source_rows_decoded": summaries[d]["source"]["decoded_rows"],
            "unique_source_documents": summaries[d]["unique_documents"],
            "no_common_span": summaries[d]["no_common_384_512_span"],
            "normalized_duplicates": summaries[d]["duplicate_normalized_documents_skipped"],
        } for d in ("cyber", "chem")},
        "status": "quota_met" if all(summaries[d]["selected_samples"] == 320 for d in ("cyber", "chem")) else "insufficient_samples",
    }, ensure_ascii=False))
    return 0 if all(summaries[d]["selected_samples"] == 320 for d in ("cyber", "chem")) else 2


def phase_tokenize(args):
    cuda_initialized_before = torch.cuda.is_initialized()
    if cuda_initialized_before:
        raise RuntimeError("CUDA was already initialized before tokenization; refusing to continue")
    config = read_json(CONFIG)
    run_config = read_json(RUN_CONFIG)
    if args.model_key not in MODEL_KEYS:
        raise ValueError("Unknown model key")
    key = args.model_key
    model_cfg = run_config["models"][key]
    prior_name = "manifest.json" if key == "qwen3vl8b" else "rmu_data_manifest.json"
    prior_path = SOURCE_ROOT / model_cfg["data_dir"] / prior_name
    tokenizer, tokmeta = bio.make_tokenizer(key, model_cfg, read_json(prior_path))
    tokmeta["model_key"] = key
    selection_dir = pathlib.Path(args.selection_dir)
    if not selection_dir.is_absolute():
        selection_dir = ROOT / selection_dir
    output_dir = ROOT / model_cfg["data_dir"] / "three_domain_rmu_20261024"
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing per-model output directory: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    metrics = {}
    for domain in ("cyber", "chem"):
        src = selection_dir / f"selected_{domain}.jsonl"
        selected = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line]
        if len(selected) != 320:
            raise ValueError(f"{domain} selection count is {len(selected)}, expected 320")
        rows = []
        lengths = []
        for index, sample in enumerate(selected):
            ids = tokenizer.encode(sample["text"], add_special_tokens=False, truncation=False)
            if not 384 <= len(ids) <= 512:
                raise ValueError(f"{domain} sample {index} has out-of-range token length")
            if any(not isinstance(token_id, int) or token_id < 0 or token_id >= tokmeta["length"] for token_id in ids):
                raise ValueError(f"{domain} sample {index} contains an out-of-range token ID")
            rows.append({
                "domain": domain,
                "sample_index": index,
                "sample_id": f"{domain}-train-{index:04d}",
                "source_repository": config["new_sources"][domain]["repository"],
                "source_revision": config["new_sources"][domain]["revision"],
                "source_configuration": config["new_sources"][domain].get("configuration"),
                "source_split": config["new_sources"][domain]["split"],
                "source_file": config["new_sources"][domain]["file"],
                "source_record_id": sample["source_record_id"],
                "source_document_id": sample["source_document_id"],
                "source_url": sample.get("source_url"),
                "source_row_index": sample["source_row_index"],
                "source_row_group": sample["source_row_group"],
                "source_document_sha256": sample["source_text_sha256"],
                "normalized_duplicate_group_sha256": sample["normalized_duplicate_group_sha256"],
                "chunk_index": 0,
                "char_start": sample["char_start"],
                "char_end_exclusive": sample["char_end_exclusive"],
                "source_text_characters": sample["source_text_characters"],
                "snippet_text_characters": len(sample["text"]),
                "text_sha256": sample["text_sha256"],
                "tokenizer_fingerprint_sha256": tokmeta["fingerprint_sha256"],
                "effective_token_length": len(ids),
                "token_ids": ids,
                "attention_mask": [1] * len(ids),
            })
            lengths.append(len(ids))
        out = output_dir / f"{domain}_forget_train_seed42.jsonl"
        write_private_jsonl(out, rows)
        metrics[domain] = {
            "count": len(rows),
            "path": str(out.relative_to(ROOT)),
            "sha256": sha256_file(out),
            "length_min": min(lengths),
            "length_mean": sum(lengths) / len(lengths),
            "length_max": max(lengths),
            "effective_token_total": sum(lengths),
            "attention_masks_all_ones": all(
                len(row["token_ids"]) == len(row["attention_mask"])
                and all(x == 1 for x in row["attention_mask"]) for row in rows
            ),
        }
    result = {
        "model_key": key,
        "model_repository": model_cfg["repository"],
        "model_revision": model_cfg["revision"],
        "tokenizer": {k: v for k, v in tokmeta.items() if k != "model_key"},
        "selection_dir": str(selection_dir.relative_to(ROOT)),
        "selection_summary_sha256": sha256_file(selection_dir / "selection_summary.json"),
        "verification_code_sha256": sha256_file(pathlib.Path(__file__).resolve()),
        "reused_bio_preparation_helper_sha256": sha256_file(pathlib.Path(__file__).resolve().parent / "prepare_bio_rmu_effect_data.py"),
        "cuda_initialized_before": cuda_initialized_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "full_model_loaded": False,
        "datasets": metrics,
    }
    if result["cuda_initialized_after"]:
        raise RuntimeError("CUDA initialized during tokenization")
    write_private_json(output_dir / "tokenization_summary.json", result)
    print(json.dumps({"model_key": key, "output_dir": str(output_dir.relative_to(ROOT)), "datasets": metrics}, ensure_ascii=False))
    return 0


def phase_verify(args):
    cuda_initialized_before = torch.cuda.is_initialized()
    if cuda_initialized_before:
        raise RuntimeError("CUDA was already initialized before token verification; refusing to continue")
    config = read_json(CONFIG)
    run_config = read_json(RUN_CONFIG)
    key = args.model_key
    model_cfg = run_config["models"][key]
    prior_name = "manifest.json" if key == "qwen3vl8b" else "rmu_data_manifest.json"
    prior_path = SOURCE_ROOT / model_cfg["data_dir"] / prior_name
    tokenizer, tokmeta = bio.make_tokenizer(key, model_cfg, read_json(prior_path))
    tokmeta["model_key"] = key
    selection_dir = pathlib.Path(args.selection_dir)
    if not selection_dir.is_absolute():
        selection_dir = ROOT / selection_dir
    output_dir = ROOT / model_cfg["data_dir"] / "three_domain_rmu_20261024"
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Token output directory is missing: {output_dir}")
    results = {}
    for domain in ("cyber", "chem"):
        source = [json.loads(line) for line in (selection_dir / f"selected_{domain}.jsonl").read_text().splitlines() if line]
        token_path = output_dir / f"{domain}_forget_train_seed42.jsonl"
        saved = [json.loads(line) for line in token_path.read_text().splitlines() if line]
        if len(source) != 320 or len(saved) != 320:
            raise ValueError(f"{domain} source/token count mismatch for {key}")
        lengths = []
        for index, (raw, row) in enumerate(zip(source, saved)):
            ids = tokenizer.encode(raw["text"], add_special_tokens=False, truncation=False)
            if ids != row["token_ids"]:
                raise ValueError(f"Saved token IDs do not reproduce for {key}/{domain}/{index}")
            if raw["text_sha256"] != row["text_sha256"] or len(ids) != row["effective_token_length"]:
                raise ValueError(f"Saved token row identity does not reproduce for {key}/{domain}/{index}")
            if not 384 <= len(ids) <= 512:
                raise ValueError(f"Saved token length outside 384..512 for {key}/{domain}/{index}")
            if len(row["attention_mask"]) != len(ids) or not all(v == 1 for v in row["attention_mask"]):
                raise ValueError(f"Saved attention mask invalid for {key}/{domain}/{index}")
            if any(not isinstance(token_id, int) or token_id < 0 or token_id >= tokmeta["length"] for token_id in ids):
                raise ValueError(f"Saved token ID outside tokenizer range for {key}/{domain}/{index}")
            lengths.append(len(ids))
        results[domain] = {
            "count": len(saved),
            "token_file": str(token_path.relative_to(ROOT)),
            "token_file_sha256": sha256_file(token_path),
            "reencoded_saved_token_ids_match": True,
            "text_order_matches_selection": [x["text_sha256"] for x in saved] == [x["text_sha256"] for x in source],
            "length_min": min(lengths),
            "length_mean": sum(lengths) / len(lengths),
            "length_max": max(lengths),
            "effective_token_total": sum(lengths),
        }
    shared_path = SOURCE_ROOT / config["existing_forget"]["shared_originals"]
    shared_rows = [json.loads(line) for line in shared_path.read_text(encoding="utf-8").splitlines() if line]
    reuse_checks = (
        ("bio", "forget", output_dir / "existing_bio_forget_train_seed42.jsonl"),
        ("retain", "retain", output_dir / "existing_wikitext_retain_train_seed42_v2.jsonl"),
    )
    for kind, shared_kind, token_path in reuse_checks:
        source_rows = [
            row for row in shared_rows
            if row["source_kind"] == shared_kind and row["group"] == "train"
        ]
        saved = [json.loads(line) for line in token_path.read_text(encoding="utf-8").splitlines() if line]
        if len(source_rows) != 320 or len(saved) != 320 or not os.path.samefile(
            token_path,
            SOURCE_ROOT / model_cfg["data_dir"] / (
                "bio_rmu_effect_forget_train_seed42.jsonl" if kind == "bio"
                else "bio_rmu_effect_retain_train_seed42_v2.jsonl"
            ),
        ):
            raise ValueError(f"Reused {kind} artifact identity/count mismatch for {key}")
        lengths = []
        for index, (raw, row) in enumerate(zip(source_rows, saved)):
            ids = tokenizer.encode(raw["text"], add_special_tokens=False, truncation=False)
            if ids != row["token_ids"] or raw["text_sha256"] != row["text_sha256"]:
                raise ValueError(f"Reused {kind} token sequence mismatch for {key}/{index}")
            if len(ids) != row.get("effective_token_length", len(ids)) or not 384 <= len(ids) <= 512:
                raise ValueError(f"Reused {kind} length invalid for {key}/{index}")
            if row.get("tokenizer_fingerprint_sha256") != tokmeta["fingerprint_sha256"] or any(
                token_id < 0 or token_id >= tokmeta["length"] for token_id in ids
            ):
                raise ValueError(f"Reused {kind} tokenizer identity/token ID range invalid for {key}/{index}")
            if len(row["attention_mask"]) != len(ids) or not all(v == 1 for v in row["attention_mask"]):
                raise ValueError(f"Reused {kind} attention mask invalid for {key}/{index}")
            lengths.append(len(ids))
        results[kind] = {
            "count": len(saved),
            "token_file": str(token_path.relative_to(ROOT)),
            "token_file_sha256": sha256_file(token_path),
            "reencoded_saved_token_ids_match": True,
            "text_order_matches_shared_originals": [r["text_sha256"] for r in saved] == [r["text_sha256"] for r in source_rows],
            "same_inode_as_original": True,
            "length_min": min(lengths),
            "length_mean": sum(lengths) / len(lengths),
            "length_max": max(lengths),
            "effective_token_total": sum(lengths),
        }
    result = {
        "status": "verified",
        "model_key": key,
        "model_revision": model_cfg["revision"],
        "tokenizer_fingerprint_sha256": tokmeta["fingerprint_sha256"],
        "selection_dir": str(selection_dir.relative_to(ROOT)),
        "selection_summary_sha256": sha256_file(selection_dir / "selection_summary.json"),
        "cuda_initialized_before": cuda_initialized_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "datasets": results,
        "full_model_loaded": False,
    }
    if result["cuda_initialized_after"]:
        raise RuntimeError("CUDA initialized during token verification")
    verification_name = args.output_name or "verification_summary.json"
    if pathlib.Path(verification_name).name != verification_name or not verification_name.endswith(".json"):
        raise ValueError("Verification output name must be a simple .json filename")
    verification_path = output_dir / verification_name
    write_private_json(verification_path, result)
    print(json.dumps({"status": result["status"], "model_key": key, "datasets": results}, ensure_ascii=False))
    return 0


def phase_freeze(args):
    config = read_json(CONFIG)
    run_config = read_json(RUN_CONFIG)
    selection_dir = ROOT / args.selection_dir
    audit_path = ROOT / args.audit_report
    audit = read_json(audit_path)
    expected_candidate_files = []
    for domain in ("cyber", "chem"):
        candidate_path = selection_dir / f"selected_{domain}.jsonl"
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Final selected originals are missing for {domain}")
        rows = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != 320:
            raise ValueError(f"Final selected originals have {len(rows)} rows for {domain}, expected 320")
        expected_candidate_files.append({
            "path": str(candidate_path.relative_to(ROOT)),
            "sha256": sha256_file(candidate_path),
            "candidate_snippet_count": 320,
        })
    if audit.get("source_sha256") != "575d52b7e86c819efbdbccf0bb4ee1fd5aa78761b5c5c5097c889af4556aa566":
        raise ValueError("WMDP stem-audit source identity does not match the frozen C list")
    if audit.get("question_stem_count") != 3668 or audit.get("candidate_snippet_count") != 640:
        raise ValueError("WMDP stem audit does not cover 3668 stems and all 640 candidates")
    if sorted(audit.get("candidate_files", []), key=lambda x: x["path"]) != sorted(expected_candidate_files, key=lambda x: x["path"]):
        raise ValueError("WMDP stem audit is not bound to the final selected files")
    if audit.get("matched_candidate_count") != 0:
        raise ValueError("Final test-stem audit still has matching candidate snippets")
    selection_summary = read_json(selection_dir / "selection_summary.json")
    if any(selection_summary[d]["selected_samples"] != 320 for d in ("cyber", "chem")):
        raise ValueError("Cannot freeze incomplete Cyber/Chem selection")
    selected_rows_by_domain = {}
    for domain in ("cyber", "chem"):
        rows = [json.loads(line) for line in (selection_dir / f"selected_{domain}.jsonl").read_text().splitlines() if line]
        selected_rows_by_domain[domain] = rows
        if (
            len({row["source_document_id"] for row in rows}) != 320
            or len({row["normalized_duplicate_group_sha256"] for row in rows}) != 320
            or len({row["text_sha256"] for row in rows}) != 320
        ):
            raise ValueError(f"{domain} has repeated source documents, duplicate groups or snippets")
        for row in rows:
            if sha256_bytes(row["text"].encode("utf-8")) != row["text_sha256"]:
                raise ValueError(f"{domain} selected text hash mismatch")
            if row["char_end_exclusive"] - row["char_start"] != len(row["text"]):
                raise ValueError(f"{domain} snippet offsets do not match character length")
            if any(
                not 384 <= row["shared_preselection_token_lengths"][key] <= 512
                for key in MODEL_KEYS
            ):
                raise ValueError(f"{domain} shared tokenizer length check failed")
    exact_cross_domain_overlap = len(
        {row["text_sha256"] for row in selected_rows_by_domain["cyber"]}
        & {row["text_sha256"] for row in selected_rows_by_domain["chem"]}
    )
    normalized_source_cross_domain_overlap = len(
        {row["normalized_duplicate_group_sha256"] for row in selected_rows_by_domain["cyber"]}
        & {row["normalized_duplicate_group_sha256"] for row in selected_rows_by_domain["chem"]}
    )
    result = {
        "schema_version": 1,
        "status": "frozen",
        "purpose": "three-domain RMU training-data preparation; this manifest does not define exposure ratios or update budgets",
        "seed": config["seed"],
        "preparation_code_sha256": sha256_file(pathlib.Path(__file__).resolve()),
        "reused_bio_preparation_helper_sha256": sha256_file(pathlib.Path(__file__).resolve().parent / "prepare_bio_rmu_effect_data.py"),
        "configuration_sha256": sha256_file(CONFIG),
        "assessment_audit_code_sha256": sha256_file(pathlib.Path(__file__).resolve().parent / "audit_three_domain_candidates_against_wmdp_stems.py"),
        "training_snippet_count_per_forget_domain": 320,
        "cyber_chem_cross_source_overlap": {
            "exact_snippet_hash_intersections": exact_cross_domain_overlap,
            "normalized_full_document_hash_intersections": normalized_source_cross_domain_overlap,
        },
        "retain_snippet_count": 320,
        "length_constraints": {"minimum_effective_tokens": 384, "maximum_tokens": 512},
        "processing": {
            "special_tokens": False,
            "chat_template": False,
            "padding": False,
            "cross_document_joining": False,
            "duplicate_normalization": config["selection_rules"]["deduplication"],
            "selection_order": config["selection_rules"]["candidate_order"],
            "no_repeated_text_padding": True,
            "test_contamination_check": {
                "report_path": str(audit_path.relative_to(ROOT)),
                "report_sha256": sha256_file(audit_path),
                "method": audit["method"],
                "question_stems_checked": audit["question_stem_count"],
                "candidate_snippets_checked": audit["candidate_snippet_count"],
                "matched_candidate_count": audit["matched_candidate_count"],
                "limitation": "Exact text overlap only; no match does not prove absence of semantic or other contamination.",
            },
        },
        "sources": {
            "bio": {
                **config["existing_forget"],
                "reuse_existing": True,
                "not_resampled": True,
                "not_redownloaded": True,
            },
            "cyber": {**config["new_sources"]["cyber"], **selection_summary["cyber"]["source"],
                      "selection": {k: v for k, v in selection_summary["cyber"].items() if k != "source"}},
            "chem": {**config["new_sources"]["chem"], **selection_summary["chem"]["source"],
                     "selection": {k: v for k, v in selection_summary["chem"].items() if k != "source"},
                     "label": "Chem 通用化学代理 forget 集",
                     "not_official_wmdp_chem_training": True,
                     "per_page_license_verified": False},
            "retain": {
                **config["existing_retain"],
                "reuse_existing": True,
                "not_resampled": True,
                "not_modified": True,
            },
        },
        "existing_bio_retain_reuse": {},
        "models": {},
    }
    shared = [json.loads(line) for line in (SOURCE_ROOT / config["existing_forget"]["shared_originals"]).read_text().splitlines() if line]
    for key in MODEL_KEYS:
        model_cfg = run_config["models"][key]
        model_dir = ROOT / model_cfg["data_dir"]
        prior = read_json(SOURCE_ROOT / model_cfg["data_dir"] / "bio_rmu_effect_manifest.json")
        keydata = {"model": {"repository": model_cfg["repository"], "revision": model_cfg["revision"]},
                   "tokenizer": prior["tokenizer"], "datasets": {}}
        for kind, filename, shared_kind in (
            ("bio", "bio_rmu_effect_forget_train_seed42.jsonl", "forget"),
            ("retain", "bio_rmu_effect_retain_train_seed42_v2.jsonl", "retain"),
        ):
            original_path = SOURCE_ROOT / model_cfg["data_dir"] / filename
            private_name = (
                "existing_bio_forget_train_seed42.jsonl" if kind == "bio"
                else "existing_wikitext_retain_train_seed42_v2.jsonl"
            )
            path = model_dir / "three_domain_rmu_20261024" / private_name
            if not path.is_file() or not os.path.samefile(path, original_path):
                raise ValueError(f"Restricted hardlink for reused {kind} data is missing or not the original file")
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            expected = [x for x in shared if x["source_kind"] == shared_kind and x["group"] == "train"]
            if len(rows) != 320 or len(expected) != 320:
                raise ValueError(f"Existing {kind} count mismatch for {key}")
            if [r["text_sha256"] for r in rows] != [r["text_sha256"] for r in expected]:
                raise ValueError(f"Existing {kind} source order differs for {key}")
            keydata["datasets"][kind] = {
                "path": str(path.relative_to(ROOT)),
                "original_path": str(original_path.relative_to(SOURCE_ROOT)),
                "same_inode_as_original": True,
                "sha256": sha256_file(path),
                "count": len(rows),
                "token_lengths": {
                    "minimum": min(len(r["token_ids"]) for r in rows),
                    "mean": sum(len(r["token_ids"]) for r in rows) / len(rows),
                    "maximum": max(len(r["token_ids"]) for r in rows),
                    "effective_token_total": sum(len(r["token_ids"]) for r in rows),
                },
                "reused_original_order": True,
            }
        token_dir = model_dir / "three_domain_rmu_20261024"
        token_summary = read_json(token_dir / "tokenization_summary.json")
        verification_name = args.verification_name or "verification_summary.json"
        verify_summary = read_json(token_dir / verification_name)
        if verify_summary.get("status") != "verified" or verify_summary.get("tokenizer_fingerprint_sha256") != token_summary["tokenizer"]["fingerprint_sha256"]:
            raise ValueError(f"{key} lacks a matching re-tokenization verification")
        for domain in ("cyber", "chem"):
            ds = token_summary["datasets"][domain]
            path = ROOT / ds["path"]
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            selected = [json.loads(line) for line in (selection_dir / f"selected_{domain}.jsonl").read_text().splitlines() if line]
            if len(rows) != 320 or [r["text_sha256"] for r in rows] != [r["text_sha256"] for r in selected]:
                raise ValueError(f"{key} {domain} token file does not match frozen shared originals/order")
            if any(not 384 <= len(r["token_ids"]) <= 512 or len(r["attention_mask"]) != len(r["token_ids"]) or not all(v == 1 for v in r["attention_mask"]) for r in rows):
                raise ValueError(f"{key} {domain} contains invalid token lengths or masks")
            if any(r["tokenizer_fingerprint_sha256"] != token_summary["tokenizer"]["fingerprint_sha256"] for r in rows):
                raise ValueError(f"{key} {domain} tokenizer fingerprint mismatch")
            verified = verify_summary["datasets"][domain]
            if verified.get("token_file_sha256") != sha256_file(path) or verified.get("reencoded_saved_token_ids_match") is not True:
                raise ValueError(f"{key} {domain} verification is not bound to the saved token file")
            keydata["datasets"][domain] = ds
        for kind in ("bio", "retain"):
            verified = verify_summary["datasets"].get(kind, {})
            current = keydata["datasets"][kind]
            if (
                verified.get("reencoded_saved_token_ids_match") is not True
                or verified.get("token_file_sha256") != current["sha256"]
                or verified.get("same_inode_as_original") is not True
            ):
                raise ValueError(f"{key} reused {kind} token verification is incomplete")
        keydata["tokenization_summary_path"] = str((token_dir / "tokenization_summary.json").relative_to(ROOT))
        keydata["tokenization_summary_sha256"] = sha256_file(token_dir / "tokenization_summary.json")
        keydata["verification_summary_path"] = str((token_dir / verification_name).relative_to(ROOT))
        keydata["verification_summary_sha256"] = sha256_file(token_dir / verification_name)
        result["models"][key] = keydata

    # Prove the same new raw text and sample order were used for all three models.
    for domain in ("cyber", "chem"):
        sequences = []
        for key in MODEL_KEYS:
            entry = result["models"][key]["datasets"][domain]
            rows = [json.loads(line) for line in (ROOT / entry["path"]).read_text().splitlines() if line]
            sequences.append([r["text_sha256"] for r in rows])
        if not all(seq == sequences[0] for seq in sequences[1:]):
            raise ValueError(f"Cross-model {domain} source order/text hashes differ")
    output_name = args.output_name or "manifest_verified.json"
    if pathlib.Path(output_name).name != output_name or not output_name.endswith(".json"):
        raise ValueError("Output manifest name must be a simple .json filename")
    manifest_path = ROOT / config["output_dir"] / output_name
    write_private_json(manifest_path, result)
    print(json.dumps({"status": "frozen", "manifest_path": str(manifest_path.relative_to(ROOT)), "manifest_sha256": sha256_file(manifest_path)}, ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("select", "tokenize", "verify", "freeze"))
    parser.add_argument("--stage-dir")
    parser.add_argument("--exclude-hashes")
    parser.add_argument("--selection-dir")
    parser.add_argument("--model-key", choices=MODEL_KEYS)
    parser.add_argument("--audit-report")
    parser.add_argument("--output-name")
    parser.add_argument("--verification-name")
    args = parser.parse_args()
    if args.phase == "select":
        if not args.stage_dir:
            parser.error("select requires --stage-dir")
        return phase_select(args)
    if args.phase == "tokenize":
        if not args.selection_dir or not args.model_key:
            parser.error("tokenize requires --selection-dir and --model-key")
        return phase_tokenize(args)
    if args.phase == "verify":
        if not args.selection_dir or not args.model_key:
            parser.error("verify requires --selection-dir and --model-key")
        return phase_verify(args)
    if not args.selection_dir or not args.audit_report:
        parser.error("freeze requires --selection-dir and --audit-report")
    return phase_freeze(args)


if __name__ == "__main__":
    raise SystemExit(main())
