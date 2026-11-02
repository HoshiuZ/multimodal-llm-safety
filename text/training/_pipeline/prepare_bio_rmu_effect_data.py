#!/usr/bin/env python3
"""Prepare shared long-text Bio RMU samples and per-model token files offline.

This script reads the fixed local Parquet imports, selects one contiguous segment
per source document, and writes private raw/token artifacts with mode 0600.
It never prints sample text or token IDs.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import re
import stat
import tempfile
import unicodedata

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = pathlib.Path(__import__("os").environ.get("RMU_DATA_ROOT", pathlib.Path(__file__).resolve().parents[3])).resolve()
PREP_CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs/bio_rmu_effect_data.json"
RUN_CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs/bio_rmu_effect.json"
QWEN_OLD_MANIFEST = ROOT / "data/processed/qwen3vl8b/manifest.json"
MODEL_KEYS = ("qwen3vl8b", "llava_next8b", "internvl3_8b")
MODEL_TOKENIZER_OPTIONS = {
    "qwen3vl8b": {"trust_remote_code": False, "use_fast": True},
    "llava_next8b": {"trust_remote_code": False, "use_fast": True},
    "internvl3_8b": {"trust_remote_code": True, "use_fast": False},
}
WORD = re.compile(r"\S+", re.UNICODE)


def parse_wikitext_heading(line):
    """Return (level, title) for balanced WikiText heading delimiters.

    WikiText uses spaces between the delimiter equals signs for sections, e.g.
    ``= = Section = =``. Count equals signs at both edges while allowing
    whitespace between them; never infer a level from adjacent equals alone.
    """
    stripped = line.strip()
    left_count = 0
    start = 0
    while start < len(stripped) and stripped[start] == "=":
        left_count += 1
        start += 1
        while start < len(stripped) and stripped[start].isspace():
            start += 1

    right_count = 0
    end = len(stripped)
    while end > start and stripped[end - 1] == "=":
        right_count += 1
        end -= 1
        while end > start and stripped[end - 1].isspace():
            end -= 1

    title = stripped[start:end].strip()
    if not left_count or left_count != right_count or not title:
        return 0, ""
    return left_count, title


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def project_relative(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def normalize_document(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def mode_private(path: pathlib.Path):
    os.chmod(path, 0o600)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PermissionError("Could not set private artifact mode 0600")


def write_exclusive(path: pathlib.Path, payload: bytes, created: list[pathlib.Path]):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {project_relative(path)}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        mode_private(tmp)
        os.link(tmp, path)
        mode_private(path)
        created.append(path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def write_jsonl(path: pathlib.Path, rows, created):
    payload = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                      for row in rows)
    write_exclusive(path, payload, created)


def write_json(path: pathlib.Path, document, created):
    payload = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_exclusive(path, payload, created)


def read_transfer_manifest(path: pathlib.Path):
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError("Malformed transfer checksum manifest")
        entries[pathlib.PurePosixPath(fields[1].strip().lstrip("* ")).name] = fields[0].lower()
    return entries


def inspect_and_read_parquet(path: pathlib.Path, expected_sha: str, text_field: str):
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"Transfer checksum mismatch for {path.name}")
    parquet = pq.ParquetFile(path)
    if text_field not in parquet.schema_arrow.names:
        raise ValueError(f"Expected text field missing from {path.name}")
    docs = []
    row_count = 0
    row_groups = []
    for group_index in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(group_index, columns=[text_field])
        row_groups.append(table.num_rows)
        values = table.column(text_field).to_pylist()
        for row_in_group, value in enumerate(values):
            row_index = row_count + row_in_group
            doc_id = f"{path.stem}:rg{group_index:02d}:row{row_in_group:04d}"
            if isinstance(value, str):
                text = normalize_document(value)
            else:
                text = ""
            docs.append({
                "source_record_id": doc_id,
                "source_document_id": doc_id,
                "text": text,
                "source_file": path.name,
                "source_row_index": row_index,
                "source_row_group": group_index,
            })
        row_count += table.num_rows
        del table, values
    if row_count != parquet.metadata.num_rows:
        raise ValueError(f"Not all rows were decoded from {path.name}")
    return docs, {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha,
        "transfer_sha256": expected_sha,
        "transfer_sha256_match": True,
        "upstream_lfs_sha256": None,
        "upstream_lfs_sha256_verified": False,
        "row_count": row_count,
        "row_group_count": parquet.metadata.num_row_groups,
        "rows_per_row_group": row_groups,
        "fields": [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow],
        "all_row_groups_read": row_count == parquet.metadata.num_rows,
    }


def read_bio_documents(input_dir: pathlib.Path, source_cfg, checksums):
    docs, files = [], []
    for name in source_cfg["files"]:
        if name not in checksums:
            raise ValueError("Transfer manifest does not cover every Bio shard")
        rows, metadata = inspect_and_read_parquet(input_dir / name, checksums[name], source_cfg["text_field"])
        docs.extend(rows)
        files.append(metadata)
    return docs, files, {"row_count": len(docs), "row_group_count": sum(f["row_group_count"] for f in files)}


def read_wikitext_documents(input_dir: pathlib.Path, source_cfg, checksums):
    name = source_cfg["files"][0]
    if name not in checksums:
        raise ValueError("Transfer manifest does not cover the WikiText shard")
    rows, metadata = inspect_and_read_parquet(input_dir / name, checksums[name], source_cfg["text_field"])
    articles = []
    current_lines = None
    current_title = ""
    preamble_nonempty = 0
    section_header_lines = 0
    heading_level_counts = {}

    def close_article():
        if current_lines is not None:
            content = normalize_document("\n".join(current_lines))
            articles.append((current_title, content))

    for row in rows:
        text = row["text"]
        if not text:
            continue
        level, title = parse_wikitext_heading(text)
        if level:
            heading_level_counts[str(level)] = heading_level_counts.get(str(level), 0) + 1
        if level == 1:
            close_article()
            current_title = title
            current_lines = [text]
        elif current_lines is None:
            if text.strip():
                preamble_nonempty += 1
        else:
            if level > 1:
                section_header_lines += 1
            current_lines.append(text)
    close_article()

    docs = []
    for article_index, (title, content) in enumerate(articles):
        title_hash = sha256_bytes(title.encode("utf-8"))
        doc_id = f"wikitext-train:article{article_index:05d}:{title_hash[:16]}"
        docs.append({
            "source_record_id": doc_id,
            "source_document_id": doc_id,
            "text": content,
            "article_index": article_index,
            "title_sha256": title_hash,
            "source_file": name,
        })
    grouping = {
        "rule": source_cfg["article_header_rule"],
        "article_count": len(articles),
        "heading_level_counts": {key: heading_level_counts[key] for key in sorted(heading_level_counts, key=int)},
        "section_header_lines_kept_inside_articles": section_header_lines,
        "nonempty_preamble_lines_excluded": preamble_nonempty,
        "source_rows": metadata["row_count"],
    }
    return docs, [metadata], grouping


def deduplicate_documents(docs):
    seen = set()
    unique = []
    empty_count = 0
    duplicate_count = 0
    for doc in docs:
        text = doc["text"]
        if not text:
            empty_count += 1
            continue
        doc_hash = sha256_bytes(text.encode("utf-8"))
        if doc_hash in seen:
            duplicate_count += 1
            continue
        seen.add(doc_hash)
        item = dict(doc)
        item["source_document_sha256"] = doc_hash
        unique.append(item)
    return unique, {"empty_documents": empty_count, "duplicate_full_documents": duplicate_count}


def make_tokenizer(model_key, model_cfg, existing_manifest):
    model_path = (ROOT / model_cfg["model_path"]).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Pinned local model snapshot is missing: {model_cfg['model_path']}")
    prior_tokenizer = existing_manifest.get("tokenizer", {})
    files_key = "files" if model_key == "qwen3vl8b" else "fingerprint_files"
    expected_files = prior_tokenizer.get(files_key)
    if not expected_files:
        raise ValueError(f"Prior tokenizer fingerprint file list missing for {model_key}")
    actual_files = []
    for entry in expected_files:
        path = model_path / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Pinned tokenizer asset is missing for {model_key}")
        actual = {"path": entry["path"], "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != entry:
            raise ValueError(f"Pinned tokenizer asset changed for {model_key}: {entry['path']}")
        actual_files.append(actual)
    fingerprint = canonical_hash(actual_files)
    expected_fingerprint = model_cfg["tokenizer_fingerprint_sha256"]
    if fingerprint != expected_fingerprint or fingerprint != prior_tokenizer.get("fingerprint_sha256"):
        raise ValueError(f"Tokenizer fingerprint differs from pinned config for {model_key}")
    options = MODEL_TOKENIZER_OPTIONS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True,
        trust_remote_code=options["trust_remote_code"], use_fast=options["use_fast"],
    )
    if len(tokenizer) != int(prior_tokenizer["length"]):
        raise ValueError(f"Tokenizer length differs from pinned manifest for {model_key}")
    return tokenizer, {
        "repository": model_cfg["repository"],
        "revision": model_cfg["revision"],
        "class": type(tokenizer).__name__,
        "is_fast": bool(tokenizer.is_fast),
        "vocab_size": int(tokenizer.vocab_size),
        "length": int(len(tokenizer)),
        "fingerprint_sha256": fingerprint,
        "fingerprint_files": actual_files,
    }


def encode_length(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))


def common_window(doc, tokenizers, minimum, maximum, seed):
    text = doc["text"]
    words = list(WORD.finditer(text))
    if not words:
        return None
    doc_seed = int.from_bytes(hashlib.sha256(
        f"{seed}|{doc['source_document_id']}".encode("utf-8")).digest()[:8], "big")
    local_rng = random.Random(doc_seed)
    max_start_index = max(0, len(words) // 4)
    starts = [local_rng.randint(0, max_start_index), 0]
    starts = list(dict.fromkeys(starts))

    for start_word in starts:
        start_char = words[start_word].start()
        cache = {}

        def counts_at(end_word_index):
            if end_word_index not in cache:
                end_char = words[end_word_index].end()
                snippet = text[start_char:end_char]
                cache[end_word_index] = tuple(encode_length(tok, snippet) for tok in tokenizers)
            return cache[end_word_index]

        low, high = start_word, len(words) - 1
        best = start_word - 1
        while low <= high:
            middle = (low + high) // 2
            counts = counts_at(middle)
            if max(counts) <= maximum:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        candidate_indexes = set(range(max(start_word, best - 3), min(len(words), best + 4)))
        if best < start_word:
            candidate_indexes = {start_word, min(start_word + 1, len(words) - 1)}
        valid = []
        for end_word_index in candidate_indexes:
            counts = counts_at(end_word_index)
            if min(counts) >= minimum and max(counts) <= maximum:
                end_char = words[end_word_index].end()
                valid.append((end_char - start_char, end_word_index, counts, end_char))
        if valid:
            _, end_word_index, counts, end_char = max(valid)
            return {
                "text": text[start_char:end_char],
                "char_start": start_char,
                "char_end_exclusive": end_char,
                "token_lengths": list(counts),
                "chunk_index": 0,
            }
    return None


def select_source(docs, tokenizers, prep_cfg, kind):
    unique_docs, duplicate_stats = deduplicate_documents(docs)
    rng = random.Random(prep_cfg["seed"])
    rng.shuffle(unique_docs)
    needed = prep_cfg["training_samples_per_source"] + prep_cfg["diagnostic_samples_per_source"]
    selected = []
    no_common_span = 0
    examined = 0
    duplicate_snippets = 0
    used_snippet_hashes = set()
    for doc in unique_docs:
        if len(selected) >= needed:
            break
        examined += 1
        try:
            result = common_window(doc, tokenizers, prep_cfg["minimum_length_tokens"],
                                  prep_cfg["max_length_tokens"], prep_cfg["seed"])
        except Exception as exc:
            raise RuntimeError(f"Tokenizer failed while processing {kind} document {examined} ({type(exc).__name__})") from None
        if result is None:
            no_common_span += 1
            continue
        text_hash = sha256_bytes(result["text"].encode("utf-8"))
        if text_hash in used_snippet_hashes:
            duplicate_snippets += 1
            continue
        used_snippet_hashes.add(text_hash)
        selected.append({**doc, **result, "text_sha256": text_hash})
    stats = {
        "source_documents_total": len(docs),
        "empty_documents": duplicate_stats["empty_documents"],
        "duplicate_full_documents": duplicate_stats["duplicate_full_documents"],
        "unique_nonempty_documents": len(unique_docs),
        "selection_order": "seeded random permutation over the complete source document set; filter until quota is met",
        "documents_examined_until_quota": examined,
        "documents_unexamined_after_quota": len(unique_docs) - examined,
        "examined_documents_without_common_384_512_span": no_common_span,
        "duplicate_selected_snippets_skipped": duplicate_snippets,
        "selected_documents": len(selected),
        "selected_source_file_counts": {
            name: sum(row.get("source_file") == name for row in selected)
            for name in sorted({row.get("source_file", "") for row in selected})
        },
        "selected_source_document_hashes_unique": len({x["source_document_sha256"] for x in selected}) == len(selected),
        "selected_snippet_hashes_unique": len({x["text_sha256"] for x in selected}) == len(selected),
    }
    if len(selected) != needed:
        stats["sample_gap"] = needed - len(selected)
    return selected, stats


def tokenizer_rows(selected, tokenizer, tokenizer_meta, source_meta, group_name):
    rows = []
    lengths = []
    records = []
    for index, sample in enumerate(selected):
        token_ids = tokenizer.encode(sample["text"], add_special_tokens=False, truncation=False)
        if not 384 <= len(token_ids) <= 512 or len(token_ids) != sample["token_lengths"][MODEL_KEYS.index(tokenizer_meta["model_key"])]:
            raise ValueError("Token length changed between shared selection and final serialization")
        if any(not isinstance(token_id, int) or token_id < 0 or token_id >= tokenizer_meta["length"] for token_id in token_ids):
            raise ValueError("Tokenizer returned an out-of-range token ID")
        sample_id = f"{source_meta['kind']}-{group_name}-{index:04d}"
        row = {
            "sample_index": index,
            "sample_id": sample_id,
            "source_repo": source_meta["repository"],
            "source_revision": source_meta["revision"],
            "split": source_meta["split"],
            "source_record_id": sample["source_record_id"],
            "source_document_id": sample["source_document_id"],
            "chunk_index": sample["chunk_index"],
            "char_start": sample["char_start"],
            "char_end_exclusive": sample["char_end_exclusive"],
            "source_document_sha256": sample["source_document_sha256"],
            "text_sha256": sample["text_sha256"],
            "tokenizer_fingerprint_sha256": tokenizer_meta["fingerprint_sha256"],
            "token_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }
        rows.append(row)
        lengths.append(len(token_ids))
        records.append({
            "sample_index": index,
            "sample_id": sample_id,
            "source_record_id": sample["source_record_id"],
            "source_document_id": sample["source_document_id"],
            "source_document_sha256": sample["source_document_sha256"],
            "chunk_index": sample["chunk_index"],
            "char_start": sample["char_start"],
            "char_end_exclusive": sample["char_end_exclusive"],
            "text_sha256": sample["text_sha256"],
            "effective_token_length": len(token_ids),
        })
    length_stats = {
        "minimum": min(lengths) if lengths else None,
        "mean": sum(lengths) / len(lengths) if lengths else None,
        "maximum": max(lengths) if lengths else None,
        "effective_token_total": sum(lengths),
        "all_attention_masks_valid": all(len(row["token_ids"]) == len(row["attention_mask"])
                                          and all(bit == 1 for bit in row["attention_mask"]) for row in rows),
    }
    return rows, records, length_stats


def source_file_lookup(source_data):
    lookup = {}
    for source_key, details in source_data.items():
        for file in details["files"]:
            lookup[file["file"]] = file
    return lookup


def main():
    prep_cfg = read_json(PREP_CONFIG_PATH)
    run_cfg = read_json(RUN_CONFIG_PATH)
    if prep_cfg.get("schema_version") != 1 or run_cfg.get("source", {}).get("commit") != "c0b6c12bb0de3decf9c13bb13f9f1aa15754a132":
        raise ValueError("Pinned preparation or author RMU configuration is invalid")
    input_dir = ROOT / prep_cfg["input_dir"]
    transfer_path = input_dir / prep_cfg["transfer_manifest"]
    transfer_hashes = read_transfer_manifest(transfer_path)

    source_data = {}
    docs_by_kind = {}
    bio_cfg = prep_cfg["sources"]["forget"]
    bio_docs, bio_files, bio_coverage = read_bio_documents(input_dir, bio_cfg, transfer_hashes)
    docs_by_kind["forget"] = bio_docs
    source_data["forget"] = {**bio_cfg, **bio_coverage, "files": bio_files}
    retain_cfg = prep_cfg["sources"]["retain"]
    wiki_docs, wiki_files, wiki_grouping = read_wikitext_documents(input_dir, retain_cfg, transfer_hashes)
    docs_by_kind["retain"] = wiki_docs
    source_data["retain"] = {**retain_cfg, "row_count": wiki_grouping["source_rows"],
                              "article_count": wiki_grouping["article_count"],
                              "files": wiki_files, "grouping": wiki_grouping}
    transfer_manifest_sha256 = sha256_file(transfer_path)

    model_tokenizers = {}
    for model_key in MODEL_KEYS:
        model_cfg = run_cfg["models"][model_key]
        prior_name = "manifest.json" if model_key == "qwen3vl8b" else "rmu_data_manifest.json"
        prior_manifest = read_json(ROOT / model_cfg["data_dir"] / prior_name)
        tokenizer, tokenizer_meta = make_tokenizer(model_key, model_cfg, prior_manifest)
        tokenizer_meta["model_key"] = model_key
        model_tokenizers[model_key] = (tokenizer, tokenizer_meta)

    tokenizer_list = [model_tokenizers[key][0] for key in MODEL_KEYS]
    selected_by_kind = {}
    sampling_stats = {}
    for kind in ("forget", "retain"):
        selected, stats = select_source(docs_by_kind[kind], tokenizer_list, prep_cfg, kind)
        selected_by_kind[kind] = selected
        sampling_stats[kind] = stats

    if any(len(selected_by_kind[kind]) != prep_cfg["training_samples_per_source"] + prep_cfg["diagnostic_samples_per_source"]
           for kind in selected_by_kind):
        print(json.dumps({"status": "insufficient_samples", "sampling": sampling_stats}, ensure_ascii=False))
        return 2

    train_count = prep_cfg["training_samples_per_source"]
    all_shared = []
    grouped_shared = {}
    for kind, selected in selected_by_kind.items():
        groups = {"train": selected[:train_count], "diagnostic": selected[train_count:]}
        grouped_shared[kind] = groups
        group_doc_ids = {name: {row["source_document_id"] for row in rows} for name, rows in groups.items()}
        if group_doc_ids["train"] & group_doc_ids["diagnostic"]:
            raise ValueError("Training and diagnostic source documents overlap")
        for group_name, rows in groups.items():
            for index, sample in enumerate(rows):
                all_shared.append({
                    "source_kind": kind,
                    "group": group_name,
                    "sample_index": index,
                    "sample_id": f"{kind}-{group_name}-{index:04d}",
                    "source_repo": prep_cfg["sources"][kind]["repository"],
                    "source_revision": prep_cfg["sources"][kind]["revision"],
                    "split": prep_cfg["sources"][kind]["split"],
                    "source_record_id": sample["source_record_id"],
                    "source_document_id": sample["source_document_id"],
                    "source_document_sha256": sample["source_document_sha256"],
                    "chunk_index": sample["chunk_index"],
                    "char_start": sample["char_start"],
                    "char_end_exclusive": sample["char_end_exclusive"],
                    "text_sha256": sample["text_sha256"],
                    "text": sample["text"],
                })

    shared_dir = ROOT / prep_cfg["shared_output_dir"]
    shared_raw_path = shared_dir / prep_cfg["shared_originals_file"]
    shared_manifest_path = shared_dir / prep_cfg["shared_sample_manifest_file"]
    created = []
    expected_paths = [shared_raw_path, shared_manifest_path]
    model_output_paths = {}
    for model_key in MODEL_KEYS:
        data_dir = ROOT / run_cfg["models"][model_key]["data_dir"]
        expected_paths.append(data_dir / prep_cfg["model_manifest_name"])
        model_output_paths[model_key] = {}
        for kind in ("forget", "retain"):
            for group_name in ("train", "diagnostic"):
                expected_paths.append(data_dir / f"bio_rmu_effect_{kind}_{group_name}_seed42.jsonl")
                model_output_paths[model_key][(kind, group_name)] = data_dir / f"bio_rmu_effect_{kind}_{group_name}_seed42.jsonl"
    conflicts = [project_relative(path) for path in expected_paths if path.exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite existing new-data outputs: " + ", ".join(conflicts))

    try:
        write_jsonl(shared_raw_path, all_shared, created)
        shared_hash = sha256_file(shared_raw_path)
        source_lookup = source_file_lookup(source_data)
        shared_model_summary = {}
        for model_key in MODEL_KEYS:
            tokenizer, tokenizer_meta = model_tokenizers[model_key]
            token_groups = {}
            source_manifest = {}
            shared_model_summary[model_key] = {}
            for kind in ("forget", "retain"):
                source_cfg = prep_cfg["sources"][kind]
                per_group = {}
                group_records_for_manifest = {}
                for group_name, samples in grouped_shared[kind].items():
                    rows, sample_records, length_stats = tokenizer_rows(
                        samples, tokenizer, tokenizer_meta,
                        {"kind": kind, "repository": source_cfg["repository"],
                         "revision": source_cfg["revision"], "split": source_cfg["split"]},
                        group_name,
                    )
                    output_path = model_output_paths[model_key][(kind, group_name)]
                    write_jsonl(output_path, rows, created)
                    entry = {
                        "sample_count": len(rows),
                        "path": project_relative(output_path),
                        "sha256": sha256_file(output_path),
                        "token_lengths": length_stats,
                        "sample_records": sample_records,
                    }
                    per_group[group_name] = entry
                    group_records_for_manifest[group_name] = sample_records
                    shared_model_summary[model_key][f"{kind}_{group_name}"] = length_stats
                file_info = [source_lookup[name] for name in source_cfg["files"]]
                source_manifest[kind] = {
                    "repository": source_cfg["repository"],
                    "revision": source_cfg["revision"],
                    "split": source_cfg["split"],
                    "configuration": source_cfg.get("configuration"),
                    "files": file_info,
                    "source_documents_total": sampling_stats[kind]["source_documents_total"],
                    "sampling": sampling_stats[kind],
                    "grouping": source_data[kind].get("grouping"),
                    "groups": per_group,
                }
            model_manifest = {
                "schema_version": 1,
                "status": "prepared",
                "model": {"repository": run_cfg["models"][model_key]["repository"],
                          "revision": run_cfg["models"][model_key]["revision"]},
                "tokenizer": {k: v for k, v in tokenizer_meta.items() if k != "model_key"},
                "processing": {
                    "seed": prep_cfg["seed"],
                    "max_length_tokens": prep_cfg["max_length_tokens"],
                    "minimum_length_tokens": prep_cfg["minimum_length_tokens"],
                    "add_special_tokens": False,
                    "chat_template": False,
                    "image_placeholders": False,
                    "padding": "none; attention_mask contains ones only",
                    "source_text_normalization": "Unicode NFC; CRLF and CR converted to LF; trim outer whitespace",
                    "segment_selection": "one whitespace-boundary-aligned contiguous span per source document; common raw snippet constrained to 384..512 tokens for all three tokenizers",
                    "sample_order": "seeded random document permutation over all supplied files/articles; first 320 eligible documents train, next 64 diagnostic",
                    "deduplication": "exact SHA-256 of normalized full document before selection and exact SHA-256 of selected snippet",
                    "shared_originals_path": project_relative(shared_raw_path),
                    "shared_originals_sha256": shared_hash,
                },
                "sources": source_manifest,
            }
            model_manifest_path = ROOT / run_cfg["models"][model_key]["data_dir"] / prep_cfg["model_manifest_name"]
            write_json(model_manifest_path, model_manifest, created)

        common_samples = {}
        for kind, groups in grouped_shared.items():
            common_samples[kind] = {}
            for group_name, samples in groups.items():
                common_samples[kind][group_name] = [
                    {"sample_index": i, "source_document_id": x["source_document_id"],
                     "text_sha256": x["text_sha256"]}
                    for i, x in enumerate(samples)
                ]
        sample_manifest = {
            "schema_version": 1,
            "seed": prep_cfg["seed"],
            "shared_originals_path": project_relative(shared_raw_path),
            "shared_originals_sha256": shared_hash,
            "model_tokenizers": {key: {"repository": run_cfg["models"][key]["repository"],
                                        "revision": run_cfg["models"][key]["revision"],
                                        "fingerprint_sha256": model_tokenizers[key][1]["fingerprint_sha256"]}
                                  for key in MODEL_KEYS},
            "sample_sequence": common_samples,
            "token_lengths_by_model": shared_model_summary,
        }
        write_json(shared_manifest_path, sample_manifest, created)
    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    # Aggregate, non-sensitive execution report for the caller. It excludes text and IDs.
    print(json.dumps({
        "status": "prepared",
        "input_sources": {kind: {
            "repository": prep_cfg["sources"][kind]["repository"],
            "revision": prep_cfg["sources"][kind]["revision"],
            "split": prep_cfg["sources"][kind]["split"],
            "row_count": source_data[kind]["row_count"],
            "article_count": source_data[kind].get("article_count"),
            "files": [{"file": f["file"], "size_bytes": f["size_bytes"],
                       "sha256": f["sha256"], "transfer_sha256_match": f["transfer_sha256_match"],
                       "upstream_lfs_sha256_verified": f["upstream_lfs_sha256_verified"],
                       "row_group_count": f["row_group_count"]} for f in source_data[kind]["files"]],
        } for kind in ("forget", "retain")},
        "transfer_manifest": {
            "path": project_relative(transfer_path),
            "sha256": transfer_manifest_sha256,
            "matches_local_files": all(f["transfer_sha256_match"] for source in source_data.values() for f in source["files"]),
            "upstream_lfs_sha256_verification": "not performed; no upstream LFS checksum metadata was present in the imported-file directory",
        },
        "sampling": sampling_stats,
        "token_lengths_by_model": shared_model_summary,
        "artifacts": {
            "shared_originals": project_relative(shared_raw_path),
            "shared_sample_manifest": project_relative(shared_manifest_path),
            "model_manifests": {key: project_relative(ROOT / run_cfg["models"][key]["data_dir"] / prep_cfg["model_manifest_name"])
                                for key in MODEL_KEYS},
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
