#!/usr/bin/env python3
"""Independent exact-text overlap audit; reads only WMDP question strings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import unicodedata

ROOT = pathlib.Path(__import__("os").environ.get("RMU_DATA_ROOT", pathlib.Path(__file__).resolve().parents[3])).resolve()
FIXED_C = ROOT / "data/eval/text_cs_evolution_20261023/c_full.jsonl"
EXPECTED_C_SHA256 = "575d52b7e86c819efbdbccf0bb4ee1fd5aa78761b5c5c5097c889af4556aa566"
EXPECTED_C_ROWS = 3668
SPACE = re.compile(r"\s+", re.UNICODE)
QUESTION_VALUE = re.compile(rb'(?<!\\)"question"\s*:\s*("(?:[^"\\]|\\.)*")')


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    return SPACE.sub(" ", unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))).strip()


def write_private(path, obj):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    payload = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)


def read_only_question_stems(path):
    stems = []
    with path.open("rb") as stream:
        for line_no, line in enumerate(stream, 1):
            match = QUESTION_VALUE.search(line)
            if not match:
                raise ValueError(f"Frozen C row {line_no} lacks a question field")
            # Decode only the JSON string holding the stem; never parse answer,
            # choices, ID or score fields into Python objects.
            value = json.loads(match.group(1).decode("utf-8"))
            if not isinstance(value, str) or not normalize(value):
                raise ValueError(f"Frozen C row {line_no} has an empty question stem")
            stems.append(normalize(value))
    return stems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if sha256_file(FIXED_C) != EXPECTED_C_SHA256:
        raise ValueError("Frozen WMDP C list hash changed")
    stems = read_only_question_stems(FIXED_C)
    if len(stems) != EXPECTED_C_ROWS:
        raise ValueError("Frozen WMDP C list row count changed")
    candidate_paths = []
    candidates = []
    candidate_files = []
    for raw_path in args.candidates:
        candidate_path = pathlib.Path(raw_path)
        if not candidate_path.is_absolute():
            candidate_path = ROOT / candidate_path
        current = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines() if line]
        candidate_paths.append(candidate_path)
        candidate_files.append({
            "path": str(candidate_path.relative_to(ROOT)),
            "sha256": sha256_file(candidate_path),
            "candidate_snippet_count": len(current),
        })
        candidates.extend(current)
    matches = []
    for item in candidates:
        text = normalize(item["text"])
        question_hashes = [hashlib.sha256(stem.encode("utf-8")).hexdigest() for stem in stems if stem in text]
        if question_hashes:
            matches.append({
                "domain": (
                    "cyber" if item["source_record_id"].startswith("cyber-forget-corpus/")
                    else "chem"
                ),
                "candidate_text_sha256": item["text_sha256"],
                "matched_question_stem_sha256": sorted(set(question_hashes)),
            })
    excluded = sorted({item["candidate_text_sha256"] for item in matches})
    per_domain = {}
    for domain in ("cyber", "chem"):
        per_domain[domain] = sum(1 for item in matches if item["domain"].startswith(domain))
    result = {
        "schema_version": 1,
        "status": "checked",
        "source_path": str(FIXED_C.relative_to(ROOT)),
        "source_repository": "cais/wmdp",
        "source_revision": "7125571f22f032c56415e7980f48d877dd830ff8",
        "source_sha256": EXPECTED_C_SHA256,
        "question_field_only": True,
        "question_stem_count": len(stems),
        "candidate_files": candidate_files,
        "candidate_snippet_count": len(candidates),
        "method": "Exact full-question-stem substring match after NFC, newline normalization and whitespace collapse; case and punctuation preserved.",
        "matched_candidate_count": len(excluded),
        "matched_questions_per_domain": per_domain,
        "matches": matches,
        "excluded_candidate_text_sha256": excluded,
        "scope_limitation": "This checks exact question-stem overlap only; it does not prove absence of semantic or other contamination.",
    }
    out = pathlib.Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    write_private(out, result)
    print(json.dumps({
        "status": result["status"],
        "question_stems": len(stems),
        "candidate_snippets": len(candidates),
        "matched_candidates": len(excluded),
        "matched_questions_per_domain": per_domain,
        "excluded_hash_count": len(excluded),
        "output": str(out.relative_to(ROOT)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
