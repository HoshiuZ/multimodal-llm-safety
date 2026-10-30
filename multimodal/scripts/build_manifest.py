#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 MM-SafetyBench 官方目录构建可复现的图文样本清单（CPU 即可）。

官方 data/ 布局包含 processed_questions/*.json 和 imgs/{scenario}/{SD,SD_TYPO,TYPO}/*.jpg。
问题文本必须随图像类型匹配，不能只扫图片后留空。

用法：
    python scripts/build_manifest.py --root ~/data/MM-SafetyBench/data \
        --questions-dir ~/data/MM-SafetyBench/data/processed_questions \
        --out ~/hkc/data/manifest_mm.csv --dry-run
    # dry-run 完整通过后，去掉 --dry-run 正式生成（会计算图像 SHA-256）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from xmodal import manifest as MF  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def image_relpath(root: str, scenario: str, img_type: str, question_id: str) -> str:
    """解析原始图片路径；官方是 JPG，也兼容同名 JPEG/PNG 文件。"""
    folder = MF.IMAGE_DIRS[img_type]
    rel_dir = os.path.join("imgs", scenario, folder)
    for ext in IMAGE_EXTENSIONS:
        rel = os.path.join(rel_dir, question_id + ext)
        if os.path.isfile(os.path.join(root, rel)):
            return rel.replace(os.sep, "/")
    # 返回规范预期路径，让缺图检查输出明确的文件名。
    return os.path.join(rel_dir, question_id + ".jpg").replace(os.sep, "/")


def rows_from_official_questions(questions_dir: str, root: str) -> list[dict]:
    """读取上游 processed_questions JSON，并按官方说明配对三类图像和提问。"""
    files = sorted(
        os.path.join(questions_dir, name)
        for name in os.listdir(questions_dir)
        if name.lower().endswith(".json")
    )
    if not files:
        raise ValueError(f"没有找到问题 JSON 文件：{questions_dir}")

    rows: list[dict] = []
    for path in files:
        scenario = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            questions = json.load(f)
        if not isinstance(questions, dict):
            raise ValueError(f"问题 JSON 顶层应为对象：{path}")

        for question_id, item in questions.items():
            try:
                qidx = int(question_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"问题编号不是整数：{path}: {question_id!r}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"问题记录应为对象：{path}: {question_id!r}")

            for img_type in MF.IMAGE_TYPES:
                field = MF.QUERY_FIELDS[img_type]
                query_value = item.get(field, "")
                query = query_value.strip() if isinstance(query_value, str) else ""
                rows.append({
                    "scenario": scenario,
                    "task": "mm-safetybench",
                    "qidx": qidx,
                    "img_type": img_type,
                    "query": query,
                    "image_path": image_relpath(root, scenario, img_type, str(question_id)),
                })
    return rows


def rows_from_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 第 {line_no} 行格式错误：{exc}") from exc
    return rows


def inspect(rows: list[dict], root: str, expected_total: int) -> tuple[list, list[str]]:
    recs = MF.records_from_rows(rows)
    problems: list[str] = []
    cells = Counter(rec.cell for rec in recs)
    scenarios = {rec.scenario for rec in recs}
    qids = {(rec.scenario, rec.qidx) for rec in recs}
    identities = [(rec.scenario, rec.qidx, rec.img_type) for rec in recs]
    duplicate_rows = [key for key, n in Counter(identities).items() if n > 1]

    if expected_total and len(recs) != expected_total:
        problems.append(f"样本行数应为 {expected_total}，实际为 {len(recs)}")
    if scenarios != set(MF.SCENARIOS):
        missing = sorted(set(MF.SCENARIOS) - scenarios)
        extra = sorted(scenarios - set(MF.SCENARIOS))
        problems.append(f"场景目录不匹配；缺少={missing[:4]}，额外={extra[:4]}")
    if len(cells) != len(MF.SCENARIOS) * len(MF.IMAGE_TYPES):
        problems.append(f"分层单元应为 39 格，实际为 {len(cells)} 格")
    if duplicate_rows:
        problems.append(f"场景/问题编号/图像类型重复 {len(duplicate_rows)} 项")
    if len(qids) != 1680 and expected_total == 5040:
        problems.append(f"唯一图文问题数应为 1680，实际为 {len(qids)}")

    missing_images = []
    missing_queries = []
    for rec in recs:
        abs_path = os.path.join(root, rec.image_path)
        if not os.path.isfile(abs_path):
            missing_images.append(rec.image_path)
        if not rec.query.strip():
            missing_queries.append((rec.scenario, rec.qidx, rec.img_type))
    if missing_images:
        problems.append(f"缺失图片 {len(missing_images)} 张，例如 {missing_images[:3]}")
    if missing_queries:
        problems.append(f"缺失配套问题文本 {len(missing_queries)} 条，例如 {missing_queries[:3]}")

    print(f"样本行数：{len(recs)}；唯一图文问题：{len(qids)}；场景：{len(scenarios)}；分层单元：{len(cells)}/39")
    for key in sorted(cells):
        print(f"  {key:48s} {cells[key]}")
    return recs, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="官方 data/ 目录，包含 imgs/ 和 processed_questions/")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--questions-dir", help="官方 processed_questions/ 目录")
    source.add_argument("--from-jsonl", help="预整理 JSONL（字段：scenario/task/qidx/img_type/query/image_path）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-total", type=int, default=5040,
                    help="预期清单行数；传 0 可跳过总数检查，默认 5040")
    ap.add_argument("--dry-run", action="store_true", help="仅盘点校验，不写文件、不计算哈希")
    ap.add_argument("--no-hash", action="store_true", help="跳过图像 SHA-256（正式交付不推荐）")
    args = ap.parse_args()

    if args.questions_dir:
        raw = rows_from_official_questions(args.questions_dir, args.root)
    else:
        raw = rows_from_jsonl(args.from_jsonl)
    print(f"读取到 {len(raw)} 条清单候选")
    if not raw:
        print("[失败] 没有清单记录")
        return 2

    recs, problems = inspect(raw, args.root, args.expected_total)
    if problems:
        print("\n[失败] 数据盘点未通过：")
        for problem in problems:
            print(f"  - {problem}")
        if args.dry_run:
            print("\n未写文件。先解决上述数据或字段问题，再重新 dry-run。")
        return 3

    if args.dry_run:
        print("\n[通过] 官方问题、提示词、图片及 39 格分布完整；未写文件。")
        return 0

    manifest_rows = MF.build_manifest(recs, args.root, hash_images=not args.no_hash)
    MF.write_csv(manifest_rows, args.out)
    report = MF.verify(manifest_rows, args.root)
    print(f"\n清单已写出：{args.out}")
    print(f"清单指纹 digest = {report['digest']}")
    print(f"SHA-256 校验：{'通过' if report['ok'] else '失败'}")
    for problem in report["problems"]:
        print(f"  - {problem}")
    return 0 if report["ok"] else 4


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        sys.exit(2)
