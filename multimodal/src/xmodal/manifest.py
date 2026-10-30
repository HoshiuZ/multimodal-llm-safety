# -*- coding: utf-8 -*-
"""MM-SafetyBench 样本清单（manifest）的构建与校验。

“可复现”的第一道锁就在这里：每个样本有一个由**元数据**算出的稳定 id，图像内容另有
sha256 做完整性校验。两者分开的原因是——

  * sample_id 只由 (scenario, task, qidx, img_type) 决定，重存图片、换压缩方式都不会变，
    所以原始模型与清除后模型的逐样本结果永远能 join 上；
  * img_sha256 只用于证明“这张图没被换过”，一旦对不上就必须中止，而不是静默跳过。

数据适配器读取官方问题 JSON，并按上游约定把问题文本与 SD、SD_TYPO、TYPO 图片配对。
`records_from_rows()` 把适配结果转成 Record，`build_manifest()` 再计算图像哈希并落成清单。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

# MM-SafetyBench 官方仓库的 13 个场景目录名。保留目录名中的编号和上游拼写，
# 使 manifest 能直接对应原始发布包；若使用不同数据版本，应记录版本并显式适配。
SCENARIOS: tuple[str, ...] = (
    "01-Illegal_Activitiy",
    "02-HateSpeech",
    "03-Malware_Generation",
    "04-Physical_Harm",
    "05-EconomicHarm",
    "06-Fraud",
    "07-Sex",
    "08-Political_Lobbying",
    "09-Privacy_Violence",
    "10-Legal_Opinion",
    "11-Financial_Advice",
    "12-Health_Consultation",
    "13-Gov_Decision",
)

# 官方图像目录分别为 SD、SD_TYPO、TYPO。
IMAGE_TYPES: tuple[str, ...] = ("SD", "SD_TYPO", "TYPO")

IMAGE_DIRS: dict[str, str] = {
    "SD": "SD",
    "SD_TYPO": "SD_TYPO",
    "TYPO": "TYPO",
}

# 上游 README 指定 SD 使用专属问题改写，其余两类使用通用 Rephrased Question。
QUERY_FIELDS: dict[str, str] = {
    "SD": "Rephrased Question(SD)",
    "SD_TYPO": "Rephrased Question",
    "TYPO": "Rephrased Question",
}

ID_PREFIX = "mm-"
ID_HEX_LEN = 12


@dataclass(frozen=True)
class Record:
    scenario: str
    task: str
    qidx: int
    img_type: str
    query: str
    image_path: str          # 相对 data_root

    @property
    def cell(self) -> str:
        """分层单元：13 场景 × 3 图像类型 = 39 格。"""
        return f"{self.scenario}|{self.img_type}"


def sample_id(rec: Record) -> str:
    """稳定样本 id：只吃元数据，不吃图片字节。"""
    key = f"{rec.scenario}|{rec.task}|{rec.qidx}|{rec.img_type}"
    return ID_PREFIX + hashlib.sha256(key.encode("utf-8")).hexdigest()[:ID_HEX_LEN]


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_digest(ids: Sequence[str]) -> str:
    """抽样清单的指纹：排序后按行 join 再 hash。第三方照这个值核对抽样是否一致。"""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def records_from_rows(rows: Iterable[Mapping]) -> list[Record]:
    """把外部行（csv/jsonl/dict）转成 Record，并做字段级校验。"""
    out: list[Record] = []
    for i, r in enumerate(rows):
        try:
            out.append(
                Record(
                    scenario=str(r["scenario"]),
                    task=str(r["task"]),
                    qidx=int(r["qidx"]),
                    img_type=str(r["img_type"]),
                    query=str(r.get("query", "")),
                    image_path=str(r["image_path"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"row {i} missing field {exc}") from exc
    return out


def build_manifest(
    records: Sequence[Record], data_root: str, hash_images: bool = True
) -> list[dict]:
    """生成清单行。缺图直接抛错——不要用空哈希占位。"""
    rows: list[dict] = []
    for rec in records:
        abs_path = os.path.join(data_root, rec.image_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"image not found: {abs_path}")
        row = asdict(rec)
        row["sample_id"] = sample_id(rec)
        row["cell"] = rec.cell
        row["img_sha256"] = file_sha256(abs_path) if hash_images else ""
        rows.append(row)
    return rows


def write_csv(rows: Sequence[Mapping], path: str) -> None:
    fields = ["sample_id", "scenario", "task", "qidx", "img_type", "cell",
              "query", "image_path", "img_sha256"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cell_counts(rows: Sequence[Mapping]) -> dict[str, int]:
    return dict(Counter(r["cell"] for r in rows))


def verify(rows: Sequence[Mapping], data_root: str) -> dict:
    """第三方自检：数量、唯一性、39 格覆盖、图像哈希。

    返回 dict，`ok=False` 时必须中止评测——不允许“跳过坏的继续跑”。
    """
    problems: list[str] = []
    ids = [r["sample_id"] for r in rows]
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        problems.append(f"duplicate sample_id: {dup[:5]} (n={len(dup)})")

    counts = cell_counts(rows)
    missing = [s for s in SCENARIOS for t in IMAGE_TYPES if f"{s}|{t}" not in counts]
    if missing:
        problems.append(f"missing cells: {len(missing)} e.g. {missing[:3]}")

    sha_bad = []
    for r in rows:
        p = os.path.join(data_root, r["image_path"])
        if not os.path.isfile(p):
            sha_bad.append((r["sample_id"], "missing"))
            continue
        if r.get("img_sha256") and file_sha256(p) != r["img_sha256"]:
            sha_bad.append((r["sample_id"], "sha256"))
    if sha_bad:
        problems.append(f"image integrity failures: {len(sha_bad)} e.g. {sha_bad[:3]}")

    return {
        "ok": not problems,
        "n_samples": len(rows),
        "n_cells": len(counts),
        "problems": problems,
        "digest": manifest_digest(ids),
    }


def to_jsonl(rows: Sequence[Mapping], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
