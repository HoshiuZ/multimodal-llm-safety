# -*- coding: utf-8 -*-
"""三个必测模型的唯一事实来源（方案 6.2 节，老师已确认）。

为什么把它写成代码而不是配置：模型 ID 一旦被抄到三份 yaml 里，就一定会漂。
第三方复现的第一个问题永远是“你们到底测的哪个 checkpoint”，这里给唯一答案。

`revision` 默认是 "main"，这是**故意留下的坑**：上线前必须 pin 到具体 commit，
否则同名分支一更新，别人拉到的就不是我们测的那份权重。见 docstring 末尾的 TODO。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------

FAMILY_QWEN = "qwen"
FAMILY_LLAVA = "llava"
FAMILY_INTERNVL = "internvl"


@dataclass(frozen=True)
class ModelSpec:
    key: str                  # 本项目内部短名，results/ 目录用它
    family: str
    name: str                 # 方案 6.2 的写法，报告里用这个
    hf_id: str                # 候选权重 ID，D1 核验后锁定
    trust_remote_code: bool
    image_token: str          # 文本侧的图像占位符
    notes: str
    revision: str = "main"    # TODO(D1): pin 到 commit sha

    @property
    def pinned(self) -> bool:
        return self.revision != "main"


REGISTRY: dict[str, ModelSpec] = {
    "qwen3vl8b": ModelSpec(
        key="qwen3vl8b",
        family=FAMILY_QWEN,
        name="Qwen3-VL-8B-Instruct",
        hf_id="Qwen/Qwen3-VL-8B-Instruct",
        trust_remote_code=False,
        image_token="<|vision_start|><|image_pad|><|vision_end|>",
        notes="优先走官方 transformers 集成；D1 需确认 5.16.1 是否已原生支持，"
              "否则要降到支持该架构的版本并写进环境表",
    ),
    "llavanext8b": ModelSpec(
        key="llavanext8b",
        family=FAMILY_LLAVA,
        name="LLaVA-NeXT-8B",
        hf_id="llava-hf/llama3-llava-next-8b-hf",
        trust_remote_code=False,
        image_token="<image>",
        notes="需配套 image processor 与 chat template；<image> 必须出现在首轮 user 文本里",
    ),
    "internvl3_8b": ModelSpec(
        key="internvl3_8b",
        family=FAMILY_INTERNVL,
        name="InternVL3-8B",
        hf_id="OpenGVLab/InternVL3-8B",
        trust_remote_code=True,
        image_token="<image>",
        notes="动态分辨率会切成多个 tile，tile 数直接决定显存与延迟，"
              "必须在 configs 里固定 max_num 并写进报告",
    ),
}

MODEL_ORDER: tuple[str, ...] = ("qwen3vl8b", "llavanext8b", "internvl3_8b")


def get(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"unknown model key: {key}; available={list(REGISTRY)}") from exc


def all_specs() -> list[ModelSpec]:
    return [REGISTRY[k] for k in MODEL_ORDER]


def build_messages(key: str, text: str, n_images: int = 1) -> list[dict]:
    """构造一条 user 消息。各家族的图像占位方式不同，这里统一，adapter 只管取用。

    返回的 content 是“带类型的片段”列表，由 adapter 翻译成 processor 的输入；
    不要在这里调用任何 processor——本模块要能在没有 torch/transformers 的机器上被测试。
    """
    spec = get(key)
    if n_images < 1:
        raise ValueError("n_images must be >= 1")

    if spec.family == FAMILY_QWEN:
        content = [{"type": "image"}] * n_images + [{"type": "text", "text": text}]
    elif spec.family == FAMILY_LLAVA:
        content = [{"type": "text", "text": spec.image_token * n_images + "\n" + text}]
    elif spec.family == FAMILY_INTERNVL:
        content = [{"type": "image"}] * n_images + [{"type": "text", "text": text}]
    else:  # pragma: no cover - 新增家族时必须显式处理
        raise ValueError(f"unsupported family: {spec.family}")

    return [{"role": "user", "content": content}]


def flatten_prompt(key: str, text: str, n_images: int = 1) -> str:
    """把消息压成纯文本提示（用于日志、审计与“第三方肉眼核对 prompt”）。"""
    spec = get(key)
    if spec.family == FAMILY_LLAVA:
        return spec.image_token * n_images + "\n" + text
    return spec.image_token * n_images + "\n" + text


def unpinned() -> list[str]:
    """还没 pin revision 的模型——封版前的检查项。"""
    return [s.key for s in all_specs() if not s.pinned]
