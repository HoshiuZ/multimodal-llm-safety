# 三模型文本与多模态评测

本仓库提供三个模型的纯文本 C/S 评测材料，以及独立模型组合的图像输入评测材料。各项结果按对应的模型版本和固定评测协议列示。

**模型版本更新（2026-11-01）：** [LLaVA 当前 HF 版本](https://huggingface.co/HoshiuZ/LLaVA-NeXT-8B-Unlearned/tree/959a02037889f1f74cacfdc0152dc53444217563)使用基模作 NPO 参考（overlay `dd9b6bb0…`）。下方 LLaVA 的 C/S 表格对应[旧版 HF1](https://huggingface.co/HoshiuZ/LLaVA-NeXT-8B-Unlearned/tree/hf1-2026-10-31)（overlay `627e1610…`）；当前 HF 版本的 S 尚未评测。新版本的独立实验记录为文本 WMDP C `80.4348%`，详见[训练说明](training/README.md)。

## WMDP C

PKG2 的 3,668 题 C 汇总保持不变：

| 模型 | C |
|---|---:|
| Qwen | 88.4871% |
| LLaVA（旧版 HF1） | 81.5217% |
| InternVL | 76.5924% |

精确计数与模型 revision 见 `text/results/c_reference.json`。

## 当前 S 结果

默认使用 `srev1_partial147_20261031_v1`：固定 734 个原题、句法改写和问题复述。下表按 original / syntax / paraphrase 排列正确数。`raw S` 保留精确原值；展示值为 `capped S = min(raw S, 100)`。

| 模型 | 基模正确数 | cleared 正确数 | raw S | capped S |
|---|---:|---:|---:|---:|
| Qwen | 421 / 411 / 425 | 200 / 201 / 195 | 99.1994% | 99.1994% |
| LLaVA（旧版 HF1） | 357 / 346 / 346 | 218 / 212 / 212 | 102.9286% | 100.0000% |
| InternVL | 405 / 405 / 406 | 220 / 208 / 227 | 96.3219% | 96.3219% |

S 参考线为 65%。清除率 C、逐场景计数、模型完整 revision、稀疏 overlay SHA256 和覆盖审计见 [`text/results/s_results_v2.json`](text/results/s_results_v2.json)。该文件只含汇总数，不含逐题分数。

此前 PKG3 的 S 数值仍保留在 [`text/results/s_results.json`](text/results/s_results.json)，对应旧版 734 题改写，可按 `pkg3_s2_v1` 重建；旧结果不由本增量改写。

## 数据与评分入口

固定上游 WMDP Parquet、两个冻结版本和评分 CLI 可读 manifest 的重建方法见 [`text/data/README.md`](text/data/README.md)。新版本是重建脚本默认值。模型 revision 与 overlay 标识在结果 JSON 中记录；模型权重和 overlay 不属于本增量。已发布模型下载地址见[模型说明](text/models/README.md)。

## 多模态评测结果

图像输入评测包采用[固定的 M1 适配器与 M3 v8 视觉复核器组合](multimodal/models/README.md)，在 MM-SafetyBench 上记录了以下参考结果。该模型组合的版本与上文三个完整最终模型的文本评测版本分别记录。

| 模型组合 | C_mm | 原始 S_mm | 展示 S_mm |
|---|---:|---:|---:|
| Qwen3-VL | 89.94% | 97.96% | 97.96% |
| LLaVA-NeXT | 91.37% | 105.37% | 100.00% |
| InternVL3 | 85.49% | 102.34% | 100.00% |

C_mm 使用固定的 5,040 题；S_mm 使用固定的 1,008 个原题及 FigStep、HADES 风格改写。展示 S_mm 取 `min(原始 S_mm, 100%)`，原始值仍予保留。评测代码、模型与数据版本、指标公式及裁判局限见[多模态评测说明](multimodal/README.md)和[参考结果文件](multimodal/results/reference_results.json)。

## 训练源码（文本输入）

三模型训练脚本、固定模型来源、GROM/NPO 方法引用、运行前置条件和许可边界见[训练源码说明](training/README.md)。脚本只提供研究代码；本仓库增量不包含基模、GROM basis、训练数据、token IDs 或检查点。LLaVA 的 2026-11-01 基模参考版已在 HF `main` 发布；原 J8 参考版由 `hf1-2026-10-31` 标签保留。两版是不同 overlay，现有 C/S 汇总仍对应旧版。

## 导航

- [三模型文本训练源码与运行条件](training/README.md)
- [完整 HF 评测命令、评分合同与 C/S 汇总](text/README.md)
- [外部数据获取、冻结改写和审核局限](text/data/README.md)
- [实际模型身份、revision 与稀疏更新 SHA256](text/models/README.md)
- [C 汇总](text/results/c_reference.json) 与 [新版 S 汇总](text/results/s_results_v2.json) 与 [历史 S 汇总](text/results/s_results.json)
- [代码与数据来源哈希](text/results/source_provenance.json)
- [第三方来源与许可](THIRD_PARTY_NOTICES.md)
- [三个已发布的完整模型](text/models/README.md#已发布的清除后模型)；本仓库不包含模型权重。

[`text/results/reference_results.json`](text/results/reference_results.json) 保留 PKG1 历史参考，不是本次 PKG3 的评分记录。
