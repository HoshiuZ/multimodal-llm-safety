# 三模型文本与多模态评测

本仓库提供三个模型的纯文本 C/S 评测材料，以及独立模型组合的图像输入评测材料。各项结果按对应的模型版本和固定评测协议列示。

**模型版本更新（2026-11-01）：** [LLaVA 当前 HF 版本](https://huggingface.co/HoshiuZ/LLaVA-NeXT-8B-Unlearned/tree/959a02037889f1f74cacfdc0152dc53444217563)使用基模作 NPO 参考（overlay `dd9b6bb0…`），文本 C/S 结果已按该检查点重新评测。

## WMDP C

3,668 题 WMDP C 汇总按对应模型版本列示：

| 模型 | C |
|---|---:|
| Qwen | 88.4871% |
| LLaVA（当前 HF2） | 80.9289% |
| InternVL | 76.5924% |

当前 LLaVA 的精确计数与 revision 见 [`text/results/srev3_llava_base_ref_cs_20261101.json`](text/results/srev3_llava_base_ref_cs_20261101.json)；Qwen 与 InternVL 见 `text/results/c_reference.json`。

## 当前 S 结果

默认使用 `srev1_partial147_20261031_v1`：固定 734 个原题、句法改写和问题复述。下表按 original / syntax / paraphrase 排列正确数。`raw S` 保留精确原值；展示值为 `capped S = min(raw S, 100)`。

| 模型 | 基模正确数 | cleared 正确数 | raw S | capped S |
|---|---:|---:|---:|---:|
| Qwen | 421 / 411 / 425 | 200 / 201 / 195 | 99.1994% | 99.1994% |
| LLaVA（当前 HF2） | 357 / 346 / 346 | 212 / 208 / 217 | 94.9878% | 94.9878% |
| InternVL | 405 / 405 / 406 | 220 / 208 / 227 | 96.3219% | 96.3219% |

S 参考线为 65%。当前 LLaVA 的逐场景计数、revision、overlay SHA256 和覆盖审计见 [`text/results/srev3_llava_base_ref_cs_20261101.json`](text/results/srev3_llava_base_ref_cs_20261101.json)；Qwen 与 InternVL 见 [`text/results/s_results_v2.json`](text/results/s_results_v2.json)。结果文件只含汇总数，不含逐题分数。

## LLaVA-NeXT-8B HF2 revision 的文本 C/S 结果

本次复核对应公开检查点 [HoshiuZ/LLaVA-NeXT-8B-Unlearned@959a020](https://huggingface.co/HoshiuZ/LLaVA-NeXT-8B-Unlearned/commit/959a02037889f1f74cacfdc0152dc53444217563)，基模 revision 为 `b041c0d0ea0dd0196d147206c210c8d1752fc2da`，更新 SHA256 为 `dd9b6bb0c01c2a40437c89c127540e83c7bd6554b1892984e2273ed946ddd1da`。输入为纯文本，评分加载完整 LLaVA 多模态 wrapper。

| 评测 | 基模正确数 | cleared 正确数 | 结果 | 参考线 |
|---|---:|---:|---:|---:|
| WMDP C（3,668 题） | 1,929 | 1,110 | 80.9288537549% | 70% |
| S（每场景 734 题） | 357 / 346 / 346 | 212 / 208 / 217 | raw S 94.9877984085% | 65% |

S 计数依次对应 original、syntax、paraphrase。完整计数、精确分数和覆盖记录见 [`text/results/srev3_llava_base_ref_cs_20261101.json`](text/results/srev3_llava_base_ref_cs_20261101.json)。


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

三模型训练脚本、固定模型来源、GROM/NPO 方法引用、运行前置条件和许可边界见[训练源码说明](training/README.md)。脚本只提供研究代码；本仓库增量不包含基模、GROM basis、训练数据、token IDs 或检查点。LLaVA 的 2026-11-01 基模参考版已在 HF `main` 发布，当前 C/S 汇总对应该版本。

## 导航

- [三模型文本训练源码与运行条件](training/README.md)
- [完整 HF 评测命令、评分合同与 C/S 汇总](text/README.md)
- [外部数据获取、冻结改写和审核局限](text/data/README.md)
- [实际模型身份、revision 与稀疏更新 SHA256](text/models/README.md)
- [当前 LLaVA C/S 汇总](text/results/srev3_llava_base_ref_cs_20261101.json)、[Qwen/InternVL C 汇总](text/results/c_reference.json)与[Qwen/InternVL S 汇总](text/results/s_results_v2.json)
- [代码与数据来源哈希](text/results/source_provenance.json)
- [第三方来源与许可](THIRD_PARTY_NOTICES.md)
- [三个已发布的完整模型](text/models/README.md#已发布的清除后模型)；本仓库不包含模型权重。

[`text/results/reference_results.json`](text/results/reference_results.json) 保留 PKG1 历史参考，不是本次 PKG3 的评分记录。
