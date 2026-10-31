# 三模型纯文本 C/S 评测

本增量更新文本 S 数据版本、公开聚合结果和 CPU 评测入口。评测输入为纯文本，评分时加载的是各自完整多模态模型 wrapper；这些结果不表示图像能力或综合验收结论。

## WMDP C

PKG2 的 3,668 题 C 汇总保持不变：

| 模型 | C |
|---|---:|
| Qwen | 88.4871% |
| LLaVA | 81.5217% |
| InternVL | 76.5924% |

精确计数与模型 revision 见 `text/results/c_reference.json`。

## 当前 S 结果

默认使用 `srev1_partial147_20261031_v1`：固定 734 个原题、句法改写和问题复述。下表按 original / syntax / paraphrase 排列正确数。`raw S` 保留精确原值；展示值为 `capped S = min(raw S, 100)`。

| 模型 | 基模正确数 | cleared 正确数 | raw S | capped S |
|---|---:|---:|---:|---:|
| Qwen | 421 / 411 / 425 | 200 / 201 / 195 | 99.1994% | 99.1994% |
| LLaVA | 357 / 346 / 346 | 218 / 212 / 212 | 102.9286% | 100.0000% |
| InternVL | 405 / 405 / 406 | 220 / 208 / 227 | 96.3219% | 96.3219% |

S 参考线为 65%。清除率 C、逐场景计数、模型完整 revision、稀疏 overlay SHA256 和覆盖审计见 [`text/results/s_results_v2.json`](text/results/s_results_v2.json)。该文件只含汇总数，不含逐题分数。

此前 PKG3 的 S 数值仍保留在 [`text/results/s_results.json`](text/results/s_results.json)，对应旧版 734 题改写，可按 `pkg3_s2_v1` 重建；旧结果不由本增量改写。

## 数据与评分入口

固定上游 WMDP Parquet、两个冻结版本和评分 CLI 可读 manifest 的重建方法见 [`text/data/README.md`](text/data/README.md)。新版本是重建脚本默认值。模型 revision 与 overlay 标识在结果 JSON 中记录；模型权重和 overlay 不属于本增量。已发布模型下载地址见[模型说明](text/models/README.md)。

## 导航

- [完整 HF 评测命令、评分合同与 C/S 汇总](text/README.md)
- [外部数据获取、冻结改写和审核局限](text/data/README.md)
- [实际模型身份、revision 与稀疏更新 SHA256](text/models/README.md)
- [C 汇总](text/results/c_reference.json) 与 [新版 S 汇总](text/results/s_results_v2.json) 与 [历史 S 汇总](text/results/s_results.json)
- [代码与数据来源哈希](text/results/source_provenance.json)
- [第三方来源与许可](THIRD_PARTY_NOTICES.md)
- [三个已发布的完整模型](text/models/README.md#已发布的清除后模型)；本仓库不包含模型权重。

[`text/results/reference_results.json`](text/results/reference_results.json) 保留 PKG1 历史参考，不是本次 PKG3 的评分记录。
