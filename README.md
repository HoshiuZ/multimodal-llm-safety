# 多模态模型有害知识清除：文本评测与数据重建

本目录保留三款完整多模态模型的文本评测代码和结果导航：PKG2 的 WMDP C 汇总，以及 PKG3 的固定样本 S 汇总。输入均为纯文本；这些结果不构成图像或其他多模态能力评测。

## 现有 C 结果

以下结果来自 PKG2 的 3,668 道 WMDP C 评测：

| 模型 | C |
|---|---:|
| Qwen | 88.4871% |
| LLaVA | 81.5217% |
| InternVL | 76.5924% |

计数、精确分数和对应模型身份见 [`text/results/c_reference.json`](text/results/c_reference.json)。

## 本次 S 结果

每场景 734 题，正确数按原题 / 句法改写 / 问题复述排列。S 使用未截断的精确分数；阈值参考为 65%。

| 模型 | 基模正确数 | cleared 正确数 | S |
|---|---:|---:|---:|
| Qwen | 421 / 410 / 422 | 200 / 199 / 195 | 100.1119% |
| InternVL | 405 / 405 / 403 | 220 / 210 / 226 | 96.5474% |
| LLaVA | 357 / 350 / 350 | 218 / 215 / 214 | 101.2055% |

完整聚合与覆盖审计见 [`text/results/s_results.json`](text/results/s_results.json)。

## 数据取得

公开映射包含固定 WMDP revision、哈希清单、S 选择清单及两套冻结改写补丁；预制的 C/S 题目 JSONL 不属于这个最小公开映射。第一次评分前先从固定上游数据重建 C/S 清单：

```bash
python -B text/data/prepare_eval_data.py \
  --download \
  --freeze-dir text/data/frozen \
  --expected-source-manifest text/data/frozen/prepared/source_manifest.json \
  --output-dir /tmp/pkg3-data
```

脚本校验固定 Parquet 的 SHA256 和行数、冻结 manifest 与改写补丁，并核对重建 C/S JSONL 的固定哈希。若使用本地 Hub 缓存或离线源缓存，请参阅 [数据取得与重建说明](text/data/README.md)。完成后再把 `/tmp/pkg3-data/C/manifest.json` 或 `S/manifest.json` 交给评分 CLI。

## 导航

- [完整 HF 评测命令、评分合同与 C/S 汇总](text/README.md)
- [外部数据获取、冻结改写和审核局限](text/data/README.md)
- [实际模型身份、revision 与稀疏更新 SHA256](text/models/README.md)
- [C 汇总](text/results/c_reference.json) 与 [S 汇总](text/results/s_results.json)
- [代码与数据来源哈希](text/results/source_provenance.json)
- [第三方来源与许可](THIRD_PARTY_NOTICES.md)
- 完整模型将另行发布到 Hugging Face，核验完成后补充下载链接；本仓库不包含模型权重。

[`text/results/reference_results.json`](text/results/reference_results.json) 保留 PKG1 历史参考，不是本次 PKG3 的评分记录。
