# 固定数据取得与重建

重建脚本从 [CAIS WMDP](https://huggingface.co/datasets/cais/wmdp/tree/7125571f22f032c56415e7980f48d877dd830ff8) 固定 revision `7125571f22f032c56415e7980f48d877dd830ff8` 读取三个官方 test Parquet，核验 SHA256、字节数和行数，再生成完整 C 与固定 734 题的三场景 S。公开增量不附 Parquet、预制 C/S 题目 JSONL、选项或答案。WMDP 许可副本保留在 [`../../licenses/WMDP-MIT.txt`](../../licenses/WMDP-MIT.txt)。

## 默认版本：新版 S

`frozen/srev1_partial147_v1/` 保存 147 题选择清单、syntax 与 paraphrase 各一份完整 734 行补丁，以及 `manifest.json` 和预期 `source_manifest.json`。未选中的 587 行逐行、逐字节沿用旧补丁；重建入口还会现场复核此条件。S 的 original 仍使用原固定 734 题选择。

在包根目录运行，输出目录必须尚不存在：

```bash
python -B text/data/prepare_eval_data.py \
  --download \
  --dataset-version srev1_partial147_20261031_v1 \
  --output-dir /tmp/srev2-data
```

`--dataset-version` 默认就是 `srev1_partial147_20261031_v1`。如已准备好固定 Parquet 的离线目录，可将 `--download` 换为：

```bash
--source-cache /path/to/wmdp-cache
```

目录中须有 `wmdp-bio/test-00000-of-00001.parquet`、`wmdp-cyber/test-00000-of-00001.parquet` 和 `wmdp-chem/test-00000-of-00001.parquet`。脚本会与新版 `source_manifest.json` 核对官方文件及重建 C/S 的哈希，并生成评分 CLI 可读的 `C/manifest.json`、`S/manifest.json`。也可按下列方式校验 manifest：

```bash
python -B text/scripts/text_eval.py validate --data /tmp/srev2-data/C/manifest.json
python -B text/scripts/text_eval.py validate --data /tmp/srev2-data/S/manifest.json
```

## 历史 PKG3 版本

原始 PKG3 S 选择、两份旧 734 行补丁和 manifest 继续保留在 `frozen/`。如需重建旧版，显式使用：

```bash
python -B text/data/prepare_eval_data.py \
  --download \
  --dataset-version pkg3_s2_v1 \
  --output-dir /tmp/pkg3-s2-data
```

对应旧评分聚合仍保留在 [`../results/s_results.json`](../results/s_results.json)；新版汇总单独放在 [`../results/s_results_v2.json`](../results/s_results_v2.json)。

## 新版选择与改写审核范围

147 项选择按旧 S 顺序抽取，领域数为 bio 51、cyber 80、chem 16；它们来自旧固定 734 题选择。syntax 和 paraphrase 补丁各有 147 行更新、587 行保持旧行原字节。清单与补丁只保存题干文本、ID、领域和来源行哈希，不复制选项、答案索引或逐题评分。

审核由 AI 完成，未获人工或领域专家签核。审核检查题干表达及其对冻结答案选项的适用性，没有独立求解每个代码或数据包题目；少数源题歧义仍保留审核注记。生成种子未留存；生成过程未提供选项、答案标签或模型分数。冻结文件用于复现已接受的文本，不据此声称语义等价或专家确认。

Python 依赖包括 `pyarrow` 和 `huggingface_hub`；按所用模型安装对应 `text/requirements-*.txt`。仅执行数据重建和 `validate` 不加载模型，也不进行模型前向评分。
