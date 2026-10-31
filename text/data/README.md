# 固定数据取得与重建

最小公开映射提供固定 S 选择清单、两套冻结改写补丁、manifest、重建脚本及许可说明；不包含预制 C/S 题目 JSONL。原题、选项和答案从固定 revision 的官方 WMDP Parquet 重建。数据源为 [CAIS WMDP](https://huggingface.co/datasets/cais/wmdp/tree/7125571f22f032c56415e7980f48d877dd830ff8)，revision `7125571f22f032c56415e7980f48d877dd830ff8`，许可副本见 [`../../licenses/WMDP-MIT.txt`](../../licenses/WMDP-MIT.txt)。

随包的 [`frozen/prepared/source_manifest.json`](frozen/prepared/source_manifest.json) 是紧凑的预期输出哈希清单，不是题目文件。它记录三个上游 Parquet 的 SHA256/大小、生成的 C/S JSONL SHA256、固定选择/改写哈希及此前通过的字段顺序核验。`frozen/prepared/C/manifest.json` 和 `S/manifest.json` 记录输出 schema；生成题目清单前请先运行以下重建命令。

## 下载并生成评分清单

在包根目录执行，输出目录必须尚不存在：

```bash
python -B text/data/prepare_eval_data.py \
  --download \
  --freeze-dir text/data/frozen \
  --expected-source-manifest text/data/frozen/prepared/source_manifest.json \
  --output-dir /tmp/pkg3-data
```

若使用本地缓存，将 `--download` 换为 `--source-cache /path/to/wmdp-cache`。缓存中需有 `wmdp-bio/test-00000-of-00001.parquet`、`wmdp-cyber/test-00000-of-00001.parquet` 和 `wmdp-chem/test-00000-of-00001.parquet`。`--expected-source-manifest` 在固定目录存在时会由脚本自动选用，也可像示例一样明确提供。

脚本需要 `huggingface_hub` 与 `pyarrow==26.0.0`；三个模型环境的完整依赖列在 `text/requirements-*.txt`。下载或离线读取前会验证固定上游文件 SHA256/大小/行数、选择和补丁哈希、冻结 manifest、ID/来源行绑定及 255/397/82 领域分层。随后会用参考 source manifest 校验重建 C 与三个 S JSONL 的精确哈希。若你另有完整预制清单，可额外提供 `--expected-c-jsonl` 和 `--expected-s-dir` 做逐字段、逐行比较。

成功后，评分清单位于 `/tmp/pkg3-data/C/manifest.json` 和 `/tmp/pkg3-data/S/manifest.json`。使用评分 CLI 前先验证生成包：

```bash
python -B text/scripts/text_eval.py validate --data /tmp/pkg3-data/C/manifest.json
python -B text/scripts/text_eval.py validate --data /tmp/pkg3-data/S/manifest.json
```

## 冻结内容与审核范围

- `frozen/s2_selection_734.jsonl` 固定了 734 个 ID、来源行哈希和顺序；报告的抽样 seed 为 42。
- `frozen/s2_syntax_questions.jsonl` 与 `frozen/s2_paraphrase_questions.jsonl` 保存同一批 ID 的已接受改写，不会在重建时重新采样或生成。
- `frozen/s2_frozen_manifest.json` 绑定源版本、选择和两份补丁哈希。历史改写生成 seed 未保留。
- 改写曾经 AI 独立审核，但审核者能看到选项和答案；审核非盲审，也没有人工或领域专家签核。

冻结文本的 CPU 字符串比较确认：syntax 和 paraphrase 各有 734/734 题在词语 token 序列层面发生变化，选项及答案索引均逐题保持相同。syntax 改写的平均词序列编辑比约 34.91%，平均题干长度为原题的 96.51%；paraphrase 分别约 43.78% 和 96.33%。样本检查显示部分 syntax 改写较浅。词语变化和长度比不能证明语义保持、答案正确或改写质量；这些统计也不是人工领域审查结论。

完整 WMDP C 重建含 3,668 行：bio 1,273、cyber 1,987、chem 408。S 三场景各 734 行，固定分层分别为 bio 255、cyber 397、chem 82。每行由源问题、四个选项和 answer index 组成；三场景同 ID 的选项和答案相同。
