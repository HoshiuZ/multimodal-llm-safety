# 外部数据接口

本轮只提供 schema 和良性合成测试，不提供真实题目、逐题 ID、答案、token IDs 或逐题分数。外部文件须由使用者从项目冻结材料或数据提供方准备；本包没有自动下载脚本。

清单 JSON 的字段如下；其中相对路径相对清单所在目录解析：

```json
{"schema":"text-eval-v1","kind":"C","source":"provider/frozen-version","scenes":{"original":"original.jsonl"}}
```

S 清单将 `kind` 改为 `S`，scenes 必须且只能包含 `original`、`syntax`、`paraphrase` 三个键，各自指向外部 JSONL。C 必须完整覆盖3668题（bio 1273、cyber 1987、chem 408）；S 每个场景必须734题（bio 255、cyber 397、chem 82）。缺失、额外、重复或场景不齐一律报错。

每行 JSON 必须含以下 C/S 字段：

| 字段 | 类型与约束 |
|---|---|
| question_id | 非空字符串；场景内唯一，S 三场景相同 ID 集 |
| source_row_sha256 | 原始冻结来源行的64位小写 SHA256；三场景同题一致 |
| domain | bio / cyber / chem |
| question | 非空字符串；评分前只去首尾空白 |
| choices | 四个非空字符串，顺序固定 |
| answer_index | 整数0至3，不接受布尔值 |

S 的题干允许变化，同题的来源 SHA、域、选项顺序及答案必须一致。实际评分输入字段全部纳入数据身份；其他历史元数据不参与评分。`source` 填实际来源与冻结版本，不能拿任意内容冒充历史数据。SHA 是来源绑定信息，并不证明内容经过专家审定。

项目原冻结 original 文件已含这些字段；装配改写文件按 `variant_type=syntax_adjustment` 与 `question_restatement` 分成 syntax/paraphrase，再保留表中字段即可。旧 `answer` 字段应由提供方明确转换成 `answer_index`，不能猜答案编号从0还是1开始。完整 C 来源同样须由提供方按冻结清单转换，保证域总数、source、ID和来源行 SHA。不要把训练题混入评测清单。

本包不附真实数据示例；测试中的颜色问题和 `synthetic-*` 标识仅供 CPU 工程测试，不可用于报告指标。输出 `base.jsonl`、`cleared.jsonl` 同样不得入库。

## other49 能力复核数据

`kind` 为 `M` 时，`scenes` 必须只有 `original`，文件必须正好含245行：固定49个非相关 MMLU subject，每科5题。排除 subject 为 `college_biology`、`high_school_biology`、`virology`、`college_chemistry`、`high_school_chemistry`、`computer_security`、`college_computer_science`、`high_school_computer_science`。manifest 的 `subjects` 必须列出且只列出文件中的49科。

M 行必需字段为 `question_id`、`source_row_sha256`、`subject`、`description`、`question`、`choices`、`answer_index`。`description` 使用冻结 MMLU 模板的官方 subject 描述。评分合同与 C 相同，选择题上下文按 `description + question + options + Answer:` 构造。

真实 MMLU 行、题目 ID 与逐题分数不得放进公开包。输出记录与 C 一样只留本地；公开结果只报告汇总正确数及整数门槛。
