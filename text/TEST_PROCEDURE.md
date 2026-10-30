# 文本测试步骤

所有命令在 `text/` 目录执行，使用对应模型的独立 Python 环境。

```bash
python -B scripts/text_eval.py --help
python -B scripts/text_eval.py validate --help
python -B scripts/text_eval.py model-preflight --help
python -B scripts/text_eval.py preflight --help
python -B scripts/text_eval.py run --help
python -B scripts/text_eval.py aggregate --help
python -B -m unittest discover -s tests -v
```

CPU 测试包括：S2/S3 三模型不同场景分母及完整 C 的整数聚合回归；负 C、负 S、超过100和 N/A；JSON键序列化；缺失、重复、额外、失败行拒绝；实际 worker 写出、断点续评和错模型/代码/prompt/数据身份拒绝；多 token 延续、最后位置、padding mask、无截断及三模型 forward 分派。模型操作由小型 CPU fixture 注入，不含真实题目，不生成研究分数。

外部材料到齐后，按 README 的顺序执行 validate、model-preflight、preflight、两态 run 和 aggregate。为每个模型、C/S 分别使用独立 preflight 和输出目录。先核对提供方模型与数据版本，再按调度平台申请单卡执行；本包不负责提交 GPU 作业。

真实模型验收应检查两态完整 coverage、零失败/缺失/重复/额外、实际完整 wrapper 类、BF16、attention 路径、同一 tokenizer/prompt、模型内容身份和聚合输出。S 默认门槛65，旧82可配置复核；C目标由实验方案另行判断。不得将随包历史聚合 JSON 复制为本次评分输出。

缺少外部文件时，用 README 命令将得到明确 FileNotFoundError；这表示材料未提供，不是模型效果不合格。PKG1 的 CPU 验证不替代这里列出的真实三模型推理验收。
