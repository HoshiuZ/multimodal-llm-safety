# 文本测试步骤

所有命令在 Linux 的 `text/` 目录执行，使用配置对应的独立 Python 环境。worker 使用 `fcntl` 文件锁，运行环境必须提供 `fcntl`。

```bash
python -B scripts/sparse_eval.py --help
python -B scripts/sparse_eval.py validate --help
python -B scripts/sparse_eval.py preflight --help
python -B scripts/sparse_eval.py restore-check --help
python -B scripts/sparse_eval.py run --help
python -B scripts/sparse_eval.py audit --help
python -B -m unittest discover -s tests -v
```

CPU 合成测试覆盖整数指标、20%能力损失门槛边界、逐题contract缺失/不匹配与配对不一致拒绝、S数据包拒绝、缺失/重复/额外/失败行拒绝、续评身份校验、逐 token LLsum、padding mask、稀疏 overlay 安全恢复和三种模型的文本 forward 分派。合成 fixture 不包含真实题目，也不生成研究分数。

真实模型复核已针对本轮完整多模态 wrapper 执行：三款模型的 CPU overlay restore-check 均逐 tensor 通过；六个正式单卡评分作业均成功完成。三组 C 与 other49 配对审计覆盖全部3668题和245题，零失败、缺失、重复或额外行。逐模型计数见 [`results/pkg2_c_and_other49.json`](results/pkg2_c_and_other49.json)。

本批 C 目标为70%；other49 能力损失预算为20%，以整数式 `5*n_cleared >= 4*n_base` 判定。当前三个 C 均通过；InternVL other49 为146/245，最低要求147，因此本批三模型整体门槛未通过。

复评分需要外部提供完整基模、对应稀疏 overlay、冻结数据清单及独立环境；这些内容不会随本目录分发。审计时可同时使用 `--base-contract` 和 `--cleared-contract`，按每条评分记录核对 preflight contract。命令说明和可复现顺序见 [README](README.md)。逐题输出应保留在本地实验目录。
