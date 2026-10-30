# PKG2 核验记录

## 已完成

- `text/tests/test_text_eval.py` 的 CPU 合成测试通过，共15项。测试覆盖 C 聚合、80% 能力门槛整数边界、逐题覆盖、断点续评、overlay 加载和三种模型的文本 forward 分派。合成题不包含真实评测题目。
- paired audit 要求每条记录带有非空 SHA256 contract；可传入 expected contract 严格核对。它只接受单场景 C/M 数据包；S 数据必须走原有多场景聚合路径。
- Qwen、LLaVA-NeXT 与 InternVL3 的完整多模态 wrapper 均在 CPU 上恢复了对应稀疏参数；恢复参数逐 tensor bitwise 相等，分别涉及4、7、5个参数张量。
- 六个正式单卡评分作业均由调度器记录为 `Done successfully`。每个作业只申请一台主机和一张 `gpu_6000ada` 卡；运行时各自只看到一张逻辑 CUDA 设备，并记录设备 UUID。
- 所有配对结果均覆盖3668道 C 题和245道 other49 能力题；基模与清除态题目身份匹配，失败、缺失、重复和额外行均为0。基模 other49 的735条记录来自逐题审核通过的固定基线缓存；清除态评分735条。
- 逐模型计数和整数门槛见 [PKG2 聚合结果](text/results/pkg2_c_and_other49.json)。三个 C 都达到70%目标。按本批20%能力损失预算，Qwen 与 LLaVA 通过，InternVL 少1题，因此全模型门槛未通过。

## 复核命令

在 Linux 的 `text/` 目录使用对应模型的独立环境。worker 使用 `fcntl` 文件锁，要求运行环境提供 `fcntl`：

```bash
python -B -m unittest discover -s tests -v
python -B scripts/sparse_eval.py --help
```

真实模型、overlay、冻结数据和逐题分数不随本目录分发。重新评分须提供配置固定的完整基模、同一 overlay 和相同冻结清单，并为每种状态使用独立输出目录。运行命令见 [文本评分说明](text/README.md)。

历史聚合结果仅作参考，不能视为本轮评分。重新评分时保留每态的 preflight contract 文件；审计命令可同时传入 `--base-contract` 与 `--cleared-contract` 校验逐题身份。
