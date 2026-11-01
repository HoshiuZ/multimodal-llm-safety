# 三模型文本训练脚本

此目录说明四个训练脚本的用途、来源和运行边界。脚本组合了 GROM 闭式权重编辑、NPO 更新以及保留集交叉熵修复；这里记录的是现有实验代码，不改变仓库当前 RMU 研究范围说明，也不改写根 README 中的正式 C/S 汇总。

## 模型身份

代码从以下固定模型仓库版本载入完整多模态模型，再只更新所列语言侧权重：

| 脚本 | 模型仓库 | Revision | 上游许可标记 |
|---|---|---|---|
| `train_llava.py` | [`llava-hf/llama3-llava-next-8b-hf`](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf/tree/b041c0d0ea0dd0196d147206c210c8d1752fc2da) | `b041c0d0ea0dd0196d147206c210c8d1752fc2da` | `llama3` |
| `train_qwen.py` | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 |
| `train_internvl.py` | [`OpenGVLab/InternVL3-8B`](https://huggingface.co/OpenGVLab/InternVL3-8B/tree/853e3a797a661694b1b8ece0cb72dc2b23e3dac9) | `853e3a797a661694b1b8ece0cb72dc2b23e3dac9` | 上游仓库元数据标为 Apache-2.0，模型卡正文标为 MIT；两者不一致 |

LLaVA 脚本记录的 **2026-11-01 base-reference 实验**从未编辑的上述基模计算参考 log-prob，再执行 GROM + NPO + repair。其稀疏输出 SHA256 为 `dd9b6bb0c01c2a40437c89c127540e83c7bd6554b1892984e2273ed946ddd1da`，对应实验记录为 WMDP 三域 `C=80.4348%`、other245 `131/245`、MMLU285 `145/285`。权重尚未公开；本次只准备源码，没有下载地址。

它与仓库现有 HF J8 LLaVA 发布模型是两个不同的 cleared overlay：后者 SHA256 为 `627e161056ed640837eb007debe17ea225c23c781e5d154971b38a53822592f2`，正式 WMDP C 为 `81.5217%`（`1104/3668`）。两个输出都使用相同的固定 LLaVA 基模仓库版本，但训练参考和 overlay 不同。新实验记录不替换根 README、C/S 结果文件或模型说明中的已有值。

Qwen 与 InternVL 脚本注释保留各自实验记录。它们的数值不是根 README 当前 PKG2 C/S 汇总的替代品；使用时请以相应 manifest 和结果报告判断模型、代码与输出对应关系。

## 方法来源

- NPO 目标参考 Zhang、Lin、Bai、Mei 的 [Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/abs/2404.05868)。脚本使用 NPO loss，并另加保留样本的交叉熵项。
- GROM 闭式编辑参考 Batorski、Spurek、Swoboda 的 [GROM: Gradient-Free Rapid One-Shot Machine Unlearning](https://arxiv.org/abs/2608.05783) 和[作者实现](https://github.com/Batorskq/GROM)。本项目脚本使用本地预先计算的 basis 执行编辑；没有把作者仓库的源码复制进这四个文件。

这些引用说明方法来源，不表示本项目脚本是作者代码的官方实现或完整复现。实际算法参数、数据版本和模型输出应以各实验报告及 manifest 为准。

## 输入与运行条件

脚本不能独立运行。本仓库增量不包含以下私有实验输入，也不分发它们：

- 完整基模权重及 tokenizer；
- GROM basis 文件（默认路径：`runs/bases/llava_grom_basis.pt`、`runs/bases/qwen_grom_basis.pt`、`runs/bases/qwen_grom_basis_cyber.pt`、`runs/bases/internvl_grom_basis_multi.pt`；可用脚本对应的 `DSH_*_GROM_BASIS` 环境变量覆盖）；
- 训练 JSONL、文本正文或 token IDs；
- 任何训练后检查点。

运行环境需要 Python、PyTorch、Transformers 和 CUDA。此源码检查所用 CPU 环境为 Python 3.12.14、PyTorch 2.8.0+cu128、Transformers 4.57.1；这仅表示语法与参数帮助可运行，不代表验证了训练环境兼容性。共享库通过 `DSH_ROOT` 指定包含 `models/`、`data/`、`runs/` 的本地根目录；未设置时使用仓库根目录。所有脚本都把设备固定为 CUDA。

只查看参数：

```bash
python training/train_llava.py --help
python training/train_qwen.py --help
python training/train_internvl.py --help
```

训练入口如下。实际执行前须由使用者准备与对应 manifest 一致的全部本地输入，并通过所在环境获批的 GPU 调度方式运行：

```bash
DSH_ROOT=/path/to/prepared/root python training/train_llava.py --out /path/to/output.pt
DSH_ROOT=/path/to/prepared/root python training/train_qwen.py --out /path/to/output.pt
DSH_ROOT=/path/to/prepared/root python training/train_internvl.py --out /path/to/output.pt
```

`--verify` 还需要本地参考 checkpoint。CPU 语法检查和 `--help` 不验证训练、输出一致性或评测结果。本增量没有重新训练或评分。

## 许可

仓库当前没有根级通用许可证，且这四个项目脚本没有单独许可证声明；公开源码不等于授予复用、修改或再分发的许可。GROM 作者仓库的 MIT 许可只适用于该上游代码，不会自动覆盖本项目脚本。模型权重继续受各模型上游许可约束。NPO 论文是方法引用，不是代码许可。请在获得权利人适用的许可后再复用这些脚本。
