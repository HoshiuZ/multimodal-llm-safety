# 三模型文本训练脚本

此目录说明四个训练脚本的用途、来源和运行边界。脚本组合了 GROM 闭式权重编辑、NPO 更新以及保留集交叉熵修复；这里记录的是现有实验代码，不改变仓库当前 RMU 研究范围说明，也不改写根 README 中的正式 C/S 汇总。

## 模型身份

代码从以下固定模型仓库版本载入完整多模态模型，再只更新所列语言侧权重：

| 脚本 | 模型仓库 | Revision | 上游许可标记 |
|---|---|---|---|
| `train_llava.py` | [`llava-hf/llama3-llava-next-8b-hf`](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf/tree/b041c0d0ea0dd0196d147206c210c8d1752fc2da) | `b041c0d0ea0dd0196d147206c210c8d1752fc2da` | `llama3` |
| `train_qwen.py` | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 |
| `train_internvl.py` | [`OpenGVLab/InternVL3-8B`](https://huggingface.co/OpenGVLab/InternVL3-8B/tree/853e3a797a661694b1b8ece0cb72dc2b23e3dac9) | `853e3a797a661694b1b8ece0cb72dc2b23e3dac9` | 上游仓库元数据标为 Apache-2.0，模型卡正文标为 MIT；两者不一致 |

LLaVA 脚本记录的 **2026-11-01 base-reference 实验**从未编辑的上述基模计算参考 log-prob，再执行 GROM + NPO + repair。其稀疏输出 SHA256 为 `dd9b6bb0c01c2a40437c89c127540e83c7bd6554b1892984e2273ed946ddd1da`，当前公开 checkpoint 的正式逐题复核结果为 `C=80.9288537549%`、新版文本 `S=94.9877984085%`；完整计数与覆盖审计见[版本化结果](../results/srev3_llava_base_ref_cs_20261101.json)。完整模型现已[公开发布](https://huggingface.co/HoshiuZ/LLaVA-NeXT-8B-Unlearned/tree/959a02037889f1f74cacfdc0152dc53444217563)（revision `959a02037889f1f74cacfdc0152dc53444217563`）。

InternVL 最终参数的五个张量对应第 4、5、6、7、9 层。训练脚本对第 5、7、9 层做 GROM 编辑，对第 4–7 层做一步 NPO；项目交付记录将早期 manifest 中的“10–13 层 NPO”标为笔误，并记载脚本产物与最终交付的五张量文件逐张量完全一致（最大绝对差为 0）。目标参数文件 SHA256 为 `1cb064d13f6370babc154197480a06705c38ebaf26a3baae8a89850419e287e8`。该参数文件不在 GitHub 仓库中；持有文件时可通过 `train_internvl.py --verify` 重新比较，失败将返回非零退出码。正式 C/S 数值仍以[文本评测结果](../README.md)为准。

Qwen 与 InternVL 脚本注释保留各自实验记录。脚本中的早期实验聚合值与正式评测采用不同记录，使用时请以相应 manifest 和结果报告判断模型、代码与输出对应关系。

## 方法来源

- NPO 目标参考 Zhang、Lin、Bai、Mei 的 [Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/abs/2404.05868)。脚本使用 NPO loss，并另加保留样本的交叉熵项。
- GROM 闭式编辑参考 Batorski、Spurek、Swoboda 的 [GROM: Gradient-Free Rapid One-Shot Machine Unlearning](https://arxiv.org/abs/2608.05783) 和[作者实现](https://github.com/Batorskq/GROM)。本项目脚本使用本地预先计算的 basis 执行编辑；没有把作者仓库的源码复制进这四个文件。

这些引用说明方法来源，不表示本项目脚本是作者代码的官方实现或完整复现。实际算法参数、数据版本和模型输出应以各实验报告及 manifest 为准。

## basis 下载与放置

从 [GitHub Release](https://github.com/HoshiuZ/multimodal-llm-safety/releases/tag/text-training-bases-20261102) 下载 `text-training-bases.zip.001` 和 `text-training-bases.zip.002`，两卷均小于 2 GiB。在仓库根目录按顺序合并并解压：

```bash
cat text-training-bases.zip.001 text-training-bases.zip.002 > text-training-bases.zip
unzip -n text-training-bases.zip -d .
```

拼接 ZIP 的 SHA256 为 `22fbcc48316c84e56c63db814c864d33e90f96810c60547531e065b48ef32b80`。各分卷及解压后文件的校验值见 Release 正文。归档内只有四个 `.pt`，解压后位于仓库的 `text/runs/bases/`，文件名与训练脚本默认路径一致。

## 输入与运行条件

脚本不能独立运行。本仓库增量不包含以下私有实验输入，也不分发它们：

- 完整基模权重及 tokenizer；
- GROM basis 文件（默认路径：`text/runs/bases/llava_grom_basis.pt`、`text/runs/bases/qwen_grom_basis.pt`、`text/runs/bases/qwen_grom_basis_cyber.pt`、`text/runs/bases/internvl_grom_basis_multi.pt`；可用脚本对应的 `TRAIN_*_GROM_BASIS` 环境变量覆盖）；
- 训练 JSONL、文本正文或 token IDs；
- 任何训练后检查点。

运行环境需要 Python、PyTorch、Transformers 和 CUDA。此源码检查所用 CPU 环境为 Python 3.12.14、PyTorch 2.8.0+cu128、Transformers 4.57.1；这仅表示语法与参数帮助可运行，不代表验证了训练环境兼容性。共享库通过 `TRAIN_ROOT` 指定包含 `models/`、`data/`、`runs/` 的本地根目录；未设置时使用仓库根目录。所有脚本都把设备固定为 CUDA。

以下命令在仓库根目录执行。只查看参数：

```bash
python text/training/train_llava.py --help
python text/training/train_qwen.py --help
python text/training/train_internvl.py --help
```

训练入口如下。实际执行前须由使用者准备与对应 manifest 一致的全部本地输入，并通过所在环境获批的 GPU 调度方式运行：

```bash
TRAIN_ROOT=/path/to/prepared/root python text/training/train_llava.py --out /path/to/output.pt
TRAIN_ROOT=/path/to/prepared/root python text/training/train_qwen.py --out /path/to/output.pt
TRAIN_ROOT=/path/to/prepared/root python text/training/train_internvl.py --out /path/to/output.pt
```

`--verify` 还需要本地参考 checkpoint。CPU 语法检查和 `--help` 不验证训练、输出一致性或评测结果。本增量没有重新训练或评分。

## 许可

仓库当前没有根级通用许可证，且这四个项目脚本没有单独许可证声明；公开源码不等于授予复用、修改或再分发的许可。GROM 作者仓库的 MIT 许可只适用于该上游代码，不会自动覆盖本项目脚本。模型权重继续受各模型上游许可约束。NPO 论文是方法引用，不是代码许可。请在获得权利人适用的许可后再复用这些脚本。

## 训练后的完整模型恢复

默认训练输出已由 `text/restore_paths.json` 连接至恢复入口。在 `text/` 目录运行对应的 `python restore_qwen.py`、`python restore_llava.py` 或 `python restore_internvl.py`。默认 `identity_mode=trained` 接受符合结构要求的新训练参数，并核验保存与重载前后的张量一致性；C/S 评分使用本地恢复记录中的实际文件哈希。发布模型固定哈希不作为新训练产物的验收条件。

## 默认输入路径与预检

训练和评分共用仓库下的 `text/models/qwen-base/`、`text/models/llava-base/`、`text/models/internvl-base/`，无需复制第二份基模。basis 位于 `text/runs/bases/`。

训练数据位于仓库根目录下的 `data/processed/<模型标识>/three_domain_rmu_20261024/`，模型标识分别为 `qwen3vl8b`、`llava_next8b`、`internvl3_8b`。每个目录需包含 `existing_bio_forget_train_seed42.jsonl`、`cyber_forget_train_seed42.jsonl`、`chem_forget_train_seed42.jsonl`、`existing_wikitext_retain_train_seed42_v2.jsonl`，每行须有对应模型的 `token_ids`。这些训练文件尚未包含在公开仓库或 basis Release 中，不能以 C/S 评测数据替代。仅克隆仓库并下载基模、basis 仍不足以启动训练；持有获准训练材料后才可运行。

可在 `text/` 下运行 `python training/check_inputs.py qwen`（另外两模型替换为 llava、internvl）。训练入口也会自动执行同一预检，缺项会列出具体路径。该检查不加载模型权重或使用 GPU。

恢复程序允许目标目录为空，或仅含仓库自带的 `README.md` / `.gitkeep` 占位文件；已有模型、其他文件或符号链接仍拒绝覆盖。

## 训练数据交付与准备辅助工具

测试方使用已获准交付的十二份训练 JSONL 时，直接按现有训练命令运行，无需增加数据准备命令。材料分别放在仓库根目录的 `data/processed/{qwen3vl8b,llava_next8b,internvl3_8b}/three_domain_rmu_20261024/`；每目录四份文件、每份 320 条。三套文本一致，但 tokenizer 不同，不能混用 token IDs。公开仓库不含受限训练正文或 token IDs；材料应单独交给具有相应访问权限的测试方。

`prepare_data.py` 与 `_pipeline/` 是供交付方重新整理材料的 CPU 辅助工具，未改变训练、恢复或 C/S 评分命令。它们不训练、不评分、不加载模型权重，不下载数据或模型。来源及固定 revision 见 [source_manifest.json](source_manifest.json)。

辅助工具需 Python 3.12、torch 2.8.0、transformers 4.57.1、pyarrow 26.0.0。交付方应准备两个目录：`--data-root` 为独立输出根目录，含固定 C 题干清单 `data/eval/text_cs_evolution_20261023/c_full.jsonl` 与 `text/models/{qwen-base,llava-base,internvl-base}` 的固定 tokenizer；`--source-dir` 为获准输入根目录，含配置列出的 Cyber/Chem Parquet、既有 Bio/WikiText 片段、逐模型 token 文件及其先前 manifest。两个目录须在同一文件系统，Bio/WikiText 复用通过硬链接验证。

交付方在仓库根目录执行：

```bash
python text/training/prepare_data.py prepare-all --data-root /path/to/fresh-output-tree --source-dir /path/to/authorized-source-tree
```

该命令按固定顺序和 seed 42 选择 Cyber/Chem 片段，核对与 3,668 条 C 题干的精确重叠，分别编码三个 tokenizer，复用 Bio/WikiText，验证并生成冻结 manifest。现有获准材料上的独立 CPU 重建，十二份输出均与原冻结训练输入逐字节 SHA256 相同；Bio/WikiText 的原始 Parquet 重新选样不属于这次验证范围。见 [准备验证摘要](preparation_verification.json)。

Bio 来源有访问限制，本工具不绕过授权、不自动下载 Bio。Chem 使用通用化学代理语料，其数据集卡声明 CC BY-NC-SA 4.0，但逐页面许可未逐项核验；此处不授予其原文或 tokenized 输出的公开再分发权。项目源码保留现有许可状态。
