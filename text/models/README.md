# 已发布模型与评测版本

本页列出三个当前交付的完整模型。模型文件在 Hugging Face，GitHub 仓库提供评测代码、配置和汇总结果。下表的 C/S 均为纯文本输入评测；加载时使用完整多模态模型。

| 模型 | 完整模型下载 | 固定基模 | 文本 C（3,668 题） | 文本 S（每场景 734 题） |
|---|---|---|---:|---:|
| Qwen3-VL-8B | [HoshiuZ/Qwen3-VL-8B-Unlearned](https://huggingface.co/HoshiuZ/Qwen3-VL-8B-Unlearned/tree/03eadaed34a3ffba86452f2c8ec42ede6bd686e8) | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) | 88.4871% | 99.1994% |
| LLaVA-NeXT-8B | [HoshiuZ/LLaVA-NeXT-8B-Unlearned](https://huggingface.co/HoshiuZ/LLaVA-NeXT-8B-Unlearned/tree/959a02037889f1f74cacfdc0152dc53444217563) | [llava-hf/llama3-llava-next-8b-hf](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf/tree/b041c0d0ea0dd0196d147206c210c8d1752fc2da) | 80.9289% | 94.9878% |
| InternVL3-8B | [HoshiuZ/InternVL3-8B-Unlearned](https://huggingface.co/HoshiuZ/InternVL3-8B-Unlearned/tree/59f6b03f8a0c0b5c7ae22d4867e3dfdb9fc0f818) | [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B/tree/853e3a797a661694b1b8ece0cb72dc2b23e3dac9) | 76.5924% | 96.3219% |

LLaVA 使用未编辑的基模作为 NPO 参考。上表 LLaVA 结果对应公开模型 revision `959a02037889f1f74cacfdc0152dc53444217563`，正式逐题复核的 C 为 `80.9288537549%`，S 为 `94.9877984085%`。其完整计数、评分身份与覆盖审计见 [LLaVA 版本化结果](../results/srev3_llava_base_ref_cs_20261101.json)。

Qwen 与 InternVL 的 C 汇总见 [C 结果](../results/c_reference.json)，新版 S 汇总见 [S 结果](../results/s_results_v2.json)。S 展示值按 `min(raw S, 100%)` 计算，同时保留原始值；评分合同、数据版本和运行方法见[文本评测说明](../README.md)。

## InternVL 最终参数的训练来源

当前交付参数的 SHA256 为 `1cb064d13f6370babc154197480a06705c38ebaf26a3baae8a89850419e287e8`，包含第 4、5、6、7、9 层的五个张量。[训练脚本](../training/train_internvl.py)先对第 5、7、9 层做 GROM 编辑，再以未编辑基模为参考对第 4–7 层执行一步 NPO，最后保存上述五层。项目交付记录将早期 manifest 中的“NPO 作用于 10–13 层”标为笔误，并记载脚本产物与交付的五张量文件逐张量完全一致（最大绝对差为 0）。因此，本页按第 4–7 层 NPO 描述最终参数的来源；该复现结论来自交付记录，仓库未分发用于重新比较的参数文件。

脚本的 `--verify` 可在持有目标参数文件的环境中逐张量比较，比较失败时返回非零退出码。正式文本 C/S 结果以[评分文件](../results/c_reference.json)和[新版 S 文件](../results/s_results_v2.json)为准。

三个 Hugging Face 仓库提供完整模型文件。评测代码仓库不包含权重、稀疏更新文件或训练数据。模型文件的适用许可见各上游模型仓库和[第三方说明](../../THIRD_PARTY_NOTICES.md)。
