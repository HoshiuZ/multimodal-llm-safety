# 已发布模型与评测版本

本页列出三个当前交付的完整模型。模型文件在 Hugging Face，GitHub 仓库提供评测代码、配置和汇总结果。下表的 C/S 均为纯文本输入评测；加载时使用完整多模态模型。

| 模型 | 完整模型下载 | 固定基模 | 文本 C（3,668 题） | 文本 S（每场景 734 题） |
|---|---|---|---:|---:|
| Qwen3-VL-8B | [HoshiuZ/Qwen3-VL-8B-Unlearned](https://huggingface.co/HoshiuZ/Qwen3-VL-8B-Unlearned/tree/03eadaed34a3ffba86452f2c8ec42ede6bd686e8) | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) | 88.4871% | 99.1994% |
| LLaVA-NeXT-8B | [HoshiuZ/LLaVA-NeXT-8B-Unlearned](https://huggingface.co/HoshiuZ/LLaVA-NeXT-8B-Unlearned/tree/959a02037889f1f74cacfdc0152dc53444217563) | [llava-hf/llama3-llava-next-8b-hf](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf/tree/b041c0d0ea0dd0196d147206c210c8d1752fc2da) | 80.9289% | 94.9878% |
| InternVL3-8B | [HoshiuZ/InternVL3-8B-Unlearned](https://huggingface.co/HoshiuZ/InternVL3-8B-Unlearned/tree/59f6b03f8a0c0b5c7ae22d4867e3dfdb9fc0f818) | [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B/tree/853e3a797a661694b1b8ece0cb72dc2b23e3dac9) | 76.5924% | 96.3219% |

LLaVA 使用未编辑的基模作为 NPO 参考。上表 LLaVA 结果对应公开模型 revision `959a02037889f1f74cacfdc0152dc53444217563`，正式逐题复核的 C 为 `80.9288537549%`，S 为 `94.9877984085%`。其完整计数、评分身份与覆盖审计见 [LLaVA 版本化结果](../results/srev3_llava_base_ref_cs_20261101.json)。

Qwen 与 InternVL 的 C 汇总见 [C 结果](../results/c_reference.json)，新版 S 汇总见 [S 结果](../results/s_results_v2.json)。S 展示值按 `min(raw S, 100%)` 计算，同时保留原始值；评分合同、数据版本和运行方法见[文本评测说明](../README.md)。

三个 Hugging Face 仓库提供完整模型文件。评测代码仓库不包含权重、稀疏更新文件或训练数据。模型文件的适用许可见各上游模型仓库和[第三方说明](../../THIRD_PARTY_NOTICES.md)。
