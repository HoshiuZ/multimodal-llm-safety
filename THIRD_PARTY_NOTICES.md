# 第三方来源与许可

本交付没有为整个包指定新的统一许可证。上游数据、模型和软件各自的许可继续适用；公开评测代码不授予重新分发模型文件的权利。

## 数据与方法来源

- **WMDP 数据与作者源码**：数据来自 [CAIS WMDP](https://huggingface.co/datasets/cais/wmdp/tree/7125571f22f032c56415e7980f48d877dd830ff8)，固定 revision 为 `7125571f22f032c56415e7980f48d877dd830ff8`。最小公开映射包含固定选择清单与衍生文本补丁；预制 C/S 全题目 JSONL 由固定上游 Parquet 重建，不随映射分发。上游许可副本为 [`licenses/WMDP-MIT.txt`](licenses/WMDP-MIT.txt)。作者源码来源为 [centerforaisafety/wmdp](https://github.com/centerforaisafety/wmdp)，固定提交 `c0b6c12bb0de3decf9c13bb13f9f1aa15754a132`；没有复制其训练 runner。
- **评测描述和选择题格式**：方法依据 [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/4600d6bf73ba2cf7037ae7feada03315839ef185)，固定提交 `4600d6bf73ba2cf7037ae7feada03315839ef185`（v0.4.2 来源）；许可副本为 [`licenses/LM-EVAL-MIT.txt`](licenses/LM-EVAL-MIT.txt)。官方领域描述按冻结的评测配置使用。

## 模型来源

本包不包含模型权重、tokenizer 文件或 sparse update。仓库和 revision 仅用于识别实际评测对象：

- [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b)：仓库声明 Apache-2.0。
- [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B/tree/853e3a797a661694b1b8ece0cb72dc2b23e3dac9)：仓库元数据标示 Apache-2.0，但模型卡正文标示 MIT；该上游信息不一致。本包不分发模型文件，也不替权利人消解许可冲突。
- [llava-hf/llama3-llava-next-8b-hf](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf/tree/b041c0d0ea0dd0196d147206c210c8d1752fc2da)：仓库声明 `llama3` 许可，Meta Llama 3 条款适用；不可将模型重新标为 MIT 或 Apache-2.0。

模型 revision、文件 SHA256 和身份限制见 [`text/models/README.md`](text/models/README.md)。使用者须自行遵守各模型仓库的访问条件和许可。

## 软件依赖与来源边界

PyTorch、Transformers、Accelerate、Hugging Face Hub、PyArrow 等由运行环境安装和提供；本包没有复制这些依赖的完整源码。

`text/eval/` 中的数据身份、schema、续评、聚合与 CLI 适配，以及 `text/scripts/`、测试和文档，按项目既有 S2/S3 实现做了最小提取与适配。评分及模型加载接口的来源 SHA 记录在随包结果材料中。本说明不额外主张对第三方模板的权利，也不为整个仓库选择许可证。

## 训练脚本方法来源

- **NPO 方法**：Zhang、Lin、Bai、Mei, [Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/abs/2404.05868)。此处引用论文方法，不复制或再分发论文之外的代码。
- **GROM 方法**：Batorski、Spurek、Swoboda, [GROM: Gradient-Free Rapid One-Shot Machine Unlearning](https://arxiv.org/abs/2608.05783)，作者实现见 [Batorskq/GROM](https://github.com/Batorskq/GROM)。上游实现为 MIT 许可；本仓库训练脚本是项目代码，未复制该实现，且不因此继承 MIT 许可。
- 本仓库新增的 `text/training/train_lib.py`、`text/training/train_llava.py`、`text/training/train_qwen.py`、`text/training/train_internvl.py` 没有单独许可证声明；仓库没有根级通用许可证。本次公开源码不应解释为授予复用或再分发权。
