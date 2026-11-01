# 模型身份与权重说明

以下为已发表 C/S 评测的模型身份及固定 revision；其中 LLaVA 是历史 HF1 版本，当前 HF `main` 已更新为 2026-11-01 基模参考版。评测输入为纯文本；加载的仍是完整 wrapper，不以语言侧祖先模型替代。

| 模型 | 仓库与 revision | 加载类 | 本地 sparse update SHA256（字节） |
|---|---|---|---|
| Qwen | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) | `Qwen3VLForConditionalGeneration` | `53b2ed3e8b9b3a21884d48ec9c784de67d6b6b9d4f6e0c912ff4326833f30d68` (402,656,007) |
| InternVL | [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B/tree/853e3a797a661694b1b8ece0cb72dc2b23e3dac9) | `InternVLChatModel` | `1cb064d13f6370babc154197480a06705c38ebaf26a3baae8a89850419e287e8` (678,955,986) |
| LLaVA（旧版 HF1） | [llava-hf/llama3-llava-next-8b-hf](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf/tree/b041c0d0ea0dd0196d147206c210c8d1752fc2da) | `LlavaNextForConditionalGeneration` | `627e161056ed640837eb007debe17ea225c23c781e5d154971b38a53822592f2` (822,087,200) |

每个模型的完整运行身份、精确 overlay 文件名和检查合同见 `../configs/*.json` 及随包的 `../results` 结果材料。Overlay 是项目本地文件，本仓库不含模型权重或 overlay；已发布的完整清除后模型见下方。评测前应核验本地文件的 SHA256、张量键、shape、dtype 和完整 wrapper 恢复结果。

本次对 LLaVA 实际评分仓库和 revision 的身份记录是确定的；它与方案中的正式 LLaVA 型号名称是否对应，现有材料仍未说明。InternVL 的实际 overlay 有 5 个参数键，层为 4、5、6、7、9；训练 manifest 记载的 NPO 阶段指向层 10–13。现有材料不能确认该 overlay 与 NPO 阶段的具体来源关系，因此结果只归属于此处 SHA 固定的 overlay。

模型许可由上游模型仓库适用条款决定，见 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)。本说明不为模型文件重新指定许可。

## 已发布的清除后模型

- [Qwen 最终模型](https://huggingface.co/HoshiuZ/Qwen3-VL-8B-Unlearned/tree/03eadaed34a3ffba86452f2c8ec42ede6bd686e8)（revision `03eadaed34a3ffba86452f2c8ec42ede6bd686e8`）
- [LLaVA 当前模型（基模参考版）](https://huggingface.co/HoshiuZ/LLaVA-NeXT-8B-Unlearned/tree/959a02037889f1f74cacfdc0152dc53444217563)（revision `959a02037889f1f74cacfdc0152dc53444217563`，overlay SHA256 `dd9b6bb0c01c2a40437c89c127540e83c7bd6554b1892984e2273ed946ddd1da`；独立实验文本 C `80.4348%`，S 尚未评测）
- [LLaVA 历史 HF1 模型](https://huggingface.co/HoshiuZ/LLaVA-NeXT-8B-Unlearned/tree/hf1-2026-10-31)（revision `1c1b8d10c7ee3d4b364c1cd0cdd0316605eb5aae`；本仓库现有 LLaVA C/S 汇总对应此版本）
- [InternVL 最终模型](https://huggingface.co/HoshiuZ/InternVL3-8B-Unlearned/tree/59f6b03f8a0c0b5c7ae22d4867e3dfdb9fc0f818)（revision `59f6b03f8a0c0b5c7ae22d4867e3dfdb9fc0f818`）

各仓库提供完整模型文件及上游许可；各项文本 C/S 结果及适用的模型版本见 [评测说明](../README.md)。
