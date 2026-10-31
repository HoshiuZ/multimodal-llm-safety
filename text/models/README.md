# 模型身份与权重说明

本次 S 评测使用以下完整多模态 Hugging Face 检查点及固定 revision。评测输入为纯文本；加载的仍是完整 wrapper，不以语言侧祖先模型替代。

| 模型 | 仓库与 revision | 加载类 | 本地 sparse update SHA256（字节） |
|---|---|---|---|
| Qwen | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) | `Qwen3VLForConditionalGeneration` | `53b2ed3e8b9b3a21884d48ec9c784de67d6b6b9d4f6e0c912ff4326833f30d68` (402,656,007) |
| InternVL | [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B/tree/853e3a797a661694b1b8ece0cb72dc2b23e3dac9) | `InternVLChatModel` | `1cb064d13f6370babc154197480a06705c38ebaf26a3baae8a89850419e287e8` (678,955,986) |
| LLaVA | [llava-hf/llama3-llava-next-8b-hf](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf/tree/b041c0d0ea0dd0196d147206c210c8d1752fc2da) | `LlavaNextForConditionalGeneration` | `627e161056ed640837eb007debe17ea225c23c781e5d154971b38a53822592f2` (822,087,200) |

每个模型的完整运行身份、精确 overlay 文件名和检查合同见 `../configs/*.json` 及随包的 `../results` 结果材料。Overlay 是项目本地文件，本包不含模型权重或 overlay，也没有可确认的公开下载地址。评测前应核验本地文件的 SHA256、张量键、shape、dtype 和完整 wrapper 恢复结果。

本次对 LLaVA 实际评分仓库和 revision 的身份记录是确定的；它与方案中的正式 LLaVA 型号名称是否对应，现有材料仍未说明。InternVL 的实际 overlay 有 5 个参数键，层为 4、5、6、7、9；训练 manifest 记载的 NPO 阶段指向层 10–13。现有材料不能确认该 overlay 与 NPO 阶段的具体来源关系，因此结果只归属于此处 SHA 固定的 overlay。

模型许可由上游模型仓库适用条款决定，见 [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)。本说明不为模型文件重新指定许可。
