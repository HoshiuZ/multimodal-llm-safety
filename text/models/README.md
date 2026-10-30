# 用户自备完整 HF 模型

基模身份由三个 `configs/*.json` 记录，来源于实际模型 manifest：

| 模型 | 基模 repository | revision |
|---|---|---|
| Qwen | Qwen/Qwen3-VL-8B-Instruct | 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b |
| InternVL | OpenGVLab/InternVL3-8B | 853e3a797a661694b1b8ece0cb72dc2b23e3dac9 |
| LLaVA | llava-hf/llama3-llava-next-8b-hf | b041c0d0ea0dd0196d147206c210c8d1752fc2da |

运行时接受用户指定的本地完整 HF `save_pretrained` 目录，必须保留完整多模态 wrapper 权重、配置、tokenizer；InternVL 还需原始本地模型代码。清除模型应由模型提供方提前导出完整 HF 目录，本包不转换旧 overlay，不附控制向量、编辑方向或训练代码。权重缺失或键/shape不匹配严格失败，不能用忽略缺键的方式继续运行。

配置中的 repository/revision 是历史基模来源记录；运行时以实际目录的内容 SHA 绑定模型，并不能凭 repository 字符串证明使用者自备文件确实来自对应远端 revision。正式复现须由提供方核验基模来源、导出过程和清除态对应关系，再运行完整模型复验。历史 overlay SHA 只用于区分旧参考检查点，不能当作完整 HF 导出的 SHA。

最终清除模型的 Hugging Face 链接待后续提供。本包不含模型下载地址、权重、自动下载逻辑或凭据。原模型的许可与访问条件仍适用，详见 THIRD_PARTY_NOTICES。
