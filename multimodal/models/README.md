# 原始基模与 M1 适配器

所有原始多模态基模从官方 Hugging Face repository 获取，并固定到实际使用的 commit。M1 是各架构单独训练的 VLGuard safety-SFT LoRA；M3 是单独加载的原始 Qwen3-VL-8B-Instruct v8 视觉复核器。三份 LoRA 都是 adapter-only 权重，不是独立完整模型；复现时无需重新训练。

| 目标模型 | 原始基模和 revision | M1 适配器仓库和 revision | 权重 SHA-256 | 加载规则 |
|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | [boli-003/mm-safetybench-qwen3vl8b-m1-vlguard-v1](https://huggingface.co/boli-003/mm-safetybench-qwen3vl8b-m1-vlguard-v1), revision 待首次发布后填确切 commit | `f512013c1df3ceec919b65b50a553867a9b0849f9b544a91e9987a2faada3584` | `PeftModel` 挂到 Qwen3-VL wrapper |
| LLaVA-NeXT-8B | [llava-hf/llama3-llava-next-8b-hf](https://huggingface.co/llava-hf/llama3-llava-next-8b-hf), `b041c0d0ea0dd0196d147206c210c8d1752fc2da` | [boli-003/mm-safetybench-llavanext8b-m1-vlguard-v1](https://huggingface.co/boli-003/mm-safetybench-llavanext8b-m1-vlguard-v1), revision 待首次发布后填确切 commit | `b8da66030ab8052354c5092d9336e694263ccbade566377656ef9fdc7bd4e313` | `PeftModel` 挂到 LLaVA-NeXT wrapper |
| InternVL3-8B | [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B), `853e3a797a661694b1b8ece0cb72dc2b23e3dac9` | [boli-003/mm-safetybench-internvl3-8b-m1-vlguard-v1](https://huggingface.co/boli-003/mm-safetybench-internvl3-8b-m1-vlguard-v1), revision 待首次发布后填确切 commit | `47974b7b6c49d465e77527278db642af2480045c25eb6199922c100fb04f2c53` | 用 Transformers 4.57.6 将 LoRA 显式挂到 `model.language_model` |

三套基模本地内容树的来源运行 SHA-256、适配器配置 SHA-256、加载器环境和 reviewer/judge 版本见 [`configs/model_registry.json`](../configs/model_registry.json)。正式交付时，表中三个 adapter revision 必须填成 HF 首次上传实际生成的 40 位 commit hash；不能用 `main` 作为正式版本。Qwen3-VL 的 HF 卡标注 Apache-2.0，LLaVA-NeXT 使用 Llama 3 Community License。InternVL3 的 HF 仓库元数据标注 Apache-2.0，但模型说明文字称 MIT 并列出 Qwen2.5 Apache-2.0 组件；请检查所 pin 版本的许可证文件和组件条款，不要将 adapter 许可理解为高于上游条款。

统一 M3：`Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，不挂载 Qwen 的 M1 LoRA。标签裁判：`Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`。两者都不属于 M1 adapter 仓库。

## 下载

使用 `hf` CLI（huggingface-hub）并将版本固定到 commit：

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --local-dir "$HOME/models/Qwen3-VL-8B-Instruct"
hf download llava-hf/llama3-llava-next-8b-hf --revision b041c0d0ea0dd0196d147206c210c8d1752fc2da --local-dir "$HOME/models/llama3-llava-next-8b-hf"
hf download OpenGVLab/InternVL3-8B --revision 853e3a797a661694b1b8ece0cb72dc2b23e3dac9 --local-dir "$HOME/models/InternVL3-8B"
hf download Qwen/Qwen2.5-14B-Instruct --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 --local-dir "$HOME/models/Qwen2.5-14B-Instruct"
```

M1 adapter 下载命令见 [`TEST_PROCEDURE.md`](../TEST_PROCEDURE.md)。LLaVA-NeXT 的模型许可为 Llama 3 community license；不得把其 LoRA 重新标注为 Apache 或 MIT。Qwen、InternVL 的上游模型许可仍适用于其 adapter。权重许可与模型访问条件以 Hugging Face 模型卡和原始许可证为准。
