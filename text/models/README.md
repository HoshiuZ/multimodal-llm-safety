# 完整基模与稀疏更新

PKG2 的模型身份及本轮 sparse update 的键、shape、dtype、大小和 SHA256 见 `configs/*.json`。基模仓库与 revision 来自项目已通过的模型文件清单：

| 模型 | 基模仓库 | revision |
|---|---|---|
| Qwen | Qwen/Qwen3-VL-8B-Instruct | 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b |
| InternVL | OpenGVLab/InternVL3-8B | 853e3a797a661694b1b8ece0cb72dc2b23e3dac9 |
| LLaVA | llava-hf/llama3-llava-next-8b-hf | b041c0d0ea0dd0196d147206c210c8d1752fc2da |

评分时须提供对应 revision 的完整多模态 HF `save_pretrained` 目录，包含 wrapper 权重、config 与 tokenizer；InternVL 还依赖该 revision 的本地 remote code。禁止替换成语言祖先模型。加载要求完整键集合匹配、参数为 BF16、模型类及 attention 路径符合配置。

overlay 文件不随公开包分发。运行前核验其大小与 SHA256；仅用 `weights_only=True` 读取张量，检查完整键集合、shape、dtype 和有限值，再复制到完整 wrapper 并逐键 bitwise 比较。恢复不会转换 dtype。`restore-check` 在 CPU 加载完整基模后执行相同恢复核验。

基模大型 shard 的 SHA256 已由原冻结模型清单与上游 LFS 信息记录；PKG2 运行逐文件核对小型 config/tokenizer 文件，但不重复哈希全部多 GB 权重。公开配置也不包含模型权重或 overlay 下载地址；目前没有已确认的公开发布链接。

LLaVA 本轮实际加载的基模是 `llava-hf/llama3-llava-next-8b-hf`，revision 为 `b041c0d0ea0dd0196d147206c210c8d1752fc2da`，wrapper 为 `LlavaNextForConditionalGeneration`。现有材料将该 HF 来源记为工程候选；尚不明确的是它与测试方案所列正式 LLaVA 型号名称的对应关系。本轮实际加载身份明确如上。InternVL 的实际 overlay 有5个参数键，层为4、5、6、7、9；训练 manifest 记载的 NPO 阶段却指向层10–13。此来源差异未能从现有材料消解；结果仅能归属于 SHA 固定的实际 sparse update，不能断言 NPO 阶段对该文件的具体贡献。

模型许可及访问限制继续适用。本包保留第三方许可与来源说明，未复制任何权重、tokenizer 数据集正文或凭据。
