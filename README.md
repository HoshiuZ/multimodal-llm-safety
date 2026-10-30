# 多模态模型有害知识清除：文本评测

本目录交付 Qwen3-VL、InternVL3 和 LLaVA-NeXT 的纯文本 C/S 评测代码。三个模型均从完整多模态 HF 检查点加载，再使用该检查点的文本推理路径。多模态评测作为独立交付包提供，见 [多模态评测说明](multimodal/README.md)。

- [文本使用说明](text/README.md)：环境、模型、外部数据和命令。
- [测试步骤](TEST_PROCEDURE.md)：本轮已验证范围与实际模型复验步骤。
- [外部数据约定](text/data/README.md)：本轮不分发原题、改写题、答案或题目 ID。
- [模型约定](text/models/README.md)：本轮不分发权重；用户自备完整 HF 模型目录。
- [历史聚合参考](text/results/reference_results.json)：仅对应记录中的具体旧检查点。
- [第三方来源](THIRD_PARTY_NOTICES.md)：源码与模板版本、原许可证及本项目代码范围。

本包不包含训练脚本、训练数据、优化器或方向计算。模型下载链接与真实评测数据将在后续交付中补充。
