# 多模态模型有害知识清除：PKG2 文本复核包

本目录记录三款完整多模态模型 sparse update 的纯文本 C 与正常能力复核。评测使用固定基模 revision 和完整 HF wrapper；不以语言祖先模型替代。实际更新的 SHA、参数键与 shape 见各模型配置。

- [文本评分合同与运行说明](text/README.md)
- [测试与交付复核步骤](TEST_PROCEDURE.md)
- [外部数据格式](text/data/README.md)：公开包不含真实题目正文、ID、答案、token IDs 或逐题分数
- [基模与稀疏更新身份](text/models/README.md)：公开包不含权重或 overlay 二进制
- [PKG2 聚合结果](text/results/pkg2_c_and_other49.json)：包含本轮三款模型的 C 与固定 other49 汇总
- [历史参考聚合](text/results/reference_results.json)：PKG1 旧检查点数据，不是本轮复评分数
- [第三方来源与许可](THIRD_PARTY_NOTICES.md)

本轮三个 C 结果均达到70%目标；other49 能力门槛按本批20%损失预算，即 `5*n_cleared >= 4*n_base`。Qwen 与 LLaVA 通过，InternVL 为146/245，低于最低正确数147，因此三模型合并门槛未通过。结果仅适用于文件中列出的完整基模 revision、确切 overlay、评分代码与冻结数据。
