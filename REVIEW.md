# PKG1 交付核验

Plan_ID：PKG1-text-evaluation-github-20261030。日期：2026-10-30。

CPU 验证使用既有 Qwen 环境的 Python 3.12.14，没有安装依赖。10项 unittest 全部通过，包含真实 CLI 的 preflight → base run → resume → cleared run → aggregate 路径；模型、tokenizer 和权重读取在该 CLI fixture 中以良性小对象替代。另有实际 LLsum 的逐位置和 mask 数值核对，以及 S2/S3 聚合整数回归。没有加载真实模型、提交 GPU 作业或新增研究评分。

全部6个 CLI `--help` 入口可用，文档列出的15条命令均对应真实脚本，包内文档链接可解析。缺少外部数据清单时明确 FileNotFoundError，没有模拟输出参考指标。

发布文件采用目录外的显式 allowlist；每个文件记录相对路径、字节数、SHA256和摘要。已逐文件检查：仅代码、配置、文档、原许可证和历史聚合数字。个人路径、账号、IP、密钥、凭据、二进制与训练导入扫描无命中。字段名 `answer_index`、`question_id` 和测试中的 `synthetic-*` 属于接口与良性合成 fixture，不是真实题库内容或真实逐题标识；模型/源码 SHA 用于版本身份，不是题目正文或分数。未复制原始日志、历史整包配置、权重、真实题目、改写正文、真实答案、token IDs或逐题评测结果。

真实三模型完整 HF 导出及外部数据仍需提供方交付并复验；CPU fixture 不证明真实 GPU forward 已在本包跑通。历史 LLaVA 完整 C 与项目 S2 的评分器不同，尚未统一完整 C 复评。
