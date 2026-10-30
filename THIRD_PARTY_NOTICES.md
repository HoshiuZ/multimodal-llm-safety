# 第三方来源与许可

本交付不为整个仓库指定新开源许可证。上游软件和模型许可继续适用；公开代码不等于获得重新分发模型或数据的许可。

- WMDP 作者源码：centerforaisafety/wmdp，固定提交 `c0b6c12bb0de3decf9c13bb13f9f1aa15754a132`。保留原 [MIT 许可](licenses/WMDP-MIT.txt)。本包不复制其训练 runner，不分发其题库；数据版本记录为 `7125571f22f032c56415e7980f48d877dd830ff8`。
- 官方域 description 和0-shot选择题格式：EleutherAI/lm-evaluation-harness，固定提交 `4600d6bf73ba2cf7037ae7feada03315839ef185`（v0.4.2 来源）。保留原 [MIT 许可](licenses/LM-EVAL-MIT.txt)。域 description 按实际冻结配置提取。
- Qwen/Qwen3-VL-8B-Instruct、OpenGVLab/InternVL3-8B：实际 manifest 标记 Apache-2.0；本包仅记录来源，不分发模型文件。InternVL 的 remote code 仍由用户的模型目录提供，需保留其原声明。
- llava-hf/llama3-llava-next-8b-hf：实际 manifest 标记 llama3，模型许可仍适用；不能把整模型重新标为 MIT 或 Apache-2.0。
- PyTorch、transformers、accelerate 等作为外部依赖使用，不复制其完整源码或更改其许可。

本项目代码范围是 `text/eval/` 中的内容身份、schema、续评、聚合、CLI适配，以及 `text/scripts/`、合成测试和文档。`scoring.py` 的评分算法和 `models.py` 的实际加载合同从项目既有 S2/S3 代码最小提取，来源 SHA 在 `text/results/source_provenance.json`。此声明不为这些文件额外选择许可证，也不声称对第三方模板拥有原创权。未来如需整仓库 LICENSE，由权利人另行确定。
