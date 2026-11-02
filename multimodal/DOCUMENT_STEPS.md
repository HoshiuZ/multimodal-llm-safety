# 跨模态测试流程短命令

本页对应测试方案中的跨模态运行步骤，使用已发布的基模、各自对应的 M1 文件及共用 M3 v8，不需要训练。环境安装、官方数据下载及准确模型版本分别见 `TEST_PROCEDURE.md`、`data/README.md`、`models/README.md`。运行方法仍使用 `SINGLE_GPU.md` 的单 GPU 入口。

以下命令均在仓库根目录、共同 Python 环境中执行。路径默认来自 `configs/document_steps.json`；已有安装可复制为本地配置，并通过 `MM_DOCUMENT_CONFIG` 指定。原始图像与模型保持原目录，不上传 Git。

```bash
python multimodal/scripts/document_steps.py environment
python multimodal/scripts/document_steps.py models qwen3vl8b
python multimodal/scripts/document_steps.py models llavanext8b
python multimodal/scripts/document_steps.py models internvl3_8b
python multimodal/scripts/document_steps.py data
```

`environment` 检查共同环境和 InternVL3 环境；`models` 核对完整基模内容树、已发布适配器权重及配置的 SHA-256，属于 CPU 文件核验，不声称已加载 GPU。`data` 在官方 5,040 条清单和已发布 FigStep/HADES 资产到位后，验证官方快照、演化清单哈希、每张图像内容与解码、固定 1,008 个配对 ID，生成 `multimodal/data/input/{full,base,figstep,hades}/manifest.jsonl`。图像通过 `source/` 软链接引用，不复制原图；已有不同清单不会覆盖。

运行方法，保存原模型及方法回答：

```bash
python multimodal/scripts/run_single_gpu.py qwen3vl8b
python multimodal/scripts/run_single_gpu.py llavanext8b
python multimodal/scripts/run_single_gpu.py internvl3_8b
```

按模型判定并汇总。下列 `MODEL` 分别替换为上面的三个模型键：

```bash
python multimodal/scripts/document_steps.py judge MODEL
python multimodal/scripts/document_steps.py c MODEL
python multimodal/scripts/document_steps.py s MODEL
```

`judge` 使用固定版本 Qwen2.5-14B-Instruct、`configs/judge_runtime.json` 及 `scripts/local_judge_rubric.py` 的本地 v2 提示词，确定性生成，单张 CUDA:0；已有完整判词先核对模型身份、样本覆盖和文件哈希。输入仍为 `outputs/<model>/<view>/manifest.{orig,m1_m3_v8}.jsonl`，判词写同目录的 `manifest.<state>.judgments.jsonl`。未决记录会阻止汇总。已有 LLaVA/InternVL3 稳定性参考结果采用单独版本的 v2＋证据补判；需要该协议时，按 `TEST_PROCEDURE.md` 保留原判词并完成补判，不能改写为原 v2 结果。本页单卡判定命令不自动调用六卡补判。

`c` 和 `s` 从完成的逐样本判词重新计数，不读取 `reference_results.json` 里的成绩。C 使用完整 5,040 条；S 使用相同 1,008 个基础样本及 FigStep/HADES，按项目冻结有效性条件取最低保持率，不封顶。结果写入 `multimodal/outputs/metrics/<model>/{C,S}.json`，同时保留判词协议、输入哈希和各视图计数。这些是项目内部跨模态协议，不是另行确认的整套方案验收结论。

对已有冻结运行的重算，可在本地配置 `judgments[model][view][state]` 下提供 `{"path":"判词文件","id_prefix":"可选的已记录前缀"}`。例如同一个已完成文件内存放 `base::orig::ID` 等记录时，显式选择相应前缀；选择后仍要求与对应清单的 ID 集完全一致。每个文件必须保留带模型名和 `output_sha256` 的原完成标记，原记录和标签不修改。Word 中的指标截图属于这类已有全量判词的重算，不能解释为重新完成全量单卡推理。
