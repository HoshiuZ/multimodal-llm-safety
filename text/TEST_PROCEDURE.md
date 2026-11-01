# 文本模态测试入口

本目录提供基模加载、C/S 评分、指标汇总和模型恢复的短命令。模型权重、清除参数、训练数据与评测题目不包含在仓库中。评测输入为纯文本，使用完整多模态模型；不能据此推断图像模态结果。

## 准备

在独立环境中按模型安装对应 `requirements-*.txt`。准备固定 C/S 数据清单；来源和重建方式见 `data/README.md`。按 `score_paths.json` 放置基模与送测的完整清除态模型。官方基模来源和清除后模型链接见 `models/README.md`；缺少基模时，可以在具备网络权限的准备环境运行 `python score_internvl.py --prepare-base`（其他模型替换名称）。配置中的路径可根据实际环境调整。

GPU 评分使用调度器分配的单张 GPU，逻辑设备为 `cuda:0`。请在计算节点运行，不在登录节点直接启动 GPU 推理。每个模型需要使用其对应依赖环境。以下命令均在 `text` 目录执行。

## 基模与 C 指标

以 InternVL 为例：

```bash
python baseline_internvl.py --load-only
python baseline_internvl.py
python score_internvl.py
python score_internvl_aggregate.py
```

第一条只加载基模；第二条独立计算基模 C 集准确率。第三条为保证逐题配对身份一致，在同一评测合同下分别对基模和清除态重新评分；第四条核验覆盖并汇总 C。Qwen、LLaVA 将命令中的 `internvl` 分别替换为 `qwen`、`llava`。输出和合同路径由 `score_paths.json` 指定；脚本不会覆盖已经完成的独立基模测试，完成的配对评分也不能直接当作新一轮测试复用。

## S 指标

以 InternVL 为例：

```bash
python score_internvl_s.py
python score_internvl_s_aggregate.py
```

每个场景各 734 题；汇总输出 raw S 和封顶后的 S。三个模型使用各自的合同、模型路径和输出目录。评分入口仅在本轮基模和清除态评分完成后允许汇总。

## 训练与恢复

`train_qwen.py` 是已记录的 Qwen GROM 与 NPO 阶段示例；`train_llava.py` 和 `train_internvl.py` 是各自已记录的 NPO 阶段，不是完整发布模型的复现脚本。运行前需根据 `train_qwen_config.json` 或 `train_stages.json` 配置包含原始训练材料的项目根目录，并准备基模、训练数据及所需中间参数。仓库不提供这些材料。训练输出是参数更新文件，不是完整模型；不能仅凭脚本运行成功就声称复现了发布模型。InternVL 历史脚本虽接收训练层参数，实际构造可训练参数时固定使用第 4–7 层；该训练阶段与最终发布参数的来源关系尚未确认。

`python restore_qwen.py`、`python restore_llava.py` 和 `python restore_internvl.py` 使用 `restore_paths.json` 指定的完整基模和**已冻结的选定清除参数**恢复完整模型。入口校验选定参数的 SHA256，保存后重新加载并逐张量比较，并核对权重文件与发布记录；已有输出目录不会覆盖。自行训练的中间参数不一定是选定参数，不能直接当作发布版本恢复。

上述入口支持 `--dry-run` 仅查看执行计划；dry-run 不进行模型推理或训练。需要下载材料、实际运行模型或修改配置时，应按真实环境分别准备和核验。
