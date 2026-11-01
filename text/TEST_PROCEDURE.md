# 文本模态测试入口

本目录提供基模加载、C/S 评分、指标汇总和模型恢复的短命令。模型权重、清除参数、训练数据与评测题目不包含在仓库中。评测输入为纯文本，使用完整多模态模型；不能据此推断图像模态结果。

## 准备

在独立环境中安装共用的 `requirements.txt`。准备固定 C/S 数据清单；来源和重建方式见 `data/README.md`。LLaVA 默认绑定公开 revision `959a02037889f1f74cacfdc0152dc53444217563`，使用独立的 `20261101` 模型目录与 C/S 输出目录；需重新准备该版完整模型，旧评分输出不能用于新版汇总。按 `score_paths.json` 放置基模与送测的完整清除态模型。官方基模来源和清除后模型链接见 `models/README.md`；缺少基模时，可以在具备网络权限的准备环境运行 `python score_internvl.py --prepare-base`（其他模型替换名称）。配置中的路径可根据实际环境调整。

GPU 评分使用调度器分配的单张 GPU，逻辑设备为 `cuda:0`。请在计算节点运行，不在登录节点直接启动 GPU 推理。三个模型的文本评测使用这套共用依赖。以下命令均在 `text` 目录执行。

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

完整训练入口位于仓库根目录的 `training/`，运行方式见 [训练说明](../training/README.md)。从仓库根目录分别运行 `python training/train_qwen.py`、`python training/train_llava.py` 和 `python training/train_internvl.py`。运行前需准备说明中列出的基模、训练数据和 GROM basis。InternVL 的 NPO 更新第 4–7 层，最终保存第 4、5、6、7、9 层；交付记录及核验范围见 [模型说明](models/README.md)。`text/train_*.py` 是历史阶段入口，不作为当前完整训练流程。

`python restore_qwen.py`、`python restore_llava.py` 和 `python restore_internvl.py` 使用 `restore_paths.json` 指定的完整基模和**已冻结的选定清除参数**恢复完整模型。入口校验选定参数的 SHA256，保存后重新加载并逐张量比较，并核对权重文件与发布记录；已有输出目录不会覆盖。自行训练的中间参数不一定是选定参数，不能直接当作发布版本恢复。

上述入口支持 `--dry-run` 仅查看执行计划；dry-run 不进行模型推理或训练。需要下载材料、实际运行模型或修改配置时，应按真实环境分别准备和核验。
