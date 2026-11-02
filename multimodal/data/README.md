# 多模态评测数据

## GitHub 仓库内的演化图像

本项目制作并用于评测的 FigStep 与 HADES 风格演化材料已随 GitHub 仓库提供：

- FigStep 图像和清单：**multimodal/data/figstep/**
- HADES 风格图像和清单：**multimodal/data/hades_local_v1/**
- 固定的 1,008 个基础样本编号：**multimodal/data/sample_1008_seed42.txt**

两个图像目录各含 1,008 张成品图、manifest.jsonl 和数据说明。图像文件由 Git LFS 管理。克隆仓库前安装 Git LFS；如果文件尚未下载，在仓库根目录运行：

    git lfs pull

数据固定来源为 Hugging Face 数据集 boli-003/mm-safetybench-vl-eval-assets 的 commit fcffbe1c5389ce9bdbeb44ca45a2e4093adf62b3。GitHub 内文件与该固定版本一致。

FigStep 是本项目依据 MM-SafetyBench 固定样本制作的图像通道移植，不是原论文 SafeBench 全集或结果。[FigStep 原论文仓库](https://github.com/CryptoAILab/FigStep)。

hades_local_v1 是本项目自建的 HADES 风格恢复版，不是官方 HADES 数据集或官方结果。它基于固定 MM-SafetyBench 子集，按本地恢复协议生成视觉补丁，与官方 HADES 的数据来源、场景覆盖、代理模型、优化器和 patch 共享方式不同。[官方 HADES 仓库](https://github.com/AoiDragon/HADES)。

## MM-SafetyBench 官方题目和图像

官方原始数据已放在本 GitHub 仓库的 `multimodal/data/mm-safetybench/`：题目 JSON 在 `processed_questions/`，图像在 `imgs/`。固定评测清单为 `multimodal/data/manifest_mm.csv`。克隆仓库后如图像未下载完整，在仓库根目录运行 `git lfs pull`。

数据保留官方目录结构，包含 13 个题目 JSON 和原图压缩包中的 5,055 张 JPG。评测清单索引其中 5,040 组图文样本、1,680 个问题和 39 个“场景×图像类型”分层单元；原包中另有 15 张不对应题目 JSON 的图片，保留在仓库中但不进入评测清单。清单 digest 为 `4f8d7c436ce345936d3917254cc57f6e8d3e233546e341696f6b02591c7b9124`。

来源版本：题目 JSON 来自官方仓库 commit `b80eedea3db312c09ded2082813390f68e750ef3`；图像来自官方 `MM-SafetyBench(imgs).zip`，SHA-256 为 `7e1a9f08212038ceaa74801964d2b97c219509663bbd017ab77abb1b063cc23e`。压缩包本身未重复放入仓库。

在项目根目录检查已上传的数据和清单：

    python -B multimodal/scripts/verify_assets.py --manifest multimodal/data/manifest_mm.csv --data-root multimodal/data/mm-safetybench --expected-rows 5040

上游说明数据按 CC BY-NC 4.0 用于非商业研究，并受 GPT-4、Stable Diffusion 相关许可约束；请保留作者归属，不得用于商业用途。[官方数据说明与许可](https://github.com/isXinLiu/MM-SafetyBench#license)。使用数据时请引用论文：[Query-Relevant Images Jailbreak Large Multi-Modal Models](https://arxiv.org/abs/2311.17600)，Xin Liu、Yichen Zhu、Yunshi Lan、Chao Yang、Yu Qiao，2023。

固定稳定性子集的 1,008 个 sample ID 位于 multimodal/data/sample_1008_seed42.txt。该文件只包含编号，不包含题目或答案。运行本项目的数据检查命令后，会校验官方数据与 GitHub 中的 FigStep、HADES 图像并生成推理清单。

## 正常能力数据

单独进行正常能力评测时，需要 1,000 组 VQAv2 和 TextVQA 问题及参考答案。小型清单位于 Hugging Face 数据集的 normal_utility/manifest.jsonl；该清单不包含图像。固定数据版本为 fcffbe1c5389ce9bdbeb44ca45a2e4093adf62b3。

    hf download boli-003/mm-safetybench-vl-eval-assets --repo-type dataset --revision fcffbe1c5389ce9bdbeb44ca45a2e4093adf62b3 --include normal_utility/** --local-dir external/eval-assets

从 [VQAv2 官方下载页](https://visualqa.org/download.html)获取 VQA v2.0 验证集标注和 COCO val2014 图片；从 [TextVQA 官方下载页](https://textvqa.org/dataset/)获取验证集标注及图片。按清单中的图像 ID 获取图片并按 manifest 中的相对路径保存。图像不随本项目仓库或 Hugging Face 清单分发。

VQA 标注按 CC BY 4.0 条款署名；COCO 和 OpenImages 图片遵循各自原始许可。[VQA 条款](https://visualqa.org/terms.html)、[TextVQA 数据集说明](https://textvqa.org/dataset/)。

## 固定版本与权限

| 材料 | 固定身份 | 说明 |
|---|---|---|
| MM-SafetyBench 官方输入 | `multimodal/data/mm-safetybench/`；5,040 行 manifest digest 4f8d7c…9124，逐图哈希 | Git LFS 提供；保留 CC BY-NC 4.0 归属与非商业研究限制 |
| 稳定性基础子集 | sample_1008_seed42.txt | 1,008 个编号，不含题目文本 |
| FigStep 移植图像 | FigStep manifest + 1,008 张成品图；上游 revision 0861b17b3d67887c06ee3534ec65b3012f9becb7 | 项目移植；保留 MM-SafetyBench 归属和许可 |
| HADES 风格图像 | HADES manifest + 1,008 张成品图；协议 hades-local-mmsb-constrained-v1-recovery-v3 | 项目自建素材；不声称为官方 HADES |
| 正常能力 manifest | 1,000 行；VQAv2 500 + TextVQA 500 | 标注 CC BY 4.0；图像从官方来源获取 |

两类演化图均源自 MM-SafetyBench 题目或素材，按上游 CC BY-NC 4.0 要求仅用于非商业研究；请保留上游归属与引用。样例含有害主题，仅用于授权的安全评测。
