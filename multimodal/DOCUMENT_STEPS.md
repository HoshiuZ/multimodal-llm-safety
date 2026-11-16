# 跨模态测试流程短命令

本页对应测试方案中的跨模态运行步骤，使用已发布的基模、各自对应的 M1 文件及共用 M3 v8，不需要训练。环境安装、官方数据下载及准确模型版本分别见 `TEST_PROCEDURE.md`、`data/README.md`、`models/README.md`。运行方法仍使用 `SINGLE_GPU.md` 的单 GPU 入口。

以下命令均在仓库根目录执行，与 Word 的六个步骤一致。按 Word 创建 `qwen_and_llava`、`intern` 并安装各自 requirements 后，入口自动选择正确环境；安装结束仍激活 `intern` 也可继续。路径统一来自 `configs/document_steps.json`，模型默认保存在 `multimodal/models/`。已有安装可通过 `MM_DOCUMENT_CONFIG` 或 `MM_MODELS_ROOT`、`MM_INPUT_ROOT`、`MM_OUTPUT_ROOT` 指定，核验和推理共同使用这些设置。

测试使用一张 CUDA GPU，推荐 80 GB 显存及至少 150 GB 可用磁盘空间。14B 裁判约有 27.5 GiB BF16 权重，另需推理缓存和运行开销。默认不量化模型。首次准备需要访问 Hugging Face 和 GitHub 图像源。

```bash
python multimodal/scripts/document_steps.py environment
python multimodal/scripts/document_steps.py models qwen3vl8b
python multimodal/scripts/document_steps.py models llavanext8b
python multimodal/scripts/document_steps.py models internvl3_8b
python multimodal/scripts/document_steps.py data
```

`environment` 核对两套环境后，在同一命令中下载固定版本的三套基模、三份 M1 适配器及 Qwen2.5-14B 本地裁判；M3 复用 Qwen 基模。缺失图像或源码 ZIP 中的 Git LFS 指针由固定 GitHub 数据版本恢复并校验 SHA-256，无需另装 Git LFS。中断后可重跑该命令，HF 下载复用本地缓存。只检查环境、不下载资源时使用 `environment --check-only`。

支持仓库虚拟环境、Word 的 Conda 环境，以及 `hkc`、`hkc_internvl`。特殊安装可用 `MM_COMMON_PYTHON`、`MM_INTERNVL_PYTHON` 指定解释器，仍会检查依赖版本。`models` 核对基模完整内容树、适配器权重和配置 SHA-256。`data` 验证 5,040 条官方图文、演化清单、图片解码和 1,008 个配对 ID，生成 `multimodal/data/input/{full,base,figstep,hades}/manifest.jsonl`。图像通过 `source/` 软链接引用；已有不同清单不会覆盖。

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

`judge` 自动使用共同环境，在 CUDA:0 上调用固定 Qwen2.5-14B 和本地 v2 提示词。已有完整判词核对身份、覆盖和哈希；中断的判定可续跑。若有未决行，同一命令自动执行证据补判：同一张 GPU 顺序运行六份复核，每份通过 45 个控制用例、六份标签一致后才补齐原未决行。已确定的 v2 行和原判词文件保持原样。补判输出在 `evidence_retry.<state>/judgments.v2_plus_v2r.canonical.jsonl`，`c`、`s` 自动选择并检查来源哈希。控制失败或六份不一致仍阻止指标输出。

新单卡补判版本是 `local-v2-evidence-retry-v1-20261028-sequential-six-v1-20261116`，与历史六卡补判分别记录，提示词、标签语义和一致性条件沿用原协议。历史 S 不能预先当作新运行结果。裁判内容树按公开 HF 快照核验，忽略本地根目录 `SHA256SUMS` 和缓存；权重、配置、分词器等内容仍须完整匹配。

`c` 和 `s` 从完成的逐样本判词重新计数，不读取 `reference_results.json` 里的成绩。C 使用完整 5,040 条；S 使用相同 1,008 个基础样本及 FigStep/HADES，按项目冻结有效性条件取最低保持率，不封顶。结果写入 `multimodal/outputs/metrics/<model>/{C,S}.json`，同时保留判词协议、输入哈希和各视图计数。这些是项目内部跨模态协议，不是另行确认的整套方案验收结论。

对已有冻结运行的重算，可在本地配置 `judgments[model][view][state]` 下提供 `{"path":"判词文件","id_prefix":"可选的已记录前缀"}`。例如同一个已完成文件内存放 `base::orig::ID` 等记录时，显式选择相应前缀；选择后仍要求与对应清单的 ID 集完全一致。每个文件必须保留带模型名和 `output_sha256` 的原完成标记，原记录和标签不修改。Word 中的指标截图属于这类已有全量判词的重算，不能解释为重新完成全量单卡推理。
