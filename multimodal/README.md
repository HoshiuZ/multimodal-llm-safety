# Multimodal M1 + M3 v8 evaluation

This package reproduces the project's frozen multimodal evaluation for Qwen3-VL-8B-Instruct, LLaVA-NeXT-8B, and InternVL3-8B. It loads the published M1 LoRA adapters and applies the frozen Qwen3-VL visual reviewer (M3 v8); it does not train or optimize models. The local Qwen2.5-14B-Instruct judge is included as code and must be downloaded separately. The reported C/S values are project-proposed component metrics, not a certification against a scheme-wide acceptance rule. See [`results/reference_results.json`](results/reference_results.json) for exact model revisions, run IDs, limits, and numbers.

## Contents

- [`TEST_PROCEDURE.md`](TEST_PROCEDURE.md): step-by-step download, inference, judging, supplement, and aggregation instructions.
- [`SINGLE_GPU.md`](SINGLE_GPU.md): one-GPU method and baseline inference for an independent evaluator.
- [`DOCUMENT_STEPS.md`](DOCUMENT_STEPS.md): short commands for environment/model/data checks, single-GPU local judging, and C/S recomputation from completed judgments.
- [`data/README.md`](data/README.md): GitHub data paths, official benchmark sources, and permissions.
- [`models/README.md`](models/README.md): pinned base and adapter model cards.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md): included code/font notices and upstream data/model terms.
- [`configs/`](configs/): frozen model registry, inference settings, judge settings, protocol, and aggregate template.
- [`scripts/`](scripts/): manifest/data preparation, generation, M3 review, local judging, evidence retry, and metric aggregation.
- [`results/reference_results.json`](results/reference_results.json): final recorded component metrics and scope limitations.

## Environment

Python 3.12.12 is pinned. Qwen3-VL, LLaVA-NeXT, M3, and the local judge use [`requirements.txt`](requirements.txt). InternVL3 generation uses the separate [`requirements-internvl.txt`](requirements-internvl.txt) environment because its tested Transformers version differs. The reference reproduction procedure launches six processes on six visible CUDA devices for each stage. For sequential inference on one GPU, use [`SINGLE_GPU.md`](SINGLE_GPU.md). Use a separate output directory for each model and dataset view.

The FigStep and HADES evaluation images are stored with Git LFS. Install Git LFS before cloning the project so these images are available locally.

The model implementation uses each processor's native image path: Qwen3-VL processor chat template and resize path; LLaVA-NeXT processor chat template and any-resolution image processing; InternVL3's 448-pixel dynamic tiling with 1–12 patches, optional thumbnail, bicubic resize, and ImageNet normalization. Exact templates, prompts, decoding, M3 settings, and judge controls are versioned in [`configs/eval_protocol.json`](configs/eval_protocol.json), [`configs/baseline_runtime.json`](configs/baseline_runtime.json), [`configs/judge_local.json`](configs/judge_local.json), and the source scripts.

## Models and data

Download the three base checkpoints and the local judge at the immutable revisions in [`models/README.md`](models/README.md). Download the matching M1 adapter from the same page and verify its file hashes; no retraining is needed. GitHub supplies the official MM-SafetyBench input files in `multimodal/data/mm-safetybench/` and the project-made FigStep/HADES images in `multimodal/data/figstep/` and `multimodal/data/hades_local_v1/`. Install Git LFS and run `git lfs pull` if the images are missing after cloning. Dataset source and license details are in [`data/README.md`](data/README.md).

The fixed 1,008-ID selection is in `multimodal/data/sample_1008_seed42.txt`. The normal-utility manifest is available from the pinned Hugging Face dataset; its VQAv2/TextVQA image files are not redistributed and must be obtained from their official sources. License and attribution instructions are in [`data/README.md`](data/README.md).

## Quick start

1. Follow [`TEST_PROCEDURE.md`](TEST_PROCEDURE.md) to prepare environments, download pinned models, and verify the GitHub-provided MM-SafetyBench data and manifest.
2. Run `scripts/run_dataset.sh` separately for the 5,040-row full set, the fixed 1,008-row base subset, FigStep, HADES-style assets, and normal utility set.
3. For LLaVA-NeXT and InternVL3 stability views, reproduce the separately versioned evidence-first retry for unresolved v2 judge rows. This supplements unresolved labels only; it never replaces resolved v2 labels. Qwen3-VL uses frozen v2 labels for its recorded S result.
4. Fill a copy of [`configs/aggregate_template.json`](configs/aggregate_template.json) with the output paths for one model and run `scripts/aggregate.py`. The aggregator verifies completion markers, file hashes, sample coverage, paired stability IDs, and target thresholds before writing metrics.

For all arguments, environment separation, output layout, and commands, use [`TEST_PROCEDURE.md`](TEST_PROCEDURE.md). This is a large evaluation; estimate storage for model checkpoints and generated outputs before running. The released FigStep and HADES images are tracked with Git LFS; do not commit model downloads, official MM-SafetyBench originals, or evaluation outputs.

## Metric scope

`C_mm = 100 × (ASR_orig − ASR_method) / ASR_orig` over the 5,040 fixed MM-SafetyBench rows. `S_mm` is the minimum relative ASR-clearance retention across the fixed FigStep and HADES-style views, compared with the same 1,008 base IDs. Normal utility uses the VQA ten-reference consensus score on 500 VQAv2 and 500 TextVQA pairs, with relative loss `100 × (P_orig − P_method) / P_orig`. The judge is local Qwen2.5-14B-Instruct with the bundled project rubric; it is not the benchmark's official GPT-4/Azure evaluator, and its independent accuracy has not been established. See the protocol and reference-results file before comparing numbers.
