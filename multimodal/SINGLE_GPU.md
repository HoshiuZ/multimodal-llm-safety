# Single-GPU multimodal inference

This entry runs the frozen M1 + M3 v8 method and its original-model baseline sequentially on CUDA device 0. It uses one GPU and writes response files for the evaluator's own judge. It does not train models or assign safety labels. The versioned method is the same candidate described in `configs/eval_protocol.json`; the single-GPU entry must be validated on the deployment hardware before formal use.

## Prepare the environment and models

Install the common and InternVL environments from the two requirements files. The Word names `qwen_and_llava` and `intern`, repository venvs, and existing `hkc`/`hkc_internvl` environments are discovered automatically. Run from the repository root. The entry selects the validated common Python for review/merging and the InternVL Python for InternVL generation, including when the current shell remains in `intern` after installation.

Run `python multimodal/scripts/document_steps.py environment` after installing both requirements files. It prepares all pinned model/adapter/judge downloads and restores missing/LFS-pointer images together. Use `DOCUMENT_STEPS.md` for the six-step Word procedure and hardware prerequisites.

Download the pinned base models and M1 adapters from `models/README.md` into:

```text
multimodal/models/
├── Qwen3-VL-8B-Instruct/
├── llama3-llava-next-8b-hf/
├── InternVL3-8B/
├── Qwen2.5-14B-Instruct/
└── adapters/
    ├── qwen3vl8b-m1-vlguard-v1/
    ├── llavanext8b-m1-vlguard-v1/
    └── internvl3-8b-m1-vlguard-v1/
```

The Qwen base is also the frozen M3 reviewer. Each adapter contains `adapter_config.json` and `adapter_model.safetensors`. The inference implementation validates model and adapter fingerprints against `configs/model_registry.json`.

## Prepare the evaluation inputs

For each dataset view, place a `manifest.jsonl` and referenced images under `multimodal/data/input/<view>/`. For example, use separate views for the base and each evolved input. Each non-empty JSON line contains a unique `sample_id`, the actual `query`, an `image_path` relative to the view directory, and its `img_sha256` hash. The runner processes every view with a manifest. Input files and outputs are ignored by Git.

```text
multimodal/data/input/
├── full/
│   ├── manifest.jsonl
│   └── images/...
└── evolved/
    ├── manifest.jsonl
    └── images/...
```

The independent evaluator may prepare its own inputs and apply its own judge. The official MM-SafetyBench and project-derived FigStep/HADES materials used in previous experiments are documented in `data/README.md`.

## Run

```bash
python multimodal/scripts/run_single_gpu.py qwen3vl8b
python multimodal/scripts/run_single_gpu.py llavanext8b
python multimodal/scripts/run_single_gpu.py internvl3_8b
```

For each view, `multimodal/outputs/<model>/<view>/manifest.orig.jsonl` contains baseline responses and `manifest.m1_m3_v8.jsonl` contains method responses. Completion markers and `run.log` support verification and diagnosis. Do not report C or S until the independent evaluator has judged the relevant baseline and method responses and applied its agreed metric protocol.

For an existing installation, `MM_DOCUMENT_CONFIG`, `MM_MODELS_ROOT`, `MM_INPUT_ROOT`, `MM_OUTPUT_ROOT`, `MM_COMMON_PYTHON`, and `MM_INTERNVL_PYTHON` can override the defaults. The checking, generation, judging and aggregation entries share the same path configuration. No overrides are needed with the Word layout.
