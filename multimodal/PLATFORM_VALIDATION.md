# Word procedure deployment validation — 2026-11-16

The Word model revisions, M1/M3 method, manifests and C/S formulas are unchanged.
The existing commands now share the Word model directory, discover both Conda
environments, prepare runtime downloads together, and supplement unresolved
judgments on one GPU without overwriting original v2 records.

Validated in the existing host environment:

- 29 CPU regression tests, including path selection, environment discovery,
  interrupted-download protection, six-replica agreement/control rejection,
  automatic supplement selection, and rejection of stale supplement inputs.
- The environment command launched from the InternVL environment finds both
  exact pinned installations; the data command automatically dispatches to the
  common environment and validates/prepares full/base/FigStep/HADES inputs.
- All 5,040 official image pairs and both 1,008-image evolved views pass image
  SHA-256, decoding and paired-ID validation.
- Actual hashing of all eight local judge weight shards and public metadata
  matches the corrected public-snapshot tree fingerprint
  `3b5bd06098645b35670a1a9c019886a84333d11fc53d5bea275c483f20b2ca36`.
  Local `SHA256SUMS` and generated caches are excluded; checkpoint content
  remains checked. No weight file was changed.
- A real public FigStep image was downloaded through the new source-ZIP/LFS
  recovery path and matched its frozen SHA-256. The complete seven-item model
  download plan is tested at immutable revisions; full model downloads were
  not repeated during this change.
- Recomputing the six C/S outputs from the existing real per-sample judgment
  archives gives the values below. These are archive recomputations, not a new
  generation run.

| Model | C (%) | S, uncapped (%) |
|---|---:|---:|
| Qwen3-VL-8B | 89.935760 | 97.959184 |
| LLaVA-NeXT-8B | 91.371515 | 105.372757 |
| InternVL3-8B | 85.493145 | 102.339605 |

The new single-GPU evidence supplement uses the separately recorded protocol
`local-v2-evidence-retry-v1-20261028-sequential-six-v1-20261116`. Its execution
and control/merge behavior have CPU regression coverage using simulated model
responses; it has not received a new full GPU evaluation. No new target-model
generation or judge accuracy measurement was performed. Third-party hardware
must satisfy the documented CUDA/memory/storage prerequisites. Newly generated
answers and labels determine the new scores; historical values are not loaded
as metric outputs.
