# V1 source release readiness

Version `1.0.0-rc1`, prepared 7 October 2026. The author requested publication of
this source candidate to `riteshthawkar/Rubric-CEPR` on GitHub. Publication of
source does not certify a stable release, a GPU reproduction, or the submitted
paper scores. No model-hub, package-index or Overleaf upload is part of this
source publication.

## Prepared and checked

- Separate source checkout; the frozen parent source and archive are unchanged.
- Original hash-pinned trainer recovered from the exact historical bundle.
- Exact 64-row manifest and 128 image references verified, including artifact
  installation into a fresh local data root.
- Historical adapter hash verified; header contains 960 BF16 LoRA tensors.
- All seven surviving score summaries match their original recorded hashes.
  Public copies redact 49 local path fields and preserve metrics; both original
  and redacted-file hashes are recorded.
- Full 737-record ImgEdit source check found zero training-source basename
  overlap and zero training-source byte-content overlap.
- Installed `qedit` Python and all packages pinned by the historical contract
  match its recorded versions. Package inspection did not initialize CUDA.
- Portable training arguments were derived from the original launcher; only
  interpreter, checkout, data-root and output paths are relocated.
- Pinned upstream scorer commits plus release patches reconstruct the local
  ImgEdit, GEdit and Complex-Edit scorer source bytes.
- 26 CPU tests passed: artifact/path rejection, overlap protection, original
  inputs, launch arguments, scheduler guards, incomplete training rejection,
  caller-relative paths, processor-loading interface and result boundaries.
- Release Python syntax and Slurm shell syntax checked; source-only payload
  scanned for credentials and host-specific config paths.
- Source archive has deterministic packaging and a SHA-256 sidecar; every
  payload file is indexed in `SHA256SUMS`.

## Remaining release work

1. Supply the authors' code license, attribution and final citation metadata.
   The frozen manuscript is anonymous and no project license was present.
2. Settle distribution terms for the companion image bundle and adapter before
   publishing them; they are excluded from the source repository and archive.
3. Run an allocated GPU smoke test for this portable wrapper and processor
   loader. No GPU training, inference, export or API scoring was executed during
   release preparation. CPU validation is not a claim of end-to-end GPU success.
4. Resolve the score discrepancies in `RESULTS.md` in the public description.
   Do not state that this package reproduces the submitted 4.60 / 8.91 numbers.

For a fresh training rerun, prepare external base-model and benchmark assets
first. The release requirements include TensorBoard because the frozen command
requests it; the current shared `qedit` environment has no TensorBoard package.
That environment was not changed during release preparation. The recorded
contract's core-version check does not by itself certify all optional logging
or benchmark dependencies.

This release establishes inspectable code, exact retained inputs and honest
historical evidence. It does not establish every paper ablation, multi-round
gain, category-wide gain or broad backbone generalization.
