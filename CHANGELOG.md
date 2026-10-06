# Changelog

## 1.0.0-rc1 — 7 October 2026

Prepared the first separate ACCV source release candidate. Restored the exact
historical extraction trainer from its recovery bundle and kept its scientific
recipe and inputs pinned. Added portable guarded commands, deterministic
artifact installation and source packaging, benchmark setup pins, CPU boundary
tests, and explicit result/provenance documentation.

Compatibility changes are limited to release entry points, input/output path
relocation, explicit official processor loading for inference, portable
Complex-Edit runtime paths, and a pinned dataset revision in its preparer.
The reference framework config disables later preference/replay branches and
emits training commands by default. Historical score summaries redact local
paths; original hashes and unmodified private copies are retained.

The author subsequently requested publication of this source candidate to
`riteshthawkar/Rubric-CEPR`. This source publication adds no experiment result
and makes no change to the active research workload. The original local source
archive retains its preparation-time documentation and checksum.
