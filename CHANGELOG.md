# Changelog

## [1.1.0] - 2026-02-23

### Added

- ONNX Wave 2 expansion: Gemm attribute matrix (`alpha`, `beta`, `transA`, `transB`), reduction keepdims and negative-axis normalization, Gelu mode mapping (`tanh` and `none`).
- Exact GeLU runtime path (`GeluExact`) and ONNX import mapping.
- Autograd integration roundtrip tests and expanded gradcheck coverage.
- Determinism regressions over repeated/threaded schedule generation.
- Fuzzing scaffold (`fuzz/`) with lexer, parser, and ONNX-bytes targets.
- Governance docs:
  - RFC-004 optimizer pass order
  - Scheduler/optimizer architecture notes
  - Quality policy
  - Phase 7 self-review note

### Changed

- CUDA executor now supports Sigmoid/Gelu/GeluExact/Gemm runtime paths and parity tests against CPU.
- Reduction ops now carry explicit keepdims semantics through IR, shape inference, interpreter, ONNX contract/import, and tests.
- ONNX coverage matrix updated to reflect implemented behavior and explicit limits.

### Fixed

- Backward scheduling correctness for multi-input backward ops through proper dependency declaration.
- Multiple parser/import edge-cases with explicit error diagnostics.

### Notes

- Unsupported features continue to fail explicitly (no silent CPU fallback in strict CUDA training mode).
