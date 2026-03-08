# Volta — Public Roadmap

Last updated: 2026-03-08

## Completed

- **Phase 1 — Adam Performance**: Adam optimizer reaches PyTorch parity (+25% faster at B≤64). SGD regression gate maintained.
- **Phase 2 — Infrastructure**: Portable gemm_shim, automatic RUSTFLAGS propagation, explicit backend capabilities matrix.
- **Phase 3 — Reliability and Correctness**: MHA full backward pass, panic-free autograd, Tensor::PartialEq.
- **Phase 4 — CPU Training Path Hardening**: Long-loop stability, fail-fast non-finite guards, compile-reuse regression gate.
- **Phase 5 — End-to-End PyTorch Parity**: ConvNet and tiny-transformer parity against PyTorch. AOT training codegen confirmed MLP-only (explicit truth-pass).

## In Progress

- **Phase 6 — Product Surface Hardening**: CLI help/output, `doctor` capability matrix and actionable diagnostics, examples and docs aligned with real behaviour.

## Planned

- **Phase 7 — Packaging and Install Story**: Release artifacts, clean install verification, smoke-tested shipped binary.

## Later / Unmapped

- Autotune tile sizes (PERF-V2-01)
- Cross-platform build (PLAT-V2-01, PLAT-V2-02)
- Broader AOT model coverage beyond MLP (MODEL-V2-01)
- CUDA path benchmarking and verification (CUDA-V2-01 through CUDA-V2-03)
