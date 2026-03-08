# Volta — Public Roadmap

Last updated: 2026-03-08

## Completed

- **Phase 1 — Adam Performance**: Adam optimizer reaches PyTorch parity (+25% faster at B≤64). SGD regression gate maintained.
- **Phase 2 — Infrastructure**: Explicit portable/native CPU target contract, portable gemm_shim, automatic RUSTFLAGS propagation, explicit backend capabilities matrix.
- **Phase 3 — Reliability and Correctness**: MHA full backward pass, panic-free autograd, Tensor::PartialEq.
- **Phase 4 — CPU Training Path Hardening**: Long-loop stability, fail-fast non-finite guards, compile-reuse regression gate.
- **Phase 5 — End-to-End PyTorch Parity**: ConvNet and tiny-transformer parity against PyTorch. AOT training codegen confirmed MLP-only, with C path = SGD-only and Rust path limited to SGD/Adam/AdamW/Adagrad.

## In Progress

- **Phase 6 — Product Surface Hardening**: CLI/help/output, `doctor` capability matrix, and docs aligned with the real CPU portability contract: Tier 1 source/runtime CPU support is x86_64 + ARM64; everything else is best-effort.
- **Phase 7 — Packaging and Install Story**: Release workflow and smoke tests aligned with what is actually shipped, without implying extra release artifacts that are not validated.

## Planned

- Linux ARM64 release artifact pipeline and validation
- Broader packaged target coverage beyond the current shipped set

## Later / Unmapped

- Autotune tile sizes (PERF-V2-01)
- Cross-platform build beyond current Tier 1 source/runtime CPU scope (PLAT-V2-01, PLAT-V2-02)
- Broader AOT model coverage beyond MLP (MODEL-V2-01)
- CUDA path benchmarking and verification (CUDA-V2-01 through CUDA-V2-03)
