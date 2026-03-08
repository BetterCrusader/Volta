# Volta

## What This Is

Volta — компілятор і runtime для нейронних мереж. Власна мова `.vt` компілюється в typed dataflow IR, оптимізується через pass pipeline, і виконується на CPU (інтерпретатор + AOT codegen) або GPU (CUDA, experimental). Цільова аудиторія — розробники, яким потрібна висока швидкість CPU training без залежності від PyTorch/CUDA.

## Core Value

CPU training швидший за PyTorch eager — виміряно, відтворювано, з числовою коректністю.

## Current State (v1.0 shipped: 2026-03-08)

Volta v1.0 MVP shipped. CPU training path повністю hardened, MHA backward реалізований, AOT codegen працює для MLP архітектур. Release pipeline автоматизований для Linux/macOS/Windows.

## Requirements

### Validated

- ✓ Compiler pipeline: `.vt` → AST → IR → ExecutionPlan — v1.0
- ✓ CPU interpreter: sequential, bit-level deterministic — v1.0
- ✓ Autograd: full reverse-mode backward graph — v1.0
- ✓ Optimization passes: DCE, CSE, constant folding, algebraic simplification, elementwise fusion, gradient fusion — v1.0
- ✓ SGD training: beats PyTorch eager 35-67% at B≤64 (MLP architectures) — v1.0
- ✓ Adam optimizer: +25% faster than PyTorch at B≤64, numerically correct — v1.0
- ✓ AOT Rust codegen: MLP → Rust crate with GEMM/Rayon → native DLL — v1.0
- ✓ Benchmark harness: 30 outer × 7 inner × 50 steps, p50/p95/min/max — v1.0
- ✓ CLI: run, check, info, compile, compile-train, doctor, extract, init — v1.0
- ✓ Model high-level API: ModelBuilder, Module trait, layers — v1.0
- ✓ MHA full backward pass (7 gradients) — v1.0
- ✓ Stable graph fingerprints (SipHasher13) — v1.0
- ✓ Portable gemm_shim (include_bytes!, no hardcoded paths) — v1.0
- ✓ Backend capabilities matrix, validate_optimizer() early failure — v1.0
- ✓ Long-loop stability: SGD/Adam/AdamW without NaN/divergence — v1.0
- ✓ End-to-end PyTorch parity: ConvNet + tiny-transformer — v1.0
- ✓ CLI smoke tests + doctor rewrite with MKL/LLVM diagnostics — v1.0
- ✓ Release artifacts: Linux/macOS universal/Windows, smoke-check CI — v1.0

### Active

- [ ] AOT training codegen beyond MLP — Conv2D, LayerNorm, MHA (MODEL-V2-01)
- [ ] AutoTuner для tile sizes MC/KC/NC (PERF-V2-01)
- [ ] CUDA stubs підключені до реальних kernels або прибрані (CUDA-V2-01/02/03)
- [ ] Cross-GPU parity benchmarks — реальні числа проти PyTorch GPU

### Out of Scope

- Distributed training (multi-node, NCCL) — не планується
- Model zoo / pretrained weights — не планується
- Browser/WASM — не планується
- HuggingFace Hub integration — не планується

## Context

Проект на Windows 11 x86-64, Rust edition 2024. v1.0 shipped з повним CPU training path.

Залишкові відомі проблеми:
- CUDA `run()` stubs — no-ops, GPU нічого не рахує (out of scope v1.0)
- AOT codegen MLP-only — Conv2D/LayerNorm/MHA потрібні для v2.0

## Constraints

- **Correctness**: будь-яка зміна optimizer повинна зберігати числову коректність (є регресійний тест)
- **Benchmark regression gate**: B=64 MLP-512 median < 2.10 ms
- **No unsafe code**: `#![deny(unsafe_code)]` на рівні crate
- **Platform**: codegen — Linux/macOS/Windows; interpreter cross-platform

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Rust AOT codegen замість інтерпретатора для training DLL | Нативна швидкість без LLVM залежності | ✓ Good — 35-67% над PyTorch SGD |
| MKL cblas_sgemm для SGD GEMM | Максимальна швидкість GEMM на Intel | ✓ Good |
| Adam elementwise path без fused GEMM | Складніший update шлях | ✓ Good — +25% faster at B≤64 досягнуто |
| CUDA за feature flag | Уникнути обов'язкової CUDA залежності | ✓ Good |
| SipHasher13 для fingerprinting | Стабільний між Rust versions | ✓ Good — замінив DefaultHasher у Phase 2 |
| include_bytes! для gemm_shim.c | Портабельність без CARGO_MANIFEST_DIR | ✓ Good |
| lipo для macOS universal binary | Один artifact для x86_64 + aarch64 | ✓ Good |
| PyTorch parity як головний oracle | Доводить математичну коректність training path | ✓ Good |
| fail-fast на non-finite в train_graph | Явна failure замість silent drift | ✓ Good |

---
*Last updated: 2026-03-08 after v1.0 milestone*
