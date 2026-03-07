# Requirements: Volta

**Defined:** 2026-03-07
**Core Value:** CPU training швидший за PyTorch eager — виміряно, відтворювано, з числовою коректністю

## v1 Requirements

### Performance

- [x] **PERF-01**: Adam optimizer досягає паритету з PyTorch (≤1.1×) або перевищує на B≤64 MLP benchmarks
- [x] **PERF-02**: SGD regression gate зберігається: B=64 MLP-512 median < 2.10 ms після будь-яких змін
- [x] **PERF-03**: `target-cpu=native` автоматично пропагується в `compile-train --rust` без ручного RUSTFLAGS

### Correctness

- [x] **CORR-01**: Adam gradient updates числово коректні (перевірка проти reference implementation)
- [x] **CORR-02**: Backend capabilities явно задокументовані — рання відмова для unsupported combinations замість silent fallback
- [x] **CORR-03**: MHA backward обчислює коректні градієнти для Q, K, V та projection weights

### Infrastructure

- [x] **INFRA-01**: Hardcoded dev MKL path (`C:/Users/User/miniforge3/...`) прибрано — повертає Err з інструкцією встановити MKL_LIB_DIR
- [x] **INFRA-02**: `CARGO_MANIFEST_DIR` не embedded в shipped binary — gemm_shim.c шукається відносно `std::env::current_exe()`
- [x] **INFRA-03**: Graph fingerprinting використовує SipHasher (стабільний між Rust versions) замість DefaultHasher

### Reliability

- [x] **RELY-01**: `autograd.rs` panic sites замінені на `Result<_, AutogradError>` для незвичних graph structures
- [x] **RELY-02**: `Tensor::PartialEq` не панікує на невалідних stride/offset — повертає false

## v2 Requirements

### Performance

- **PERF-V2-01**: AutoTuner для tile sizes (MC, KC, NC) — sampling конфігурацій при DLL compile time
- **PERF-V2-02**: JitCache підключений до training loop в `train_graph_with_backend`
- **PERF-V2-03**: Arc<RwLock<Tensor>> для параметрів — уникнути N heap allocations + N tensor copies per optimizer step

### Platform

- **PLAT-V2-01**: Linux/macOS build та тест interpreter path
- **PLAT-V2-02**: ARM/AArch64 підтримка в codegen (без AVX2 hardcode, без MKL залежності)

### CUDA

- **CUDA-V2-01**: CUDA `run()` stubs підключені до реальних kernels або прибрані
- **CUDA-V2-02**: CUDA softmax підтримує >1024 елементів (multi-block reduction)
- **CUDA-V2-03**: Benchmark CUDA проти PyTorch GPU — реальні числа

### Model Coverage

- **MODEL-V2-01**: Conv2D, LayerNorm, MultiHeadAttention в AOT training codegen path

## Out of Scope

| Feature | Reason |
|---------|--------|
| Distributed training (multi-node, NCCL) | Поза фокусом |
| Model zoo / pretrained weights | Поза фокусом |
| Browser/WASM | Поза фокусом |
| HuggingFace Hub | Поза фокусом |
| AVX-512 в generated Cargo.toml | Ламає non-AVX512 CPUs — прибрати x86-v4 feature |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PERF-01 | Phase 1 | Complete |
| PERF-02 | Phase 1 | Complete |
| CORR-01 | Phase 1 | Complete |
| PERF-03 | Phase 2 | Complete |
| CORR-02 | Phase 2 | Complete |
| INFRA-01 | Phase 2 | Complete |
| INFRA-02 | Phase 2 | Complete |
| INFRA-03 | Phase 2 | Complete |
| CORR-03 | Phase 3 | Complete |
| RELY-01 | Phase 3 | Complete |
| RELY-02 | Phase 3 | Complete |

**Coverage:**
- v1 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-07*
*Last updated: 2026-03-07 — traceability updated after roadmap creation*
