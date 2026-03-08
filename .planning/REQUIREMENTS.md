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
- **PERF-V2-02**: Compiled execution reuse в `train_graph_with_backend` доведений і regression-tested — repeated training steps не створюють per-step compile churn
- **PERF-V2-03**: Training path прибирає per-step parameter rewrap churn; стабільні outer parameter handles зберігаються між optimizer steps, навіть якщо execution boundary ще використовує snapshot copies

### Training Reliability

- **TRAIN-V2-01**: CPU training loops для `SGD`, `Adam`, `AdamW` проходять довгі багатокрокові запуски без `NaN`, divergence або nondeterministic drift на контрольних моделях
- **TRAIN-V2-02**: Щонайменше 2 end-to-end training cases (не micro parity) проходять реальне навчання і знижують loss на CPU path
- **TRAIN-V2-03**: Є щонайменше 1 integration-grade CPU training regression case вище за raw IR helpers

### Correctness

- **CORR-V2-01**: PyTorch parity покриває не тільки op-level checks, а й multi-step training loops та реальні mini-model cases

### Platform

- **PLAT-V2-01**: Linux/macOS build та тест interpreter path
- **PLAT-V2-02**: ARM/AArch64 підтримка в codegen (без AVX2 hardcode, без MKL залежності)

### CUDA

- **CUDA-V2-01**: CUDA `run()` stubs підключені до реальних kernels або прибрані
- **CUDA-V2-02**: CUDA softmax підтримує >1024 елементів (multi-block reduction)
- **CUDA-V2-03**: Benchmark CUDA проти PyTorch GPU — реальні числа

### Model Coverage

- **MODEL-V2-01**: Conv2D, LayerNorm, MultiHeadAttention в AOT training codegen path

### Product Surface

- **UX-V2-01**: CLI help/output узгоджені між `run/check/info/compile/compile-train/doctor` і дають корисні actionable повідомлення замість розмитих помилок
- **UX-V2-02**: `doctor` показує capability matrix, maturity, determinism guarantees і явні next steps для unsupported setups
- **UX-V2-03**: Examples і docs відображають реальний supported path, команди перевіряються smoke-тестами

### Distribution

- **DIST-V2-01**: Репозиторій генерує нормальні release artifacts для підтримуваної платформи, а не тільки локальний dev build
- **DIST-V2-02**: Install story перевірений з clean environment і задокументований кроками, які реально працюють
- **DIST-V2-03**: Після install shipped binary проходить smoke-path для `--help`, `doctor` і хоча б одного compile/run сценарію

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
| PERF-V2-02 | Phase 4 | Complete |
| PERF-V2-03 | Phase 4 | Complete |
| TRAIN-V2-01 | Phase 4 | Complete |
| TRAIN-V2-03 | Phase 4 | Complete |
| CORR-V2-01 | Phase 5 | Complete |
| TRAIN-V2-02 | Phase 5 | Complete |
| MODEL-V2-01 | Deferred | Unmapped |
| UX-V2-01 | Phase 6 | Planned |
| UX-V2-02 | Phase 6 | Complete |
| UX-V2-03 | Phase 6 | Planned |
| PLAT-V2-01 | Phase 7 | Complete |
| PLAT-V2-02 | Phase 7 | Complete |
| DIST-V2-01 | Phase 7 | Complete |
| DIST-V2-02 | Phase 7 | Planned |
| DIST-V2-03 | Phase 7 | Planned |

**Coverage:**
- Total requirements: 25
- Mapped to phases: 21
- Unmapped: 4 (`CUDA-V2-01`, `CUDA-V2-02`, `CUDA-V2-03`, `MODEL-V2-01`)

---
*Requirements defined: 2026-03-07*
*Last updated: 2026-03-08 — Phase 5 real-model parity requirements marked complete; `MODEL-V2-01` explicitly deferred until a future AOT/codegen phase*
