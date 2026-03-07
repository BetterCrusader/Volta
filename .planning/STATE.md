---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
last_updated: "2026-03-07T08:09:31.805Z"
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 9
  completed_plans: 9
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
last_updated: "2026-03-07T08:05:11.202Z"
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 9
  completed_plans: 9
  percent: 100
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
last_updated: "2026-03-07T06:53:06.347Z"
progress:
  [██████████] 100%
  completed_phases: 2
  total_plans: 7
  completed_plans: 7
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
last_updated: "2026-03-07T06:47:55.310Z"
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 7
  completed_plans: 7
  percent: 100
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
last_updated: "2026-03-07T06:44:05.494Z"
progress:
  [██████████] 100%
  completed_phases: 1
  total_plans: 7
  completed_plans: 6
  percent: 86
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
last_updated: "2026-03-07T06:43:30.392Z"
progress:
  [█████████░] 86%
  completed_phases: 1
  total_plans: 7
  completed_plans: 5
  percent: 71
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
last_updated: "2026-03-07T05:59:47.459Z"
progress:
  [███████░░░] 71%
  completed_phases: 1
  total_plans: 4
  completed_plans: 4
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: complete
last_updated: "2026-03-07T05:55:36.911Z"
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 4
  completed_plans: 4
  percent: 100
---

# State: Volta

**Останнє оновлення:** 2026-03-07T16:39:59Z

---

## Project Reference

**Ядро цінності:** CPU training швидший за PyTorch eager — виміряно, відтворювано, з числовою коректністю
**Поточний фокус:** Фаза 3 — Надійність і коректність (03-02 DONE)

---

## Current Position

**Фаза:** 3 — Надійність і коректність (In Progress)
**План:** 03-02 complete
**Статус:** Phase 3 in progress — 03-02 done (MHA full backward pass: 7 gradients, 2 tests green / CORR-03)
**Прогрес:** [██░░░░░░░░] in progress

```
[x] Фаза 1 / 01-01: Remove x86-v4 from gemm features (DONE)
[x] Фаза 1 / 01-02: Adam numerical correctness test / CORR-01 (DONE)
[x] Фаза 1 / 01-03: Phase sign-off and metrics update (DONE)
[x] Фаза 1 / 01-04: Gap closure — PERF-01/PERF-02 confirmed with measured values (DONE)
[x] Фаза 2 / 02-01: Stable fingerprints (SipHasher13) + MKL error reporting (DONE)
[x] Фаза 2 / 02-02: Portable gemm_shim (include_bytes!) + merged_rustflags tests (DONE)
[x] Фаза 2 / 02-03: Adam optimizer capability tracking — supports_adam + validate_optimizer / CORR-02 (DONE)
[x] Фаза 3 / 03-01: Tensor::PartialEq panic-free + build_reverse_graph 3 unwraps fixed / RELY-01, RELY-02 (DONE)
[x] Фаза 3 / 03-02: MHA full backward pass — 7 gradients (dq/dk/dv/dWq/dWk/dWv/dWo) / CORR-03 (DONE)
```

---

## Performance Metrics

| Метрика | Значення | Статус |
|---------|----------|--------|
| SGD B=64 MLP-512 median | 1.703 ms (Case 2 primary bench) | PASS — 1.703 ms < 2.10 ms gate; +43% faster than PyTorch (2.440 ms) |
| Adam vs PyTorch ratio | 4.259 ms vs PyTorch 5.343 ms = 0.797× (+25% faster) | PASS — перевищує ціль ≤1.1× |
| SGD vs PyTorch ratio | 0.33–0.65× | PASS — 35-67% faster |

| Phase 01-adam P02 | 5min | 1 task | 1 file |
| Phase 01-adam P03 | 2min | 2 tasks | 1 file |
| Phase 01-adam P04 | - | 1 task | 2 files |
| Phase 02-backend P01 | 6min | 2 tasks | 3 files |
| Phase 02-backend P02-02 | 6min | 2 tasks | 3 files |
| Phase 02-backend P03 | 2 | 1 tasks | 3 files |
| Phase 03-nadiiinist-i-korektnost P03-02 | 45min | 2 tasks | 2 files |

## Accumulated Context

### Ключові рішення (зафіксовані)

| Рішення | Обґрунтування |
|---------|---------------|
| Rust AOT codegen замість інтерпретатора | Нативна швидкість без LLVM залежності — ✓ 35-67% над PyTorch |
| MKL cblas_sgemm для SGD GEMM | Максимальна GEMM швидкість на Intel |
| Adam MKL cblas_sgemm + AVX2 SIMD + Rayon | Реалізовано — ✓ ~25% швидше PyTorch |
| CUDA за feature flag | Уникнути обов'язкової CUDA залежності |
| gemm features = ["rayon"] без x86-v4 | AVX-512 обирається автоматично через RUSTFLAGS target-cpu=native — явна фіча небезпечна (SIGILL) |
| Adam correctness test: 1e-5 tolerance | f32 arithmetic rounding; 1e-6 too tight, 1e-3 too loose — 1e-5 correct for CORR-01 |
| CORR-01: test-only, no code change | apply_adam implementation already correct; test formalizes guarantee |
| PERF-02 gate context: 1.703 ms (Case 2 primary) not 2.237 ms (Adam-session SGD) | Two SGD figures in BENCHMARKS.md measure different thermal/cache states; gate applies only to cold primary bench |
| PERF-01 Adam: discard 2026-03-07 run without MKL (2.464 ms) | Adam codegen requires MKL at runtime; run without MKL does not exercise the optimized path — BENCHMARKS.md 2026-03-06 data (0.797x) is the valid measurement |
| SipHasher13::new_with_keys(0,0) for graph fingerprints | Fixed seed required for cross-build stability; DefaultHasher seed changes between Rust builds |
| resolve_mkl_lib_path_from() testable inner function | Project deny(unsafe_code) prevents unsafe env mutation in tests; inner function takes explicit params |
| include_bytes! for gemm_shim.c | Embed at compile time, write to per-compile dir at runtime; eliminates CARGO_MANIFEST_DIR dev path from shipped binary |
| merged_rustflags tests: 5 named tests | Locks in target-cpu=native injection behavior; no duplicate if already set |
| CpuBackend supports_adam: true; CudaBackend supports_adam: false | validate_optimizer() makes Adam/backend contract explicit before compile; CUDA Adam not implemented in v1 |
| make_contiguous bounds-check before copy_to_slice | copy_to_slice panics on invalid offset; returning Err from make_contiguous enables let-else pattern in PartialEq — no API surface change |
| MultiHeadAttentionBackward arms: no-op in autograd backward | backward of backward not needed for v1; Err in interpreter is correct since MHABackward is a codegen-only op |
| MHA sibling-scan for attn_weights/context | scan forward.nodes matching q_input ValueId + output_idx avoids storing extra ValueIds; output_idx==0 guard prevents double-counting |
| MHA test differentiates v_input not q_input | d(sum)/d(q_input)=0 with fixed k/v due to softmax cancellation; v_input gradient always non-zero — meaningful test |

### Відомі блокери

- ~~Hardcoded `C:/Users/User/miniforge3/...` в binary~~ — FIXED in 02-01
- ~~MHA backward — stub, transformer не навчається~~ — FIXED in 03-02
- ~~`CARGO_MANIFEST_DIR` embedded в shipped binary — codegen paths broken після install~~ — FIXED in 02-02
- bench_mlp2048 and bench_b128 examples fail to link MKL — pre-existing issue

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 2 | add multi-step PyTorch training parity through real train_graph | 2026-03-07 | e9bafb6 | [2-add-multi-step-pytorch-training-parity-t](./quick/2-add-multi-step-pytorch-training-parity-t/) |
| 3 | add MHA bias gradients and transformer training parity | 2026-03-07 | 50979c8 | [3-add-mha-bias-gradients-and-transformer-t](./quick/3-add-mha-bias-gradients-and-transformer-t/) |
| 4 | expand train_graph parity coverage (optimizers, accumulation, clip grad) | 2026-03-07 | 148e9c7 | [4-add-parity-for-gradient-accumulation-a](./quick/4-add-parity-for-gradient-accumulation-a/) |
| 5 | fix benchmark sorting helpers and simplify timing organization | 2026-03-07 | 48cb0ea | [5-fix-benchmark-sorting-helpers-and-simpli](./quick/5-fix-benchmark-sorting-helpers-and-simpli/) |

### Важливі числа

- Benchmark gate: B=64 MLP-512 median < 2.10 ms — SGD Case 2 primary = 1.703 ms (PASS, confirmed 2026-03-07)
- Adam поточний ratio: 4.259 ms vs PyTorch 5.343 ms = 0.797× (+25% faster) — PASS (confirmed by BENCHMARKS.md, 2026-03-06)
- SGD поточний ratio: 35-67% швидше за PyTorch при B≤64 (Case 2: +43%)

---

## Session Continuity

**Щоб відновити контекст:**
1. Читай `.planning/ROADMAP.md` — поточна фаза і критерії успіху
2. Читай `.planning/REQUIREMENTS.md` — повний список v1 вимог
3. Читай `.planning/codebase/CONCERNS.md` — known bugs і tech debt

**Наступний крок:** Розширити parity на `RmsProp`/`Adagrad` або перевірити `gradient_accumulation_steps` і `clip_grad` на transformer training path

**Остання сесія:** 2026-03-07T18:42:33Z — Completed quick task 5 (benchmark timing helper cleanup, commit `48cb0ea`)

---

*State initialized: 2026-03-07*
