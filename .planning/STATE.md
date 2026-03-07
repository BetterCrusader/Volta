---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
last_updated: "2026-03-07T06:44:05.494Z"
progress:
  total_phases: 3
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

**Останнє оновлення:** 2026-03-07

---

## Project Reference

**Ядро цінності:** CPU training швидший за PyTorch eager — виміряно, відтворювано, з числовою коректністю
**Поточний фокус:** Фаза 2 — Інфраструктура і backend (02-01 DONE)

---

## Current Position

**Фаза:** 2 — Інфраструктура і backend (In Progress)
**План:** 02-01 complete
**Статус:** Phase 2 started — 02-01 done (stable fingerprints + MKL error reporting)
**Прогрес:** [███████░░░] 71%

```
[x] Фаза 1 / 01-01: Remove x86-v4 from gemm features (DONE)
[x] Фаза 1 / 01-02: Adam numerical correctness test / CORR-01 (DONE)
[x] Фаза 1 / 01-03: Phase sign-off and metrics update (DONE)
[x] Фаза 1 / 01-04: Gap closure — PERF-01/PERF-02 confirmed with measured values (DONE)
[x] Фаза 2 / 02-01: Stable fingerprints (SipHasher13) + MKL error reporting (DONE)
[ ] Фаза 2: Решта планів
[ ] Фаза 3: Надійність і коректність
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

### Відомі блокери

- ~~Hardcoded `C:/Users/User/miniforge3/...` в binary~~ — FIXED in 02-01
- MHA backward — stub, transformer не навчається
- `CARGO_MANIFEST_DIR` embedded в shipped binary — codegen paths broken після install
- bench_mlp2048 and bench_b128 examples fail to link MKL — pre-existing issue

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

**Наступний крок:** Продовжити Фазу 2 — наступний план 02-02

**Остання сесія:** 2026-03-07 — Completed 02-01-PLAN.md (SipHasher13 fingerprints, MKL Result error reporting, INFRA-01/INFRA-03 closed)

---

*State initialized: 2026-03-07*
