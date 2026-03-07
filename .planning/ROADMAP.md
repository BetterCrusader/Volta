# Roadmap: Volta

**Проект:** Volta — компілятор і runtime для нейронних мереж
**Ядро цінності:** CPU training швидший за PyTorch eager — виміряно, відтворювано, з числовою коректністю
**Granularity:** coarse
**Покриття вимог:** 11/11 ✓

---

## Phases

- [ ] **Фаза 1: Продуктивність Adam** — Закрити gap Adam vs PyTorch, утримати SGD gate
- [ ] **Фаза 2: Інфраструктура і backend** — Прибрати hardcoded paths, propagate RUSTFLAGS, capabilities matrix
- [ ] **Фаза 3: Надійність і коректність** — MHA backward, паніки → Result, Tensor::PartialEq

---

## Phase Details

### Phase 1: Продуктивність Adam
**Goal:** Adam optimizer досягає паритету з PyTorch (≤1.1×) при збереженні числової коректності та SGD regression gate
**Залежить від:** нічого (перша фаза)
**Requirements:** PERF-01, PERF-02, CORR-01
**Success Criteria** (що має бути ПРАВДОЮ після фази):
  1. Бенчмарк Adam MLP B≤64 показує ≤1.1× відносно PyTorch eager — вимірювано тим самим harness (30 outer × 7 inner × 50 steps)
  2. SGD regression gate утримується: B=64 MLP-512 median < 2.10 ms після всіх змін optimizer
  3. Числовий регресійний тест Adam проходить проти reference implementation — градієнти відповідають до eps
**Plans:** 4/4 plans complete

Plans:
- [x] 01-01-PLAN.md — AVX-512 safety fix: remove x86-v4 from gemm features in codegen template and benchmark crate
- [x] 01-02-PLAN.md — Adam correctness test: add adam_updates_parameter_numerically_correct unit test (CORR-01)
- [x] 01-03-PLAN.md — SGD gate verification + STATE.md metrics update (benchmark checkpoint)
- [ ] 01-04-PLAN.md — Gap closure: human benchmark run to confirm PERF-01 and PERF-02 with actual measured values

### Phase 2: Інфраструктура і backend
**Goal:** Volta запускається на будь-якій машині без ручних env-vars, codegen використовує native CPU оптимізації, backend відмовляє явно для непідтримуваних комбінацій
**Залежить від:** Phase 1
**Requirements:** PERF-03, CORR-02, INFRA-01, INFRA-02, INFRA-03
**Success Criteria** (що має бути ПРАВДОЮ після фази):
  1. `volta compile-train --rust` на чистій машині без `MKL_LIB_DIR` повертає зрозумілу помилку з інструкцією — замість panic або silent link failure
  2. `volta compile-train --rust` без ручного RUSTFLAGS компілює DLL з `target-cpu=native` автоматично
  3. Shipped binary `volta compile-train` не містить `C:/Users/User/...` шляху — `gemm_shim.c` знаходиться відносно `current_exe()`
  4. Graph fingerprint стабільний між Rust compiler versions — один і той же граф дає один і той же hash при різних `rustc`
  5. `CpuBackend::capabilities()` повертає явну матрицю; спроба Adam на unsupported backend повертає `Err` замість silent fallback
**Plans:** 3/3 plans complete

Plans:
- [ ] 02-01-PLAN.md — SipHasher swap (fingerprint.rs + backend.rs) + MKL error handling (resolve_mkl_lib_path → Result)
- [ ] 02-02-PLAN.md — gemm_shim.c portability (include_bytes! replaces CARGO_MANIFEST_DIR at both sites) + merged_rustflags tests
- [ ] 02-03-PLAN.md — BackendCapabilities: add supports_adam field + validate_optimizer() method

### Phase 3: Надійність і коректність
**Goal:** Transformer архітектури навчаються коректно, аварійні виходи в бібліотечному коді замінені на Result
**Залежить від:** Phase 2
**Requirements:** CORR-03, RELY-01, RELY-02
**Success Criteria** (що має бути ПРАВДОЮ після фази):
  1. Модель з `MultiHeadAttention` навчається і знижує loss — перевіряється тестом на 100 кроків з порівнянням градієнтів Q/K/V проти PyTorch reference
  2. `build_reverse_graph()` на незвичних graph structures (shared outputs) повертає `Err(AutogradError)` — не панікує
  3. `Tensor::PartialEq` на тензорі з невалідними stride/offset повертає `false` — не панікує
**Plans:** 1/2 plans executed

Plans:
- [ ] 03-01-PLAN.md — RELY-01 + RELY-02: fix 4 panic sites in tensor.rs and autograd.rs → ok_or_else / let-else
- [ ] 03-02-PLAN.md — CORR-03: full MHA backward (Op::MultiHeadAttentionBackward + interpreter + autograd arm + tests)

---

## Progress Table

| Фаза | Планів виконано | Статус | Завершено |
|------|-----------------|--------|-----------|
| 1. Продуктивність Adam | 4/4 | Complete   | 2026-03-07 |
| 2. Інфраструктура і backend | 3/3 | Complete   | 2026-03-07 |
| 3. Надійність і коректність | 1/2 | In Progress|  |

---

*Roadmap created: 2026-03-07*
*Last updated: 2026-03-07 — Phase 3 plans created (03-01 through 03-02)*
