# Roadmap: Volta

**Проект:** Volta — компілятор і runtime для нейронних мереж
**Ядро цінності:** CPU training швидший за PyTorch eager — виміряно, відтворювано, з числовою коректністю
**Granularity:** coarse
**Покриття вимог:** 21/25 mapped

---

## Phases

- [x] **Фаза 1: Продуктивність Adam** — Закрити gap Adam vs PyTorch, утримати SGD gate
- [x] **Фаза 2: Інфраструктура і backend** — Прибрати hardcoded paths, propagate RUSTFLAGS, capabilities matrix
- [x] **Фаза 3: Надійність і коректність** — MHA backward, паніки → Result, Tensor::PartialEq
- [x] **Фаза 4: CPU training path hardening and long-loop stability** — Довгі train loops, stability, runtime/training overhead cleanup
- [x] **Фаза 5: End-to-end PyTorch parity and real model training cases** — Не тільки micro parity, а реальні ConvNet/tiny-transformer training cases
- [ ] **Фаза 6: Product surface hardening for CLI doctor examples and docs** — CLI/help/doctor/examples/docs без розсинхрону
- [ ] **Фаза 7: Packaging and install story for distributable releases** — Release artifacts, clean install, smoke-tested shipped binaries

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
- [x] 01-04-PLAN.md — Gap closure: human benchmark run to confirm PERF-01 and PERF-02 with actual measured values

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
- [x] 02-01-PLAN.md — SipHasher swap (fingerprint.rs + backend.rs) + MKL error handling (resolve_mkl_lib_path → Result)
- [x] 02-02-PLAN.md — gemm_shim.c portability (include_bytes! replaces CARGO_MANIFEST_DIR at both sites) + merged_rustflags tests
- [x] 02-03-PLAN.md — BackendCapabilities: add supports_adam field + validate_optimizer() method

### Phase 3: Надійність і коректність
**Goal:** Transformer архітектури навчаються коректно, аварійні виходи в бібліотечному коді замінені на Result
**Залежить від:** Phase 2
**Requirements:** CORR-03, RELY-01, RELY-02
**Success Criteria** (що має бути ПРАВДОЮ після фази):
  1. Модель з `MultiHeadAttention` навчається і знижує loss — перевіряється тестом на 100 кроків з порівнянням градієнтів Q/K/V проти PyTorch reference
  2. `build_reverse_graph()` на незвичних graph structures (shared outputs) повертає `Err(AutogradError)` — не панікує
  3. `Tensor::PartialEq` на тензорі з невалідними stride/offset повертає `false` — не панікує
**Plans:** 2/2 plans complete

Plans:
- [x] 03-01-PLAN.md — RELY-01 + RELY-02: fix 4 panic sites in tensor.rs and autograd.rs → ok_or_else / let-else
- [x] 03-02-PLAN.md — CORR-03: full MHA backward (Op::MultiHeadAttentionBackward + interpreter + autograd arm + tests)

---

### Phase 4: CPU training path hardening and long-loop stability
**Goal:** CPU training path тримає довші train loops стабільно і без прихованого runtime debt, а не тільки проходить короткі parity smoke tests
**Залежить від:** Phase 3
**Requirements:** PERF-V2-02, PERF-V2-03, TRAIN-V2-01, TRAIN-V2-03
**Success Criteria** (що має бути ПРАВДОЮ після фази):
  1. `SGD`, `Adam`, `AdamW` проходять довгі контрольні CPU training loops без `NaN`, divergence або flaky drift на однаковому seed
  2. Training runtime не робить очевидно зайвих пер-крокових realloc/copy bottlenecks на параметрах і compiled path використовує cache там, де це вже підтримується архітектурою
  3. Є хоча б один end-to-end regression case, який валиться при stability/regression поломці довгого training path
**Plans:** 3/3 plans complete

Plans:
- [x] 04-01-PLAN.md — Harden `train_graph_with_backend`: fail-fast non-finite guards, deterministic long-loop checks, and consistent early-stopping restore semantics
- [x] 04-02-PLAN.md — Add one compiled-MLP long-loop integration regression in `train_api.rs` and mirror it with a matching PyTorch oracle
- [x] 04-03-PLAN.md — Lock down compile-reuse behavior and remove per-step parameter rewrap churn in the CPU training path

### Phase 5: End-to-end PyTorch parity and real model training cases
**Goal:** Volta доводить коректність не тільки на op-level parity, а на реальних mini-model training cases проти PyTorch
**Залежить від:** Phase 4
**Requirements:** CORR-V2-01, TRAIN-V2-02
**Success Criteria** (що має бути ПРАВДОЮ після фази):
  1. Щонайменше 2 реальні model cases (наприклад, ConvNet і tiny transformer) проходять multi-step training parity проти PyTorch
  2. Loss curves, key gradients і фінальні параметри узгоджуються в допустимому eps для зафіксованих deterministic test cases
  3. Ці кейси живуть як автоматичні regression tests, а не як разовий локальний запуск
**Traceability Note:** `MODEL-V2-01` свідомо винесений з Phase 5. Поточний AOT training codegen MLP-only; ця фаза закриває parity breadth і truth-pass, але не general AOT model coverage.
**Plans:** 3/3 plans complete

Plans:
- [x] 05-01-PLAN.md — Add one real compiled-model ConvNet training case in `train_api` and mirror it with PyTorch parity
- [x] 05-02-PLAN.md — Replace fake tiny-transformer confidence with one honest compiled attention/norm/FFN mini-model plus PyTorch parity
- [x] 05-03-PLAN.md — Add early unsupported-path regression for non-MLP `compile-train` and repair roadmap/traceability if `MODEL-V2-01` is still unmet

### Phase 6: Product surface hardening for CLI doctor examples and docs
**Goal:** CLI, doctor, examples і docs говорять одну й ту саму правду про supported path, limitations і next steps
**Залежить від:** Phase 5
**Requirements:** UX-V2-01, UX-V2-02, UX-V2-03
**Success Criteria** (що має бути ПРАВДОЮ після фази):
  1. `--help`/usage/output для основних команд не суперечать реальній поведінці
  2. `doctor` показує capability matrix і actionable diagnostics замість сирого технічного шуму
  3. README/examples/docs smoke-перевіряються й не ведуть користувача в dead path
**Plans:** 4 plans

Plans:
- [ ] 06-01-PLAN.md — Wave 0: Create tests/cli_smoke.rs integration test scaffold (6 smoke test stubs, CARGO_BIN_EXE_volta pattern)
- [ ] 06-02-PLAN.md — Wave 1: Fix stale README claims (Adam benchmark, roadmap snapshot) and USAGE/docs/ROADMAP.md sync (UX-V2-01)
- [ ] 06-03-PLAN.md — Wave 1: Rewrite doctor — capability matrix table, MKL/LLVM env detection, Next Steps section (UX-V2-02)
- [ ] 06-04-PLAN.md — Wave 2: Fix misleading example .vt comments and tighten cli_smoke assertions (UX-V2-03)

### Phase 7: Packaging and install story for distributable releases
**Goal:** Volta можна не тільки запускати локально з репо, а й нормально зібрати, роздати і встановити на чисту машину
**Залежить від:** Phase 6
**Requirements:** PLAT-V2-01, PLAT-V2-02, DIST-V2-01, DIST-V2-02, DIST-V2-03
**Success Criteria** (що має бути ПРАВДОЮ після фази):
  1. Є задокументований release flow і artifacts для підтримуваної платформи
  2. Clean install story перевірений не з dev shell, а з чистого сценарію
  3. Shipped binary проходить базові smoke checks (`help`, `doctor`, compile/run path) після install
**Plans:** 0/0 plans

Plans:
- [ ] TBD (`$gsd-plan-phase 7`)

---

## Progress Table

| Фаза | Планів виконано | Статус | Завершено |
|------|-----------------|--------|-----------|
| 1. Продуктивність Adam | 4/4 | Complete | 2026-03-07 |
| 2. Інфраструктура і backend | 3/3 | Complete | 2026-03-07 |
| 3. Надійність і коректність | 2/2 | Complete | 2026-03-07 |
| 4. CPU training path hardening and long-loop stability | 3/3 | Complete | 2026-03-08 |
| 5. End-to-end PyTorch parity and real model training cases | 3/3 | Complete | 2026-03-08 |
| 6. Product surface hardening for CLI doctor examples and docs | 0/4 | In Progress | - |
| 7. Packaging and install story for distributable releases | 0/0 | Planned | - |

---

*Roadmap created: 2026-03-07*
*Last updated: 2026-03-08 — Phase 6 planned: 4 plans across 3 waves for CLI/doctor/examples/docs hardening*
