---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: reliability-product-packaging
status: executing
last_updated: "2026-03-08T00:56:03+02:00"
progress:
  total_phases: 7
  completed_phases: 4
  total_plans: 12
  completed_plans: 12
  percent: 100
---

# State: Volta

**Останнє оновлення:** 2026-03-08T00:56:03+02:00

---

## Project Reference

**Ядро цінності:** CPU training швидший за PyTorch eager — виміряно, відтворювано, з числовою коректністю
**Поточний фокус:** Phase 4 complete on branch; next step is Phase 5 planning

---

## Current Position

**Фаза:** 4 — CPU training path hardening (complete)
**План:** 04-01, 04-02, 04-03 complete
**Статус:** Phase 4 complete on current branch; 04-02 added compiled-MLP long-loop integration + PyTorch oracle and 04-03 closed compile-reuse / stable-handle debt
**Прогрес:** [██████████] Phase 4 complete

```
[x] Фаза 1 / 01-01: Remove x86-v4 from gemm features (DONE)
[x] Фаза 1 / 01-02: Adam numerical correctness test / CORR-01 (DONE)
[x] Фаза 1 / 01-03: Phase sign-off and metrics update (DONE)
[x] Фаза 1 / 01-04: Gap closure — PERF-01/PERF-02 confirmed with measured values (DONE)
[x] Фаза 2 / 02-01: Stable fingerprints (SipHasher13) + MKL error reporting (DONE)
[x] Фаза 2 / 02-02: Portable gemm_shim (include_bytes!) + merged_rustflags tests (DONE)
[x] Фаза 2 / 02-03: Adam optimizer capability tracking — supports_adam + validate_optimizer / CORR-02 (DONE)
[x] Фаза 3 / 03-01: Tensor::PartialEq panic-free + build_reverse_graph unwraps fixed / RELY-01, RELY-02 (DONE)
[x] Фаза 3 / 03-02: MHA full backward pass — 7 gradients / CORR-03 (DONE)
[x] Фаза 4: CPU training path hardening (DONE: 04-01..04-03)
[ ] Фаза 5: End-to-end parity і real model cases (PLANNED)
[ ] Фаза 6: Product surface hardening (PLANNED)
[ ] Фаза 7: Packaging і install story (PLANNED)
```

---

## Performance Metrics

| Метрика | Значення | Статус |
|---------|----------|--------|
| SGD B=64 MLP-512 median | 1.703 ms (Case 2 primary bench) | PASS — 1.703 ms < 2.10 ms gate; +43% faster than PyTorch (2.440 ms) |
| Adam vs PyTorch ratio | 4.259 ms vs PyTorch 5.343 ms = 0.797× (+25% faster) | PASS — перевищує ціль ≤1.1× |
| SGD vs PyTorch ratio | 0.33–0.65× | PASS — 35-67% faster |

---

## Accumulated Context

### Roadmap Evolution

- Phase 4 added: CPU training path hardening
- Phase 5 added: End-to-end parity і real model cases
- Phase 6 added: Product surface hardening
- Phase 7 added: Packaging і install story

### Ключові рішення

| Рішення | Обґрунтування |
|---------|---------------|
| Rust AOT codegen замість інтерпретатора | Нативна швидкість без LLVM залежності; CPU path уже показав виміряний виграш |
| MKL cblas_sgemm для CPU GEMM | Максимальна практична швидкість на поточній Windows/Intel цілі |
| CUDA claims не роздувати | GPU correctness/perf не доведені, тому це не current milestone |
| Backends мають фейлити явно | Capability matrix і рання відмова важливіші за silent fallback |
| PyTorch parity — головний oracle | Внутрішні тести самі по собі не доводять математичну коректність training path |
| train_graph fail-fast на non-finite | Loss, gradients, parameters і optimizer buffers тепер валяться stage-specific `TrainError`, а не дрейфують мовчки |
| Early stopping restore має бути full snapshot | Повертаються не лише weights, а й `optimizer_state`, `final_loss`, `final_val_loss` з того самого winning epoch |
| Canonical Phase 4 regression case = compiled-model MLP у `train_api.rs` | Старий linear smoke занадто слабкий; serious regression gate має жити вище raw IR helpers |
| Long-loop parity must mirror the exact Rust regression fixture | Один і той самий seed/dataset/model/loop depth у `train_api.rs` і PyTorch oracle знижує шанс false confidence |
| Compile reuse треба доводити тестом, а не припускати | 04-03 показав, що existing `plan_cache` already covers CPU training loop; правильний хід — зафіксувати це regression test'ом |
| Stable handles лишаються локальними для training path | `Arc<RwLock<Tensor>>` живе всередині training loop; runtime/backend ABI не ламались, а snapshot boundary лишився явним |

### Відомі ризики

- Narrow compiled-MLP long-loop regression уже доведений; ширша real-model parity і model-family breadth досі deferred to Phase 5
- Product surface досі місцями відстає від реальної поведінки коду
- Clean install / release story не перевірені як готовий продукт
- CUDA v2 requirements лишаються unmapped і не входять у поточний phase train

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 2 | add multi-step PyTorch training parity through real train_graph | 2026-03-07 | e9bafb6 | [2-add-multi-step-pytorch-training-parity-t](./quick/2-add-multi-step-pytorch-training-parity-t/) |
| 3 | add MHA bias gradients and transformer training parity | 2026-03-07 | 50979c8 | [3-add-mha-bias-gradients-and-transformer-t](./quick/3-add-mha-bias-gradients-and-transformer-t/) |
| 4 | expand train_graph parity coverage (optimizers, accumulation, clip grad) | 2026-03-07 | 148e9c7 | [4-add-parity-for-gradient-accumulation-a](./quick/4-add-parity-for-gradient-accumulation-a/) |
| 5 | fix benchmark sorting helpers and simplify timing organization | 2026-03-07 | 48cb0ea | [5-fix-benchmark-sorting-helpers-and-simpli](./quick/5-fix-benchmark-sorting-helpers-and-simpli/) |

---

## Session Continuity

**Щоб відновити контекст:**
1. Читай `.planning/ROADMAP.md`
2. Читай `.planning/REQUIREMENTS.md`
3. Читай `.planning/codebase/CONCERNS.md`
4. Для наступного planning кроку враховуй summaries `04-02` і `04-03`: Phase 4 completed narrow long-loop regression + training-path reuse hardening

**Наступний крок:** планувати `Phase 5`; Phase 4 execution на поточній гілці завершений

---

*State refreshed: 2026-03-08 after Phase 4 plan 04-02 completion and current-branch Phase 4 sign-off*
