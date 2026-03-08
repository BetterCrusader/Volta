---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-03-08T09:58:24.269Z"
progress:
  total_phases: 7
  completed_phases: 6
  total_plans: 19
  completed_plans: 19
---

# State: Volta

**Останнє оновлення:** 2026-03-08T03:35:00+02:00

---

## Project Reference

**Ядро цінності:** CPU training швидший за PyTorch eager — виміряно, відтворювано, з числовою коректністю
**Поточний фокус:** Phase 6 planning — product surface hardening for CLI/doctor/examples/docs

---

## Current Position

**Фаза:** 6 — Product surface hardening for CLI doctor examples and docs (COMPLETE)
**План:** 06-04 DONE — Phase 6 complete
**Статус:** 06-04 complete — misleading .vt example comments corrected; all 6 cli_smoke tests pass; 19/19 plans complete
**Прогрес:** [██████████] 100% — 19/19 plans complete

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
[x] Фаза 5 / 05-01: compiled-model ConvNet parity above `train_api` (DONE)
[x] Фаза 5 / 05-02: honest tiny-transformer compiled-model parity (DONE)
[x] Фаза 5 / 05-03: AOT model-coverage truth pass / unsupported regression gates (DONE)
[x] Фаза 6: Product surface hardening (COMPLETE — 06-01..06-04 done)
    [x] 06-01: CLI smoke test scaffold — 6-function integration test (DONE)
    [x] 06-02: README/USAGE/docs accuracy corrections (DONE)
    [x] 06-03: volta doctor rewrite — MKL/LLVM diagnostics + capability matrix (DONE)
    [x] 06-04: misleading example .vt comments corrected + cli_smoke tightened (DONE)
[ ] Фаза 7: Packaging і install story (PLANNED)
```

---

## Performance Metrics

| Метрика | Значення | Статус |
|---------|----------|--------|
| SGD B=64 MLP-512 median | 1.703 ms (Case 2 primary bench) | PASS — 1.703 ms < 2.10 ms gate; +43% faster than PyTorch (2.440 ms) |
| Adam vs PyTorch ratio | 4.259 ms vs PyTorch 5.343 ms = 0.797× (+25% faster) | PASS — перевищує ціль ≤1.1× |
| SGD vs PyTorch ratio | 0.33–0.65× | PASS — 35-67% faster |
| Phase 06 P01 CLI smoke scaffold | 8 min | 1 task | 1 file |
| Phase 06 P02 README/USAGE/docs accuracy | 8 min | 2 tasks | 3 files |
| Phase 06 P03 | 12 | 2 tasks | 2 files |
| Phase 06 P04 | 5 | 2 tasks | 3 files |

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
| CLI smoke tests use CARGO_BIN_EXE_volta (not cargo run) | Zero-rebuild binary access; temp .vt with 'use fn' model syntax triggers MLP-only rejection in compile_first_model_to_train_dll |
| README Adam claim reflects measured result, not aspirational goal | Changed from "1.9× slower" to "+25% faster at B≤64" — public text must match STATE.md metrics |
| docs/ROADMAP.md rewritten to reflect executed phases | Old content referenced wrong internal phase names and stale next-steps throughout |
| check_mkl_from() duplicated inline (not calling private mlp_train_rust_codegen fn) | Plan explicitly forbade calling private fn; duplicate is ~10 lines and testable via injection |
| MKL-not-found adds actionable warning making healthy:false | Correct behaviour — Adam/AdamW --rust will not link without MKL; user needs actionable guidance |
| cli_smoke.rs Task 2 was already complete from plan 06-03 | 06-03 auto-fixed smoke_doctor and doctor_json_fields assertions when rewriting print_doctor output; no 06-04 changes needed |
| Example file comments corrected in-place without touching model definitions | Only headers changed — layers/activation/optimizer lines untouched; files still parse and compile-train correctly |

### Відомі ризики

- Phase 5 broadened compiled-model parity to ConvNet and honest tiny-transformer cases, but product surface and packaging still lag behind code reality
- Product surface досі місцями відстає від реальної поведінки коду
- Clean install / release story не перевірені як готовий продукт
- CUDA v2 requirements лишаються unmapped і не входять у поточний phase train
- `MODEL-V2-01` все ще не виконаний: AOT training codegen зараз MLP-only, а Phase 5 спеціально закрив truth-pass замість fake closure

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
4. Для наступного planning кроку враховуй summaries `05-01`, `05-02`, `05-03` і verification `05-VERIFICATION.md`

**Наступний крок:** виконати Wave 1 плани Phase 6 (06-03 examples/CLI fixes)

---

*State refreshed: 2026-03-08 after 06-02 execution — README/USAGE/docs accuracy corrections*
