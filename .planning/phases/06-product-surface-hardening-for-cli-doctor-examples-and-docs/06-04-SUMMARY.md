---
phase: 06-product-surface-hardening-for-cli-doctor-examples-and-docs
plan: 04
subsystem: examples
tags: [examples, cli-smoke, vt-files, documentation, ux]

requires:
  - phase: 06-01
    provides: CLI smoke test scaffold (6-function integration test binary)
  - phase: 06-03
    provides: volta doctor rewrite with MKL/LLVM diagnostics and capability matrix

provides:
  - cnn_classifier.vt with honest "Dense MLP" comment replacing misleading "CNN-style" header
  - transformer_ffn.vt with honest "Dense MLP with GELU" comment replacing misleading "Transformer FFN" header
  - resnet_block.vt with honest "Dense MLP" comment replacing misleading "ResNet-style" header
  - All 6 cli_smoke tests passing including tightened doctor assertions and compile-train rejection test

affects: [packaging, 06-05, user-facing-docs]

tech-stack:
  added: []
  patterns:
    - Honest example annotations: example files must match what the runtime actually supports, not aspirational names

key-files:
  created: []
  modified:
    - examples/cnn_classifier.vt
    - examples/transformer_ffn.vt
    - examples/resnet_block.vt

key-decisions:
  - "cli_smoke.rs Task 2 was already complete from plan 06-03 — no changes needed"
  - "Example file comments corrected in-place without touching model definition lines"

patterns-established:
  - "Example files must not imply unsupported architectures (Conv2D, attention, residual) in their comments"

requirements-completed: [UX-V2-03]

duration: 5min
completed: 2026-03-08
---

# Phase 6 Plan 04: Example Comments and CLI Smoke Summary

**Three misleading example .vt file headers corrected to honest "Dense MLP" annotations; all 6 cli_smoke tests confirmed passing with tightened doctor and compile-train assertions**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-08T10:00:00Z
- **Completed:** 2026-03-08T10:05:00Z
- **Tasks:** 2 (Task 1 executed; Task 2 was already complete from 06-03)
- **Files modified:** 3

## Accomplishments

- `examples/cnn_classifier.vt` header replaced: "CNN-style classifier" -> "Dense MLP classifier (784->512->256->128->10)" with explicit note that Conv2D is not supported
- `examples/transformer_ffn.vt` header replaced: "Transformer FFN block" -> "Dense MLP with GELU activation (256->1024->256)" with explicit note that MultiHeadAttention is not used
- `examples/resnet_block.vt` header replaced: "ResNet-style block" -> "Dense MLP (512->512->512)" with explicit note that there are no residual connections
- All 6 cli_smoke tests pass: smoke_version, smoke_doctor, doctor_json_fields, smoke_run_xor, smoke_check_bench_real, smoke_compile_train_rejects_non_mlp
- Full test suite passes (no regressions)

## Task Commits

1. **Task 1: Correct misleading comments in three example .vt files** - `9abfa4c` (fix)
2. **Task 2: Tighten cli_smoke tests** - already complete in `1a8e47d` (plan 06-03 commit)

## Files Created/Modified

- `examples/cnn_classifier.vt` - Header comment corrected to honest Dense MLP annotation
- `examples/transformer_ffn.vt` - Header comment corrected to honest Dense MLP annotation
- `examples/resnet_block.vt` - Header comment corrected to honest Dense MLP annotation

## Decisions Made

- cli_smoke.rs Task 2 was already fully implemented in plan 06-03: all tightened assertions (smoke_doctor 5-section check, doctor_json_fields mkl_available/llvm_available, smoke_compile_train_rejects_non_mlp with temp .vt and `use fn` rejection path) were committed in 1a8e47d. No changes needed.
- Example file model definition lines (layers, activation, optimizer, dataset, train) left untouched — only comment headers modified.

## Deviations from Plan

### No New Deviations

Task 2 of this plan (tighten cli_smoke tests) was already executed as part of plan 06-03 (auto-fix deviation 1a8e47d). When 06-03 rewrote print_doctor output, it simultaneously updated cli_smoke.rs with all required assertions. The plan 06-04 Task 2 work was pre-empted.

None - Task 1 executed exactly as written. Task 2 was already complete.

## Issues Encountered

None — all tests green on first attempt.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All three misleading example .vt file headers corrected
- UX-V2-03 satisfied: users reading cnn_classifier.vt, transformer_ffn.vt, resnet_block.vt now get honest expectations about supported architectures
- Phase 6 complete (06-01..06-04 done)
- Phase 7: Packaging and install story is next

---
*Phase: 06-product-surface-hardening-for-cli-doctor-examples-and-docs*
*Completed: 2026-03-08*
