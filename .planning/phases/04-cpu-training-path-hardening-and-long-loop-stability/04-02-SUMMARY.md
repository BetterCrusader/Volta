---
phase: 04-cpu-training-path-hardening-and-long-loop-stability
plan: "04-02"
subsystem: testing
tags: [rust, pytorch, cpu-training, mlp, determinism]
requires:
  - phase: 04-cpu-training-path-hardening-and-long-loop-stability
    provides: 04-01 non-finite fail-fast guards and deterministic long-loop train_graph behavior
provides:
  - compiled-model MLP long-loop regression coverage in train_api.rs
  - PyTorch oracle coverage for the same narrow MLP long-loop case
  - explicit deterministic gates for SGD, Adam, and AdamW on the shared Phase 4 fixture
affects: [phase-4, phase-5, pytorch-parity, cpu-training]
tech-stack:
  added: []
  patterns: [shared narrow MLP regression fixture, explicit optimizer-specific long-loop test naming]
key-files:
  created: [.planning/phases/04-cpu-training-path-hardening-and-long-loop-stability/04-02-SUMMARY.md]
  modified: [src/engine/model/train_api.rs, tests/pytorch_parity.rs, examples/pytorch_parity.py]
key-decisions:
  - "Phase 4 uses one explicit compiled-model MLP fixture instead of reusing the old linear smoke case."
  - "PyTorch parity stays constrained to that same MLP long-loop regression case rather than expanding model breadth."
patterns-established:
  - "Compiled-model regression fixtures should prove deterministic repeated runs and loss reduction on the same seeded dataset."
  - "PyTorch parity cases should mirror the exact optimizer family, seed, dataset, and loop depth used by the Rust integration fixture."
requirements-completed: [TRAIN-V2-01, TRAIN-V2-03]
duration: 13min
completed: 2026-03-08
---

# Phase 4 04-02 Summary

**Compiled-model MLP long-loop regression coverage now exists in both `train_api.rs` and the PyTorch parity harness for the same seeded CPU case**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-08T00:42:00+02:00
- **Completed:** 2026-03-08T00:54:37+02:00
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added one explicit compiled-model MLP fixture above raw IR helpers and made it the Phase 4 long-loop regression gate for `SGD`, `Adam`, and `AdamW`
- Proved repeated Rust runs stay deterministic on that exact fixture by checking identical `final_loss` and identical final parameter tensors
- Extended PyTorch parity with matching 24-epoch MLP long-loop reference cases and explicit `pytorch_parity_mlp_long_loop_*_train_graph` tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Add deterministic long-loop compiled-MLP regression above raw IR training** - `d8e497b` (test)
2. **Task 2: Extend the PyTorch oracle for the same compiled-MLP long-loop case** - `a727bcf` (test)

## Files Created/Modified

- `.planning/phases/04-cpu-training-path-hardening-and-long-loop-stability/04-02-SUMMARY.md` - execution summary for this plan
- `src/engine/model/train_api.rs` - compiled-model MLP long-loop fixture and deterministic optimizer gates
- `tests/pytorch_parity.rs` - matching long-loop parity tests with explicit finite and drift checks
- `examples/pytorch_parity.py` - PyTorch reference generation for the same 24-epoch MLP long-loop case

## Decisions Made

- Used the existing tiny MLP dimensions and dataset values from parity as the canonical Phase 4 regression case, but increased loop depth to 24 epochs for a meaningful long-loop signal
- Kept parity scope intentionally narrow to the same MLP fixture so Phase 4 does not sprawl into Phase 5 model breadth

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Parallel `cargo test` processes produced a transient compiler-resolution failure in another test module, so final verification was run sequentially; the targeted verifies passed cleanly

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 4 now has both a Rust integration regression and a PyTorch oracle for the same long-loop compiled MLP case
- This summary feeds Phase 5 by documenting the narrow regression fixture that broader real-model parity should build from, not replace

---
*Phase: 04-cpu-training-path-hardening-and-long-loop-stability*
*Completed: 2026-03-08*
