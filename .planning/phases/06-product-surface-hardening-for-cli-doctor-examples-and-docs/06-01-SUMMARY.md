---
phase: 06-product-surface-hardening-for-cli-doctor-examples-and-docs
plan: 01
subsystem: testing
tags: [rust, cargo-test, integration-test, cli-smoke, CARGO_BIN_EXE]

requires:
  - phase: 05-end-to-end-parity
    provides: working CLI binary with doctor, compile-train, run, check commands

provides:
  - tests/cli_smoke.rs with 6 named integration test functions
  - smoke_version, smoke_doctor, doctor_json_fields, smoke_run_xor,
    smoke_check_bench_real, smoke_compile_train_rejects_non_mlp
  - Wave 0 prerequisite unblocking all Wave 1 Phase 6 plans

affects: [06-02, 06-03]

tech-stack:
  added: []
  patterns:
    - CARGO_BIN_EXE_volta for binary path in integration tests
    - current_dir(CARGO_MANIFEST_DIR) for resolving example paths
    - temp .vt file via std::env::temp_dir() for rejection path testing

key-files:
  created:
    - tests/cli_smoke.rs
  modified: []

key-decisions:
  - "All six tests use CARGO_BIN_EXE_volta instead of cargo run to avoid rebuild overhead"
  - "smoke_compile_train_rejects_non_mlp creates a temp .vt using 'use fn' model syntax to trigger the MLP-only rejection in compile_first_model_to_train_dll; all example .vt files are pure MLPs so a synthetic file is required"
  - "doctor_json_fields uses substring matching on 'tool', 'healthy', 'backends' rather than a JSON parser — no serde_json dep needed in test binary"

patterns-established:
  - "Integration tests live in tests/*.rs and use CARGO_BIN_EXE_<binary> pattern"
  - "Each smoke test is fully independent with no shared state"

requirements-completed: [UX-V2-02, UX-V2-03]

duration: 8min
completed: 2026-03-08
---

# Phase 6 Plan 01: CLI Smoke Test Scaffold Summary

**6-function Rust integration test scaffold using CARGO_BIN_EXE_volta, covering version, doctor (text+JSON), run, check, and compile-train rejection — Wave 0 prerequisite for Phase 6**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-08T01:43:07Z
- **Completed:** 2026-03-08T01:51:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `tests/cli_smoke.rs` with all 6 required test function stubs
- All 6 tests pass in 0.72s; full 61-test suite still green (no regressions)
- Established CARGO_BIN_EXE_volta + CARGO_MANIFEST_DIR pattern for all future CLI integration tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Create tests/cli_smoke.rs with all required test stubs** - `ae05458` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `tests/cli_smoke.rs` - 6-function CLI integration smoke test scaffold: smoke_version, smoke_doctor, doctor_json_fields, smoke_run_xor, smoke_check_bench_real, smoke_compile_train_rejects_non_mlp

## Decisions Made
- Used CARGO_BIN_EXE_volta (not `cargo run`) for zero-rebuild binary access
- `doctor_json_fields` checks JSON key presence via `contains()` — no serde_json needed in integration test binary
- All example `.vt` files are pure MLPs, so `smoke_compile_train_rejects_non_mlp` creates a temp `.vt` file using `use <fn>` model syntax to set `use_fn` in the executor and trigger the "MLP-only today" rejection in `compile_first_model_to_train_dll`
- Temp file uses 1-epoch, 2-3-1 model to keep execution time under 100ms

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. The only non-obvious detail: all `examples/*.vt` files are pure MLPs (no `use fn`), requiring a synthetic temp file for the non-MLP rejection test. This was anticipated in the plan's `<action>` section.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `tests/cli_smoke.rs` exists and all 6 stubs are green — Wave 1 plans (06-02, 06-03) can add `<automated>` verify commands that reference this file
- Doctor JSON fields are confirmed present; 06-02 can tighten those assertions when it rewrites the doctor command
- compile-train rejection is confirmed working; 06-03 smoke for examples is confirmed working

---
*Phase: 06-product-surface-hardening-for-cli-doctor-examples-and-docs*
*Completed: 2026-03-08*
