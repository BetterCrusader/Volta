---
phase: 06-product-surface-hardening-for-cli-doctor-examples-and-docs
plan: 03
subsystem: cli
tags: [doctor, mkl, llvm, diagnostics, capability-matrix, env-vars]

requires:
  - phase: 06-01
    provides: CLI smoke test scaffold (binary integration tests)

provides:
  - check_mkl_from() injectable MKL detection helper with unit tests
  - check_mkl_available() reads real env vars and verifies file existence
  - check_llvm_available() checks LLVM_SYS_210_PREFIX then clang in PATH
  - DoctorReport with mkl_lib_path, llvm_info, sgd_backend_env, llvm_prefix_env fields
  - print_doctor text output with 5 structured sections
  - print_doctor JSON output with mkl_available and llvm_available fields

affects: [06-04, cli-ux, packaging]

tech-stack:
  added: []
  patterns:
    - Injectable env detection: check_mkl_from() accepts injected values so unit tests bypass real env
    - Actionable warnings: MKL-not-found adds a warning with the fix command, not just a flag
    - Structured doctor sections: Environment, Capability Matrix, AOT Codegen, Environment Variables, Next Steps

key-files:
  created: []
  modified:
    - src/main.rs
    - tests/cli_smoke.rs

key-decisions:
  - "MKL detection duplicated inline in main.rs (not calling private mlp_train_rust_codegen fn) per plan constraint"
  - "check_mkl_from() uses injected env values to allow unit testing without real MKL install"
  - "MKL-not-found adds actionable warning to warnings Vec (makes healthy:false when MKL absent)"
  - "smoke_doctor test updated to check for 5 section headers instead of bare Volta doctor string"

requirements-completed: [UX-V2-02]

duration: 12min
completed: 2026-03-08
---

# Phase 6 Plan 03: Doctor Rewrite Summary

**`volta doctor` rewritten with MKL/LLVM detection, 5-section capability matrix output, and env-var diagnostics including actionable next steps**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-08T09:40:00Z
- **Completed:** 2026-03-08T09:52:46Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `check_mkl_from()` (injectable), `check_mkl_available()`, `check_llvm_available()` to `src/main.rs`
- 5 unit tests for `check_mkl_from` covering None, file-missing, MKL_LIB_DIR hit, MKLROOT/lib hit, and priority ordering
- `DoctorReport` expanded with `mkl_lib_path`, `llvm_info`, `sgd_backend_env`, `llvm_prefix_env`
- `print_doctor` text output now has 5 labelled sections: Environment, Capability Matrix, AOT Codegen, Environment Variables, Next Steps
- `print_doctor` JSON output now includes `mkl_available` (bool) and `llvm_available` (bool) fields
- `cargo test --quiet` and `cargo test --quiet --test cli_smoke doctor` both pass (313 tests total)

## Task Commits

1. **Task 1: MKL and LLVM detection helpers** - `959dddd` (feat)
2. **Task 2: Expand DoctorReport and rewrite print_doctor** - `4675d9e` + `1a8e47d` (feat — main.rs changes landed in prior execution context; test file update committed in 1a8e47d)

## Files Created/Modified

- `src/main.rs` - Added 3 helper functions (check_mkl_from, check_mkl_available, check_llvm_available), expanded DoctorReport struct, rewrote print_doctor text and JSON output, added 5 unit tests
- `tests/cli_smoke.rs` - Updated smoke_doctor to assert 5 section headers; updated doctor_json_fields to also assert mkl_available and llvm_available

## Decisions Made

- MKL detection duplicated inline (not calling the private `resolve_mkl_lib_path_from` in mlp_train_rust_codegen.rs) — plan explicitly forbade calling that private fn; duplicate is ~10 lines
- MKL-not-found adds a warning to `warnings` vec, making `healthy: false` when MKL is absent — this is the correct behaviour since Adam/AdamW --rust will not link without MKL
- `smoke_doctor` test string updated from `"Volta doctor"` to checking all 5 section names — old string was a stale assertion against the old flat-dump format

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated smoke_doctor test string to match new output format**
- **Found during:** Task 2 (rewrite print_doctor output)
- **Issue:** `smoke_doctor` asserted `stdout.contains("Volta doctor")` but new header is `"--- Volta Doctor ---"` (capital D); test would have failed
- **Fix:** Updated test to assert presence of all 5 section headers ("Volta Doctor", "Capability Matrix", "AOT Codegen", "Environment Variables", "Next Steps")
- **Files modified:** tests/cli_smoke.rs
- **Verification:** `cargo test --quiet --test cli_smoke doctor` passes
- **Committed in:** 1a8e47d (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in test matching new intended output)
**Impact on plan:** Required fix — without it cli_smoke doctor would fail. No scope creep.

## Issues Encountered

None — compilation clean on first attempt, all tests green.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `volta doctor` now gives actionable output for users lacking MKL or LLVM
- JSON output suitable for scripted environment checks
- UX-V2-02 satisfied
- Phase 6 Wave 1 complete (06-01 smoke scaffold, 06-02 usage/docs, 06-03 doctor rewrite)

---
*Phase: 06-product-surface-hardening-for-cli-doctor-examples-and-docs*
*Completed: 2026-03-08*
