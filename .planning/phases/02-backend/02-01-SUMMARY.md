---
phase: 02-backend
plan: 01
subsystem: engine/ir
tags: [fingerprint, hashing, codegen, mkl, stability]
dependency_graph:
  requires: []
  provides: [stable-graph-fingerprints, mkl-error-reporting]
  affects: [src/engine/ir/fingerprint.rs, src/engine/ir/backend.rs, src/engine/ir/codegen/mlp_train_rust_codegen.rs]
tech_stack:
  added: []
  patterns: [SipHasher13 fixed-seed hashing, Result-based error propagation, testable-inner-function pattern]
key_files:
  created: []
  modified:
    - src/engine/ir/fingerprint.rs
    - src/engine/ir/backend.rs
    - src/engine/ir/codegen/mlp_train_rust_codegen.rs
decisions:
  - "SipHasher13::new_with_keys(0, 0) chosen over SipHasher13::new() — fixed seed required for cross-build stability"
  - "resolve_mkl_lib_path_from() extracted as testable inner function — avoids unsafe env var mutation in deny(unsafe_code) project"
metrics:
  duration: "~6 minutes"
  completed: "2026-03-07"
  tasks: 2
  files: 3
requirements: [INFRA-03, INFRA-01]
---

# Phase 02 Plan 01: Stable Graph Fingerprints and MKL Error Reporting Summary

SipHasher13 with fixed seed replaces DefaultHasher for cross-build stable fingerprints; resolve_mkl_lib_path returns Result with actionable error instead of silently linking a hardcoded dev path.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Replace DefaultHasher with SipHasher13 in fingerprint.rs and backend.rs | 56a4361 | fingerprint.rs, backend.rs |
| 2 | Change resolve_mkl_lib_path to return Result and remove hardcoded fallback | a048d8b | mlp_train_rust_codegen.rs |

## Decisions Made

1. **SipHasher13 fixed seed** — `new_with_keys(0, 0)` not `new()`. The random OS seed in `new()` defeats the purpose of cross-build stability. Pattern already used in scheduler.rs — consistent.

2. **Testable inner function** — Extracted `resolve_mkl_lib_path_from(mkl_lib_dir, mklroot, conda_prefix)` to avoid mutating env vars in tests. Project has `#![deny(unsafe_code)]` which conflicts with `unsafe { set_var(...) }`. Clean solution: pass explicit values to the inner function in tests, outer function reads env vars and delegates.

3. **MKL_LIB_DIR bad path returns specific error** — If `MKL_LIB_DIR` is set but mkl_rt.lib is not there, the error names the exact path. Previously, the code would silently fall through to the hardcoded fallback. Now it fails fast and names what was wrong.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Extracted testable inner function instead of using env var mutation**
- **Found during:** Task 2 — tests needed to call `resolve_mkl_lib_path` without mutating env vars
- **Issue:** Project has `#![deny(unsafe_code)]`; `std::env::set_var`/`remove_var` require `unsafe` in this Rust version; tests that mutate env vars also have race conditions in parallel test runs
- **Fix:** Extracted `resolve_mkl_lib_path_from(Option<&str>, Option<&str>, Option<&str>)` as the logic-bearing function; outer `resolve_mkl_lib_path()` reads env vars and delegates. Tests call inner function directly.
- **Files modified:** src/engine/ir/codegen/mlp_train_rust_codegen.rs
- **Commit:** a048d8b

## Verification Results

```
cargo test --lib fingerprint -q  → 3 passed (fingerprint_is_deterministic, fingerprint_different_graphs_differ, backend test)
cargo test --lib mkl -q          → 2 passed (mkl_not_found_returns_err_with_instructions, mkl_lib_dir_bad_path_returns_specific_err)
cargo test --lib                 → 221 passed; 0 failed
```

## Success Criteria Check

- [x] `cargo test --lib fingerprint -q` — ok, all pass
- [x] `cargo test --lib mkl -q` — ok, new tests pass
- [x] No `DefaultHasher` left in fingerprint.rs or backend.rs
- [x] No `C:/Users/User/` string literal left in mlp_train_rust_codegen.rs
- [x] `resolve_mkl_lib_path` signature is `-> Result<String, RustTrainCodegenError>`
