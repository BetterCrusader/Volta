---
phase: 03-nadiiinist-i-korektnost
plan: "03-01"
subsystem: engine/ir
tags: [panic-safety, autograd, tensor, correctness]
dependency_graph:
  requires: []
  provides: [RELY-01, RELY-02]
  affects: [tensor.rs, autograd.rs]
tech_stack:
  added: []
  patterns: [let-else, ok_or_else-AutogradError, make_contiguous-bounds-check]
key_files:
  created: []
  modified:
    - src/engine/ir/tensor.rs
    - src/engine/ir/autograd.rs
    - src/engine/ir/fingerprint.rs
    - src/engine/ir/printer.rs
    - src/engine/ir/shape_inference.rs
    - src/engine/ir/verifier.rs
    - src/engine/ir/interpreter.rs
    - src/engine/ir/lowering.rs
decisions:
  - "make_contiguous bounds-check before copy_to_slice: returns Err instead of panicking on invalid offset"
  - "MultiHeadAttentionBackward match arms: no-op in autograd (backward of backward not needed), Err in interpreter (unsupported)"
metrics:
  duration: "~7 min"
  completed: "2026-03-07"
  tasks_completed: 2
  files_modified: 8
---

# Phase 3 Plan 1: Fix Tensor::PartialEq and build_reverse_graph Panic Sites Summary

**One-liner:** Eliminated 4 unwrap() panics in tensor.rs and autograd.rs using let-else and ok_or_else, plus added exhaustive match arms for Op::MultiHeadAttentionBackward (pre-existing compile blocker).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Fix Tensor::PartialEq panic + test | d387089 | tensor.rs + 6 other files |
| 2 | Fix build_reverse_graph 3 unwraps + test | 62ace88 | autograd.rs |

## Changes Made

### Task 1: Tensor::PartialEq panic-free
- Replaced `self.make_contiguous().unwrap()` and `other.make_contiguous().unwrap()` with `let Ok(...) else { return false; }` in `PartialEq::eq`
- Added bounds validation in `make_contiguous()`: checks offset + len <= data.len() for contiguous path, and checks max index per dimension for non-contiguous path, returning `TensorError` instead of panicking
- Added `tensor_eq_invalid_offset` test: tensor with offset=100, data.len()=4 returns false on both `invalid.eq(&valid)` and `valid.eq(&invalid)`

### Task 2: build_reverse_graph unwraps
- Line 86 `Op::Plugin` arm: `.copied().unwrap()` → `.copied().ok_or_else(|| AutogradError { message: format!(...) })?`
- Line 1698 `Op::QuantizeLinear` arm: `*grad_map.get(...).unwrap()` → `.copied().ok_or_else(...)?`
- Line 1704 `Op::DequantizeLinear` arm: same pattern
- Added `Op::MultiHeadAttentionBackward` arm in autograd backward match (no-op — backward of backward not needed)
- Added `build_reverse_graph_shared_output` test: `Add(a, a)` (shared output node) returns `Ok`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocker] Pre-existing compile failures: Op::MultiHeadAttentionBackward missing from 6 match sites**
- **Found during:** Task 1 (first cargo test run)
- **Issue:** `Op::MultiHeadAttentionBackward` was added to op.rs but match arms were not added to fingerprint.rs, printer.rs (format_op), shape_inference.rs, verifier.rs, interpreter.rs, lowering.rs — preventing compilation entirely
- **Fix:** Added exhaustive arms to all 6 sites with appropriate behavior (hash all fields, format as "mha_bwd ...", shape inference via q_input shape, ValueType::Tensor, interpreter Err, lowering no-op)
- **Files modified:** fingerprint.rs, printer.rs, shape_inference.rs, verifier.rs, interpreter.rs, lowering.rs
- **Commits:** d387089

**2. [Rule 1 - Bug] make_contiguous() panics instead of returning Err on invalid offset**
- **Found during:** Task 1 TDD RED phase (test panicked, not failed)
- **Issue:** `copy_to_slice` uses `self.data[data_idx]` without bounds check; `make_contiguous` called `copy_to_slice` after `is_contiguous()` returned false, causing index-out-of-bounds panic rather than returning `Err`
- **Fix:** Added pre-copy bounds validation in `make_contiguous`: checks offset+len bounds for contiguous path, max index per dimension for non-contiguous path — returns `TensorError` on violation
- **Files modified:** tensor.rs
- **Commits:** d387089

## Verification

- `tensor_eq_invalid_offset` — PASS
- `build_reverse_graph_shared_output` — PASS
- Full `cargo test -p volta` — 229 passed, 0 failed

## Self-Check: PASSED
