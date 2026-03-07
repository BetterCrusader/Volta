---
phase: 02-backend
plan: 03
subsystem: backend
tags: [rust, optimizer, adam, backend-capabilities, validation]

# Dependency graph
requires:
  - phase: 02-backend/02-01
    provides: SipHasher13 fingerprints, CpuBackend/CudaBackend compile() implementations
provides:
  - BackendCapabilities.supports_adam bool field
  - BackendCapabilities.validate_optimizer() method
  - Early optimizer/backend rejection in train_graph_with_backend
affects: [02-backend, 03-reliability]

# Tech tracking
tech-stack:
  added: []
  patterns: [capability-check-before-compile, optimizer-validation-at-runtime-boundary]

key-files:
  created: []
  modified:
    - src/engine/ir/backend_capabilities.rs
    - src/engine/ir/backend.rs
    - src/engine/ir/train.rs

key-decisions:
  - "CpuBackend supports_adam: true — MKL cblas_sgemm + AVX2 Adam is implemented"
  - "CudaBackend supports_adam: false — CUDA Adam path not implemented in v1"
  - "validate_optimizer uses string matching (lowercase) not enum pattern — avoids coupling to OptimizerConfig in backend_capabilities.rs"
  - "Call site in train.rs uses enum match to produce string — clean separation between optimizer enum and capability check"

patterns-established:
  - "Capability check before compilation: validate() then validate_optimizer() then build_execution_plan"
  - "Optimizer name as string at capability boundary — decoupled from optimizer.rs enum"

requirements-completed: [CORR-02]

# Metrics
duration: 2min
completed: 2026-03-07
---

# Phase 2 Plan 3: Adam Optimizer Capability Tracking Summary

**BackendCapabilities gains supports_adam field and validate_optimizer() method; train.rs rejects Adam on unsupported backends before compilation with a clear error message**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-07T06:45:50Z
- **Completed:** 2026-03-07T06:47:11Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments
- Added `supports_adam: bool` to `BackendCapabilities` struct (after `supports_gradient_updates`)
- Implemented `validate_optimizer()` method: returns Err naming the backend for adam/adamw when `supports_adam: false`
- CpuBackend returns `supports_adam: true`, CudaBackend returns `supports_adam: false`
- Wired `validate_optimizer()` call site in `train_graph_with_backend` immediately after existing `validate()` call
- 4 new unit tests covering all 4 matrix cells (sgd/adam x supported/unsupported)
- All 225 lib tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Add supports_adam field, validate_optimizer() method, and wire call site in train.rs** - `fed2c63` (feat)

**Plan metadata:** (docs commit — see below)

_Note: TDD — tests written first (RED), then implementation (GREEN). Single commit covers both since GREEN immediately followed RED within same task._

## Files Created/Modified
- `src/engine/ir/backend_capabilities.rs` - Added `supports_adam: bool` field, `validate_optimizer()` method, 4 new tests, updated `cpu_caps()` test helper
- `src/engine/ir/backend.rs` - Added `supports_adam: true` to CpuBackend, `supports_adam: false` to CudaBackend
- `src/engine/ir/train.rs` - Added `validate_optimizer` call site in `train_graph_with_backend`, updated `InferenceOnlyBackend` test struct literal

## Decisions Made
- `CpuBackend::supports_adam = true`: MKL + AVX2 Adam path is implemented and benchmarked (+25% vs PyTorch)
- `CudaBackend::supports_adam = false`: CUDA Adam not implemented in v1; explicit false prevents silent fallthrough
- String-based optimizer name at the capability boundary decouples `backend_capabilities.rs` from `optimizer.rs` — avoids cross-module enum dependency
- Match in `train.rs` maps `OptimizerConfig` enum variants to strings — single source of truth for the mapping, close to the call site

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CORR-02 requirement closed
- All existing 225 lib tests still pass
- Next: remaining Phase 2 plans (02-04 if any, or Phase 3)

---
*Phase: 02-backend*
*Completed: 2026-03-07*
