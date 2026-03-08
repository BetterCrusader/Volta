---
phase: 06-product-surface-hardening-for-cli-doctor-examples-and-docs
plan: 02
subsystem: docs
tags: [readme, cli, roadmap, documentation, usage]

# Dependency graph
requires:
  - phase: 06-01
    provides: CLI smoke test scaffold that confirms commands are wired correctly
provides:
  - README.md with accurate Adam benchmark claim (+25% faster, not 1.9× slower)
  - README.md Roadmap snapshot reflecting Phases 1-5 done, Phase 6 in progress
  - README.md CLI table with export-py row
  - src/main.rs USAGE with llvm-codegen note on compile and correct extract description
  - docs/ROADMAP.md synced to current phase completion state
affects: [07-packaging, future-docs-readers, release-readiness]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - README.md
    - src/main.rs
    - docs/ROADMAP.md

key-decisions:
  - "README Adam claim changed from 1.9x slower to +25% faster, matching STATE.md metrics (4.259 ms vs 5.343 ms)"
  - "docs/ROADMAP.md fully rewritten to reflect executed phases 1-5 with accurate names and outcomes"
  - "USAGE extract description changed from <model_name> to <file.gguf|file.safetensors> to match actual CLI contract"

patterns-established:
  - "Public-facing claims must match STATE.md performance metrics exactly"

requirements-completed: [UX-V2-01]

# Metrics
duration: 8min
completed: 2026-03-08
---

# Phase 6 Plan 02: Documentation Accuracy Summary

**README Adam claim corrected to +25% faster, USAGE compile annotated with llvm-codegen requirement, docs/ROADMAP.md rewritten to reflect Phases 1-5 complete and Phase 6 in progress**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-08T09:43:00Z
- **Completed:** 2026-03-08T09:51:16Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Removed factually wrong "Adam 1.9× slower" claim from README Limitations and Benchmark highlights sections — replaced with "+25% faster at B≤64" which matches STATE.md metrics
- Updated README Roadmap snapshot from 5-phases-stale "Next: Adam fusion" to current Phases 1-5 done / Phase 6 in progress / Phase 7 planned structure
- Added `export-py` to README CLI table (it was already wired in code and USAGE string, just missing from table)
- Updated USAGE `compile` line to note `(requires --features llvm-codegen)` — prevents silent failures for users who try compile without the feature
- Updated USAGE `extract` from `<model_name>` to `<file.gguf|file.safetensors>  (GGUF/SafeTensors → .vt)` — matches actual CLI contract
- Rewrote docs/ROADMAP.md to accurately reflect executed phase history and current status

## Task Commits

1. **Task 1: Fix README stale claims** - `63c9dc9` (docs)
2. **Task 2: Fix USAGE string and sync docs/ROADMAP.md** - `4675d9e` (docs)

## Files Created/Modified

- `README.md` - Adam claim corrected, benchmark honest note fixed, roadmap snapshot updated, export-py added to CLI table
- `src/main.rs` - USAGE compile line annotated with llvm-codegen requirement; extract description corrected to file path form
- `docs/ROADMAP.md` - Full rewrite: Phases 1-5 completed with accurate descriptions, Phase 6 in progress, Phase 7 planned, unmapped items listed

## Decisions Made

- Adam benchmark number used: 4.259 ms vs PyTorch 5.343 ms = +25% faster (from STATE.md Performance Metrics)
- docs/ROADMAP.md rewritten rather than patched — the old content referenced wrong internal phase names and stale "next steps" throughout; a full rewrite was cleaner and less error-prone

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

Initial `cargo test` appeared to show a compile error (`missing fields in DoctorReport`) but this was a stale error message from a previous incremental build state — the linter had already added fields to the struct and the constructor. A clean `cargo build` succeeded immediately. All 66 tests passed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- README and public roadmap now accurately represent the project state
- USAGE string prevents user confusion on compile and extract commands
- 06-03 (examples/CLI fixes) can proceed independently — no blockers from this plan

---
*Phase: 06-product-surface-hardening-for-cli-doctor-examples-and-docs*
*Completed: 2026-03-08*
