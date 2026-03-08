---
phase: 07-packaging-and-install-story-for-distributable-releases
plan: "03"
subsystem: docs
tags: [readme, install, packaging, releases, platform-support]

# Dependency graph
requires:
  - phase: 07-01
    provides: artifact names (linux-x86_64.tar.gz, macos-universal.tar.gz, windows-x86_64.zip) and CI pipeline
provides:
  - README.md Install section with download table and unpack steps for Linux/macOS/Windows
  - Updated Platform note section reflecting CI-tested Linux and macOS
  - Removal of stale "build from source only" and "not tested" claims
affects: [public-facing docs, onboarding, release checklist]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "README Install section references exact artifact filenames from release.yml"

key-files:
  created: []
  modified:
    - README.md

key-decisions:
  - "Install section placed before Quickstart so first-time users see binary install path first"
  - "cargo install NOT listed as primary path per user decision"
  - "Platform note renamed from 'Install / platform note' to 'Platform note' — Install is now a standalone section"
  - "MKL bullet added to Platform note since compile-train --rust requires it"

patterns-established:
  - "Public README limitations must match STATE.md facts — no aspirational claims"

requirements-completed: [DIST-V2-02]

# Metrics
duration: 2min
completed: 2026-03-08
---

# Phase 7 Plan 03: README Install Section Summary

**README Install section added with artifact table, unpack steps for all 3 platforms, verify block, and Platform note updated to reflect CI-tested Linux/macOS**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-08T14:59:00Z
- **Completed:** 2026-03-08T15:00:46Z
- **Tasks:** 1 completed
- **Files modified:** 1

## Accomplishments

- Added `## Install` section before `## Quickstart` with download table linking to releases/latest
- Artifact table includes all 3 filenames matching Plan 07-01: `linux-x86_64.tar.gz`, `macos-universal.tar.gz`, `windows-x86_64.zip`
- Unpack + PATH steps for Linux/macOS (`tar xzf` + `chmod` + `mv`) and Windows (unzip)
- Verify block with `volta --help`, `volta doctor`, `volta run examples/xor.vt`
- "No Rust toolchain required" note for prebuilt binary users
- Replaced `## Install / platform note` with `## Platform note` — Linux and macOS now say "tested in CI"
- Added MKL bullet to Platform note
- Removed stale "Not tested on macOS or Linux" and "build from source only" bullets

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Install section and update platform note in README.md** - `fcab362` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `README.md` - Added Install section (49 lines net), replaced Install/platform note heading, updated Limitations bullet

## Decisions Made

- Install section is "first thing users see" — placed before Quickstart per user decision recorded in plan context
- `cargo install` not listed as primary path per user decision
- MKL bullet added to Platform note since `compile-train --rust` requires it and `volta doctor` surfaces status

## Deviations from Plan

None - plan executed exactly as written. One minor fix: accidentally introduced a duplicate Limitations line during the first edit attempt, immediately corrected before commit.

## Issues Encountered

None — all verification checks passed on first run after corrections.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 7 complete: release pipeline (07-01), smoke-test artifact (07-02), and README install story (07-03) all done.
The public-facing README now accurately reflects the binary release story established in 07-01.

---
*Phase: 07-packaging-and-install-story-for-distributable-releases*
*Completed: 2026-03-08*
