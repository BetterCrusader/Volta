---
phase: 07-packaging-and-install-story-for-distributable-releases
plan: 02
subsystem: infra
tags: [github-actions, ci, smoke-test, release, artifacts]

# Dependency graph
requires:
  - phase: 07-01
    provides: build job artifacts (linux tar.gz, macos-arm64 raw binary, windows zip)
provides:
  - smoke job in release.yml validating built artifacts on fresh OS runners
affects: [07-03]

# Tech tracking
tech-stack:
  added: []
  patterns: [post-build smoke matrix on 3 OS runners, artifact download + unpack + exec verify]

key-files:
  created: []
  modified: [.github/workflows/release.yml]

key-decisions:
  - "smoke needs: build (not release) — avoids circular dependency since release job creates universal binary from build artifacts"
  - "macOS smoke uses volta-macos-arm64 raw binary, not universal — macos-latest is Apple Silicon so aarch64 runs natively"
  - "volta doctor called without --strict — exits 0 even when MKL absent on GHA runners"
  - "actions/checkout runs before artifact download so examples/xor.vt is available for run smoke"
  - "fail-fast: false — all 3 OS entries run independently"
  - "Windows unpack uses shell: pwsh for Expand-Archive; smoke run steps use ./dist/ prefix (works on all 3 OS runners)"

patterns-established:
  - "Post-build smoke: checkout repo first, then download artifact separately into dist/"

requirements-completed: [DIST-V2-02, DIST-V2-03]

# Metrics
duration: 2min
completed: 2026-03-08
---

# Phase 7 Plan 02: Smoke Check Release Artifact Summary

**3-OS smoke job added to release.yml: downloads per-OS build artifact, unpacks binary, asserts --help/doctor/run exit 0 on fresh ubuntu-latest/macos-latest/windows-latest runners**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-08T14:58:59Z
- **Completed:** 2026-03-08T15:00:43Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Added `smoke` job to `.github/workflows/release.yml` running after the `build` job
- 3-entry matrix (ubuntu-latest, macos-latest, windows-latest) with per-OS artifact names, binary names, and unpack strategies
- Each OS runner: checkout repo, download artifact, unpack, run `volta --help`, `volta doctor`, `volta run examples/xor.vt`
- macOS uses raw aarch64 binary (not universal) to avoid circular dependency with release job

## Task Commits

1. **Task 1: Add smoke check job to release.yml** - `2845d26` (feat)

## Files Created/Modified

- `.github/workflows/release.yml` - smoke job appended (59 lines added, zero existing lines changed)

## Decisions Made

- `needs: build` not `needs: release`: the release job creates the universal macOS binary via lipo; if smoke needed release, it would create a circular or sequential dependency. Using build artifacts directly is simpler and faster.
- macOS smoke uses `volta-macos-arm64` raw binary: `macos-latest` is Apple Silicon (aarch64). The aarch64 binary from the build matrix runs natively. The universal binary only exists after the release job runs lipo.
- `volta doctor` without `--strict`: MKL is not installed on any GHA OS runner. Without `--strict`, doctor exits 0 and reports warnings. With `--strict` it would exit non-zero, which would be a false failure.
- `actions/checkout` before artifact download: the downloaded artifact contains only the binary. `examples/xor.vt` lives in the repo, so checkout must run first.
- Windows unpack via `shell: pwsh` + `Expand-Archive`: PowerShell cmdlet, not available in bash. Smoke run steps use `./dist/` prefix which works on all 3 runners.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `python3 -m yaml` not available initially; installed pyyaml to run verification. No impact on output.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Smoke job is live in release.yml and will run on next tag push
- 07-03 (README install section + download URLs) can now proceed — both release artifacts and smoke verification are in place
- No blockers

---
*Phase: 07-packaging-and-install-story-for-distributable-releases*
*Completed: 2026-03-08*
