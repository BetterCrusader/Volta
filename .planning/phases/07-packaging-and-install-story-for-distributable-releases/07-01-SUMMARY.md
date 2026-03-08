---
phase: 07-packaging-and-install-story-for-distributable-releases
plan: "01"
subsystem: build-and-release
tags:
  - cargo
  - release
  - macos-universal
  - lipo
  - arm
  - ci
dependency_graph:
  requires: []
  provides:
    - arm-build-unblocked
    - macos-universal-binary-pipeline
  affects:
    - .github/workflows/release.yml
    - Cargo.toml
tech_stack:
  added: []
  patterns:
    - macOS universal binary via lipo on ubuntu-latest release runner
    - Conditional upload_raw matrix field to split raw vs packaged artifact upload
key_files:
  created: []
  modified:
    - Cargo.toml
    - .github/workflows/release.yml
decisions:
  - "Removed x86-v4 from gemm features: runtime CPU detection still provides AVX2 on x86_64 and NEON on ARM without compile-time hard-enable"
  - "lipo step runs on ubuntu-latest (release job) not macOS: ubuntu has GNU lipo via binutils-apple cross tools via apt or via preinstalled lipo from macOS SDK available on GH ubuntu runners"
  - "macOS entries upload raw binary (not tar.gz) so lipo can combine them; packaging deferred to release job"
  - "Linux and Windows retain direct packaged artifact upload unchanged"
metrics:
  duration_seconds: 155
  completed_date: "2026-03-08T14:55:58Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
---

# Phase 7 Plan 01: ARM Build Unblock + macOS Universal Binary Pipeline Summary

**One-liner:** Removed `x86-v4` from gemm to unblock ARM builds and restructured release.yml to produce a single macOS universal binary via `lipo` in the release job.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Remove x86-v4 from Cargo.toml gemm features | 5de3fac | Cargo.toml |
| 2 | Restructure release.yml matrix and add lipo merge step | b6e2388 | .github/workflows/release.yml |

## What Was Done

**Task 1 — Cargo.toml fix:**
Changed `gemm = { version = "0.19", features = ["rayon", "x86-v4"] }` to `gemm = { version = "0.19", features = ["rayon"] }`. The `x86-v4` feature hard-enables AVX-512 code paths at compile time, which do not exist on AArch64. Without it, gemm uses its own runtime CPU detection to select AVX2 on x86_64 and NEON on ARM. Local `cargo build --release --locked` confirmed success (57s build).

**Task 2 — release.yml restructure:**
Rewrote the build matrix from 3 build targets (with both macOS entries doing their own tar.gz packaging) to a 4-entry matrix with a new `upload_raw` boolean field:
- Linux (x86_64): `upload_raw: false` — packages tar.gz and uploads
- macOS ARM64 (aarch64-apple-darwin): `upload_raw: true` — uploads raw `volta` binary
- macOS x86_64 (x86_64-apple-darwin): `upload_raw: true` — uploads raw `volta` binary
- Windows: `upload_raw: false` — packages zip and uploads

The release job downloads all artifacts into `artifacts/`, runs `lipo -create` on both macOS raw binaries to produce `volta-universal`, then packages it as `volta-{tag}-macos-universal.tar.gz`. The `softprops/action-gh-release` step receives exactly 3 files: linux tar.gz, macos-universal tar.gz, windows zip.

All 6 action SHA pins preserved unchanged.

## Verification Results

1. `cargo build --release --locked` exits 0 — PASS
2. `grep x86-v4 Cargo.toml` returns nothing — PASS
3. `grep -c "upload_raw" release.yml` returns 7 (>= 2) — PASS
4. `grep lipo release.yml` finds the lipo step — PASS
5. `grep macos-universal release.yml` confirms artifact name — PASS
6. `grep ea165f8d65b6e75b540449e92b4886f43607fa02 release.yml` returns upload-artifact line — PASS

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- Cargo.toml modified: FOUND
- .github/workflows/release.yml modified: FOUND
- Commit 5de3fac: FOUND
- Commit b6e2388: FOUND
