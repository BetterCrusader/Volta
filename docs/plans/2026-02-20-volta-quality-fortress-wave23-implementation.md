# Volta Quality Fortress Wave 2/3 Implementation Plan

## Goal

Complete release-depth and nightly-quality hardening after Wave 1 governance baseline.

## Wave 2 (Release Depth)

- Add release-gates workflow with:
  - Wave 1 gate replay
  - fuzz smoke
  - short soak
  - double-pass perf gate (median delta threshold)
  - rollback script verification
- Add perf governance scripts:
  - CPU signature detector
  - perf probe binary
  - baseline compare renderer
  - perf gate checker

## Wave 3 (Nightly Maturity)

- Add nightly-quality workflow with:
  - heavy fuzz
  - long soak
  - perf matrix and report artifacts
  - automated nightly regression issue updates

## Local Verification

- `bash scripts/ci/wave23_local_verify.sh`
- `python -m unittest scripts.perf.tests.test_cpu_signature scripts.perf.tests.test_perf_gate scripts.perf.tests.test_baseline_compare -v`
- `bash scripts/ci/release_perf_double_pass.sh`
