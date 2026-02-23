# Docs Contract Audit (1.2.0)

Status: In Progress
Related issue: #37

## Purpose

Track claim-by-claim verification so documentation never overstates implementation behavior.

## Audit Method

1. Enumerate explicit claims from README and governance docs.
2. Map each claim to executable evidence (tests, scripts, CI lanes).
3. Mark claims as `verified`, `pending`, or `mismatch`.
4. Fix docs or implementation when mismatch is found.

## Claim Ledger

| Claim Source | Claim | Evidence | Status | Notes |
| --- | --- | --- | --- | --- |
| `README.md` | Core verification commands run successfully | `docs/governance/evidence-2026-02-23-local-verification.md` | verified | Includes fmt/clippy/tests/onnx-import/fuzz check |
| `README.md` | Wave1 local verify command path is current | `scripts/ci/wave1_local_verify.sh` + local run evidence | verified | stale `scripts.ci.tests` references removed |
| `README.md` | ONNX feature tests are part of verification flow | `.github/workflows/pr-gates.yml` (`test-onnx-import`) | verified | landed in PR gate workflow |
| `docs/ONNX_COVERAGE.md` | ONNX operator status matrix matches runtime behavior | `cargo test --features onnx-import --test interop_onnx_wave2_parser`, `bash scripts/ci/interop_onnx_verify.sh`, `cargo test --features onnx-import` | verified | LeakyRelu coverage added and verified in current local cycle |
| `docs/governance/cuda-training-determinism.md` | strict mode has no silent fallback | `tests/cuda_train_no_fallback.rs` + verify scripts | pending | validate with latest release-candidate run before close |
| `docs/governance/ci-topology.md` | PR/release/nightly topology reflects active workflows | workflow files + governance docs tests | verified | governance docs content tests pass |

## Open Actions

1. Complete ONNX coverage claim verification pass before RC.
2. Complete strict CUDA fallback claim verification pass before RC.
3. Add links to final CI run URLs during release cut.

## Exit Criteria

- No `mismatch` rows remain.
- No `pending` rows remain for release-blocking claims.
- Issue #37 has links to final evidence artifacts.
