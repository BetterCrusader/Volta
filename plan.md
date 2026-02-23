# Volta 1.2.0 Plan

Status: Draft
Target release: `release-v1.2.0`
Planning window: 8 weeks

## Release Thesis

Volta 1.2.0 extends capability without diluting deterministic-first guarantees.

Primary objective:

- expand supported ONNX and CUDA paths in a controlled, test-backed way
- preserve strict no-silent-fallback behavior
- keep docs and governance claims exactly aligned with executable behavior

## Principles

1. Determinism is a product contract, not best effort.
2. Unsupported behavior must fail explicitly and informatively.
3. Governance and CI rules are release blockers, not suggestions.
4. Docs are part of the contract and must be continuously synchronized.
5. Every feature requires tests, policy impact review, and CI integration.

## Current Baseline (1.1.0)

- Strong deterministic IR and scheduler hardening already in place.
- ONNX support is Wave 2 static subset with explicit limits.
- CUDA support is partial but policy-backed and fail-fast for unsupported paths.
- Core gates (`fmt`, `clippy`, `test`, policy lanes) are active.

## In Scope

- ONNX Wave 2.5 operator expansion for deterministic-safe static paths.
- CUDA strict-lane parity expansion for supported operator families.
- Autograd correctness broadening for high-value ops and graph patterns.
- Numerical stability enforcement expansion (NaN/Inf/extreme cases).
- CI/release pipeline tightening and anti-flake hardening.
- Documentation and governance synchronization as a hard gate.

## Out of Scope

- Full ONNX ecosystem coverage.
- Dynamic-shape control flow import.
- Silent fallback paths in strict mode.
- Broad syntax/API redesign.
- Throughput-only optimizations that weaken deterministic guarantees.

## Success Criteria (KPIs)

1. All mandatory quality gates pass on release-candidate commits.
2. `cargo test --features onnx-import` is green and blocking in CI.
3. At least 4 new ONNX operator paths land with full test + docs coverage.
4. No known silent fallback behavior remains in strict CUDA lanes.
5. Determinism regressions pass repeated-run and threaded stability checks.
6. Perf SLO gates stay within policy or have explicit approved baseline updates.
7. README/governance/coverage docs contain no claim drift.
8. Nightly reliability is stable or all blockers are triaged with ownership.

## Workstreams

## A) Deterministic Core

Objective: eliminate hidden nondeterministic iteration and ordering risks.

Deliverables:

- pass internals deterministic iteration audit
- first-divergence diagnostics for pass pipeline and schedule hash changes
- expanded deterministic regression suite (multi-run and multi-thread)

Acceptance:

- deterministic suite stable in debug and release profiles
- no flaky ordering regressions under repeated CI reruns

## B) ONNX Wave 2.5 Expansion

Objective: increase practical import coverage within strict static contracts.

Candidate operators (subject to contract review):

- `LeakyRelu`
- `BatchNorm` (inference scope)
- `MaxPool` / `AveragePool` (static subset)
- `LayerNorm` (if static contract remains robust)

Deliverables:

- parser + contract + lowering + runtime path per accepted operator
- explicit unsupported-attribute diagnostics
- fixture-based tests and coverage matrix updates

Acceptance:

- every added operator is tested and documented as-shipped
- unsupported combinations fail loudly and deterministically

## C) CUDA Strict-Lane Hardening

Objective: improve strict-mode reliability and parity confidence.

Deliverables:

- stronger parity tests for supported kernels
- replay and no-fallback regressions for training and inference
- deterministic allocation/workspace verification improvements

Acceptance:

- `cuda_infer_verify`, `cuda_train_verify`, `xl_verify` consistently pass
- no strict-mode silent fallback paths

## D) Autograd Correctness

Objective: broaden backward correctness and deterministic optimizer behavior.

Deliverables:

- gradcheck expansion for broadcast/reduction/new op paths
- integration tests for forward->backward->optimizer loops
- explicit handling for still-unsupported backward paths

Acceptance:

- autograd suites stay green and deterministic
- no ambiguous partial-support behavior without docs/tests

## E) Numerical Stability Program

Objective: turn numerical policy into enforced tests.

Deliverables:

- NaN/Inf/extreme-value matrix by operator family
- CPU/CUDA parity behavior checks with explicit tolerances
- documented policy exceptions with test evidence

Acceptance:

- policy-linked tests pass and remain non-flaky
- NaN/Inf behavior matches documented contracts

## F) Performance and Cache Governance

Objective: preserve predictable performance with explainable baseline changes.

Deliverables:

- stable perf-gate outputs and artifact clarity
- cache hit/miss diagnostics for repeated workloads
- release double-pass perf report quality improvements

Acceptance:

- perf regressions are visible, attributable, and actionable
- baseline updates are explicit and traceable

## G) CI and Release Pipeline Tightening

Objective: maximize CI signal quality and reduce false failures.

Deliverables:

- onnx-import lane in blocking PR gates
- stale workflow reference cleanup
- weekly CI health review process (flake ownership and SLA)

Acceptance:

- one-week stability window with no unexplained blocker lane failures

## H) Docs and Governance Sync

Objective: keep claims truthful and synchronized.

Deliverables:

- synchronized updates across `README.md`, governance docs, and coverage docs
- release checklist for 1.2.0
- contributor guidance for safe operator additions

Acceptance:

- docs-contract audit passes before RC

## Timeline

### Week 1

- lock scope and owners
- publish milestone and P0/P1/P2 issue breakdown
- finalize operator intake rules

### Week 2

- CI upgrades (blocking onnx-import lane, stale reference cleanup)
- deterministic harness improvements

### Week 3

- ONNX expansion batch 1
- tests + docs for batch 1

### Week 4

- ONNX expansion batch 2
- interop parity hardening

### Week 5

- CUDA strict replay/no-fallback hardening sprint

### Week 6

- autograd + numerical stability expansion sprint

### Week 7

- bug bash, soak, docs freeze, changelog prep

### Week 8

- RC cut
- full release gates, perf double-pass, rollback verification
- release tag and notes

## P0 Backlog (Release Blockers)

1. CI gate for `cargo test --features onnx-import`.
2. Deterministic audit + 100-run stability verification.
3. Four ONNX operator paths end-to-end (code + tests + docs).
4. Strict CUDA no-silent-fallback proof set.
5. Numerical stability test matrix enforcement.
6. Release perf double-pass green status.
7. Docs-contract audit with zero mismatches.

## Verification Commands

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo test --features onnx-import
cargo check --manifest-path fuzz/Cargo.toml
bash scripts/ci/wave1_local_verify.sh
bash scripts/ci/cuda_infer_verify.sh
bash scripts/ci/cuda_train_verify.sh
bash scripts/ci/xl_verify.sh
bash scripts/ci/interop_onnx_verify.sh
bash scripts/ci/release_perf_double_pass.sh
```

## Definition of Done

Release 1.2.0 is complete when all of the following are true:

1. versioning and release metadata are consistent
2. all P0 items are closed with passing evidence
3. PR/release blocking gates are green without bypass
4. docs and coverage matrices match tested behavior exactly
5. no Tier-A correctness blockers remain open
6. rollback verification passes
