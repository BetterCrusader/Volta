# Volta Roadmap v3: Execution Plan from Real Baseline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Date:** 2026-02-21  
**Scope:** Next 2 quarters with an infinite continuation model  
**Project Mode:** Open-source, free forever, no ads, no paid lock-in

---

## 0. Baseline (What already exists)

Volta is not a blank project. Current baseline in `main` already includes:

1. Language frontend and runtime path:
- `src/lexer.rs`
- `src/parser.rs`
- `src/semantic.rs`
- `src/executor.rs`

2. Compiler-first IR stack:
- `src/ir/graph.rs`, `src/ir/verifier.rs`, `src/ir/shape_inference.rs`
- `src/ir/scheduler.rs`, `src/ir/allocation.rs`, `src/ir/execution_plan.rs`
- `src/ir/autograd.rs`, `src/ir/train.rs`

3. Interop foundation:
- Stable interop contract in `src/interop/contract.rs`
- Real ONNX protobuf import in `src/interop/onnx.rs` (Wave 1 ops)

4. CUDA foundation:
- CUDA kernels and dispatch scaffold in `src/ir/cuda/**`
- Parity/determinism/memory guard tests under `tests/cuda_*`

5. Governance and release gates:
- `.github/workflows/ci.yml`, `pr-gates.yml`, `release-gates.yml`
- release checklist in `docs/release/v1-release-checklist.md`

This roadmap is for scaling a working core into an elite platform.

---

## 1. North Star

### Product promise
"The fastest way to write, train, inspect, and reproduce ML models with deterministic guarantees."

### Strategic model
- 70% Core Fortress (compiler/runtime/interop correctness)
- 20% Community Velocity (DX/docs/examples)
- 10% Moonshots (isolated, feature-flagged breakthroughs)

### Non-negotiables
1. Determinism before ergonomics.
2. No silent fallbacks.
3. Contract compatibility is explicit and versioned.
4. Every shipped feature has tests and rollback story.

---

## 2. What we will NOT do now (YAGNI guardrails)

1. No full framework parity claims versus PyTorch/TensorFlow.
2. No broad operator explosion without test-backed demand.
3. No distributed training in stable path this quarter.
4. No production serving platform before interop wave 2 + CLI maturity.

---

## 3. Q1 Plan (Weeks 1-12): Hardening + Interop Wave 2

## Phase A (Weeks 1-3): Language + Runtime Hardening

### Goals
- Close correctness gaps in existing DSL features.
- Upgrade diagnostics to human-quality actionable errors.
- Freeze behavior with regression tests.

### Tasks
1. Feature audit matrix (parse, semantic, execute) for every DSL construct in README and examples.
2. Diagnostic quality pass:
- unknown keyword suggestions
- line/column + code context
- "how to fix" hints
3. Deterministic behavior checks for:
- train defaults
- shuffle behavior
- save/load consistency

### Files to modify
- `src/parser.rs`
- `src/semantic.rs`
- `src/executor.rs`
- `src/autopilot.rs`
- `tests/*` (new focused suites)

### New tests
- `tests/language_feature_matrix.rs`
- `tests/error_messages_quality.rs`
- `tests/determinism_replay_contract.rs`

### Verification commands
```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --test language_feature_matrix -- --nocapture
cargo test --test error_messages_quality -- --nocapture
cargo test --test determinism_replay_contract -- --nocapture
```

### Definition of Done
1. No parser/semantic/runtime crash on malformed scripts.
2. Errors include line context + actionable suggestion.
3. Determinism tests green on repeated runs.

---

## Phase B (Weeks 4-8): ONNX Interop Wave 2

### Goals
- Extend ONNX importer beyond Wave 1.
- Support common graph transforms used by modern model pipelines.

### Wave 2 operator target
- `Reshape`
- `Concat`
- `Gather`
- `Slice`

### Tasks
1. Extend contract enum and validation rules.
2. Map ONNX protobuf nodes to new contract ops.
3. Enforce static-shape constraints and explicit unsupported paths.
4. Add end-to-end parser tests for each op and mixed graphs.

### Files to modify
- `src/interop/contract.rs`
- `src/interop/onnx.rs`
- `src/interop/mod.rs`

### New tests
- `tests/interop_onnx_wave2_contract.rs`
- `tests/interop_onnx_wave2_parser.rs`
- `tests/interop_onnx_wave2_e2e.rs`

### Verification commands
```bash
cargo test --features onnx-import --test interop_onnx_parser
cargo test --features onnx-import --test interop_onnx_wave2_contract
cargo test --features onnx-import --test interop_onnx_wave2_parser
cargo test --features onnx-import --test interop_onnx_wave2_e2e
bash scripts/ci/interop_onnx_verify.sh
```

### Definition of Done
1. Wave 2 ops imported and executed in deterministic path.
2. Unsupported ONNX cases fail loudly with precise reason.
3. Existing Wave 1 tests remain green.

---

## Phase C (Weeks 9-12): Shape/Broadcast and Perf Guard Hardening

### Goals
- Prevent semantic drift when graph complexity increases.
- Move perf governance from "monitor" to "enforce".

### Tasks
1. Formalize broadcast semantics and reject ambiguous cases.
2. Add regression tests for shape edge cases.
3. Upgrade perf gate to hard-fail above strict threshold with incident marker.

### Files to modify
- `src/ir/shape_inference.rs`
- `src/ir/verifier.rs`
- `scripts/perf/perf_gate.py`
- `docs/governance/perf-slo.md`

### New tests
- `tests/shape_broadcast_contract.rs`
- `tests/shape_error_regressions.rs`
- `scripts/perf/tests/test_perf_gate_incident_policy.py`

### Verification commands
```bash
cargo test --test shape_broadcast_contract -- --nocapture
cargo test --test shape_error_regressions -- --nocapture
pytest scripts/perf/tests/test_perf_gate_incident_policy.py -q
bash scripts/ci/perf_gate.sh
```

### Definition of Done
1. Shape mismatch bugs fail fast and deterministic.
2. Perf regression above policy threshold blocks merge/release.

---

## 4. Q2 Plan (Weeks 13-24): Model Primitives + DX + Community

## Phase D (Weeks 13-17): Transformer-usable Primitive Set

### Goals
- Add missing primitives to move from toy models to practical NLP/Vision baselines.

### Target additions
- Embedding
- LayerNorm
- GELU
- Attention primitive (CPU reference first, CUDA gated)

### Files to modify
- `src/model/layers.rs`
- `src/model/builder.rs`
- `src/ir/op.rs`
- `src/ir/lowering.rs`
- `src/ir/interpreter.rs`
- `src/ir/cuda/lowering.rs`

### New tests
- `tests/transformer_primitives_layers.rs`
- `tests/attention_cpu_reference.rs`
- `tests/attention_cuda_parity.rs`

### Definition of Done
1. Primitive set composes into a stable mini-transformer workflow.
2. CPU path is deterministic; CUDA path parity is tested where supported.

---

## Phase E (Weeks 18-21): Minimalist CLI and Onboarding

### Goals
- New contributor reaches first successful training run in under 15 minutes.

### CLI target
- `volta init`
- `volta check`
- `volta run`
- `volta info`

### Files to modify
- `src/main.rs`
- `scripts/ci/cli_smoke.sh`
- `scripts/ci/cli_smoke.ps1`
- `README.md`

### New docs
- `CONTRIBUTING.md`
- `docs/community/first-pr-guide.md`
- `examples/text-classifier/*`

### New tests
- `tests/cli_quickstart.rs`
- `tests/docs_onboarding_sync.rs`

### Definition of Done
1. Clean install -> init -> run validated in CI.
2. Onboarding docs produce first successful run without tribal knowledge.

---

## Phase F (Weeks 22-24): Release OS and Cross-Platform Reliability

### Goals
- Make release process reproducible on Linux and Windows.

### Tasks
1. Add PowerShell parity for release cut script.
2. Align release docs with both shell and PowerShell flows.
3. Add release playbook sync tests.

### Files to modify
- `scripts/release/cut_v1.sh`
- `scripts/release/cut_v1.ps1` (new)
- `docs/release/v1-release-checklist.md`
- `.github/workflows/release-gates.yml`

### New tests
- `tests/release_playbook_sync.rs`

### Definition of Done
1. Release can be cut locally on Windows without WSL dependency.
2. Rollback verify path documented and tested.

---

## 5. KPI and Governance Scorecard

Track every 2 weeks and publish to `docs/governance/scorecards/YYYY-MM-DD.md`.

### Core KPIs
1. Determinism regression count: `0`
2. Critical runtime crashes in CI: `0`
3. Perf regressions above hard SLO unresolved: `0`

### DX KPIs
1. Time-to-first-success median: `<= 15 min`
2. CLI smoke success rate: `>= 99%`
3. Docs-driven setup success (new contributor trial): `>= 85%`

### Community KPIs
1. External contributors/month: increasing trend
2. First PR merge median time: decreasing trend
3. Open issues with reproduction labels: `100%` coverage

---

## 6. Risk Register

1. Risk: expanding ops breaks determinism.
- Mitigation: feature flags + parity tests + rollback checks.

2. Risk: DX work dilutes core rigor.
- Mitigation: 70/20/10 capacity cap and mandatory core gates.

3. Risk: roadmap scope creep.
- Mitigation: every feature requires explicit "non-goals" and DoD.

4. Risk: release fragility on Windows.
- Mitigation: first-class PowerShell scripts and release sync tests.

---

## 7. Merge/Release Gate (must stay green)

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
bash scripts/ci/interop_onnx_verify.sh
bash scripts/ci/xl_verify.sh
bash scripts/release/rollback.sh --verify-only
```

Windows note: use PowerShell equivalents for shell scripts where needed.

---

## 8. Infinite Continuation Protocol

At cycle end:
1. Write retrospective with shipped/not-shipped/root-causes.
2. Copy this roadmap to next cycle file and prune completed tasks.
3. Keep 70/20/10 unless incident report justifies temporary override.
4. Never advance to next phase with unresolved deterministic regressions.

---

## 9. Immediate next 5 actions (this week)

1. Create `tests/language_feature_matrix.rs` and `tests/error_messages_quality.rs`.
2. Draft ONNX Wave 2 contract changes in `src/interop/contract.rs`.
3. Add importer test skeleton `tests/interop_onnx_wave2_parser.rs`.
4. Add `scripts/release/cut_v1.ps1` with parity behavior.
5. Add contributor onboarding docs and link them from README.

