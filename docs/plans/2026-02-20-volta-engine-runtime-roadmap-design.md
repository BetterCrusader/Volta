# Volta Engine Runtime Roadmap Design

## Status

Approved in-session for planning baseline.

## Goal

Build a real ML engine core where DSL is an interface layer, and runtime execution is the product center.

Primary milestone for this roadmap:

- Tiny-transformer `train + infer + save + load + infer-after-reload` on CPU
- Deterministic and reproducible runs
- No PyTorch runtime dependency

This is the tiny-transformer CPU milestone contract for runtime integrity.

## Scope and Constraints

### In Scope (Phase 1)

- Stable CPU execution path for tiny-transformer E2E
- Autograd correctness and optimizer correctness on the same runtime path
- Model format v1 with deterministic checkpoint round-trip
- Memory planner stability and peak-memory visibility
- Blocking verification gates for determinism, correctness, and checkpoint parity

### Out of Scope (Phase 1)

- CUDA as a blocking dependency for initial milestone
- Distributed runtime
- Full production LLVM/CUDA backend stack
- Broad operator catalog beyond tiny-transformer needs

CUDA enters Phase 2 after CPU milestone lock.

## Recommended Approach

Adopt **Runtime-first** execution:

1. Lock functional correctness and determinism on CPU
2. Lock model format v1 and checkpoint parity
3. Stabilize verification and reliability gates
4. Expand to CUDA inference MVP

Why this approach:

- Minimizes risk of perf-first dead ends
- Preserves ability to reason about correctness before optimization
- Gives a concrete, demo-ready milestone with real product signal

## Architecture

Critical path:

`DSL/Frontend -> Stable IR -> Passes -> Execution Plan -> CPU Kernel Runtime -> Autograd/Optimizer -> Checkpoint Format`

System rule:

- Runtime is the source of truth for train/infer behavior
- Frontend must not bypass verifier or runtime contracts

## Component Design

### 1. Tensor and Kernel Core (CPU)

Required operator set for tiny-transformer path:

- `matmul`, `add`, `norm`, `softmax`, activation ops, embedding ops, loss ops

Requirements:

- Explicit shape checks
- Deterministic math policy where promised by governance docs
- Stable error surfaces for invalid graph/shape paths

### 2. Autograd Contract

- Backward graph is built separately
- Forward graph is immutable during backward construction
- Gradient path is validated with deterministic repeat tests

### 3. Memory Planner

- Stable buffer planning and reuse model
- Peak-memory estimation/reporting for each execution plan
- Regression checks for planner stability on repeated runs

### 4. Runtime Execution Loop

- Single runtime path for train and infer
- Optimizers (SGD/Adam) remain runtime-level components
- No hidden eager fallback shortcuts

### 5. Model Format v1

- Deterministic serialization for weights + metadata + format version
- Backward-compatible loader behavior for v1 evolution policy
- Round-trip checks: save -> load -> infer parity

### 6. Verification Layer

- Golden tests for tiny-transformer functional path
- Determinism and replay tests
- Snapshot/regression checks for IR+plan contracts

## Data Flow

`parse -> IR verify -> optimize -> plan -> execute -> checkpoint -> reload -> infer parity check`

Each stage exposes explicit failure diagnostics and remains test-addressable.

## Quality Gates (Blocking)

1. Tiny-transformer train loss decreases on tiny dataset
2. Inference is stable before and after reload
3. Checkpoint round-trip preserves expected numeric behavior
4. Determinism suite passes repeated runs with fixed seeds
5. Memory planner stays under declared peak budget for reference configs

## Timeline (1 Full-Time Developer)

### Phase A (4-6 weeks)

- CPU correctness completion for tiny-transformer operator set
- Model format v1 completion
- End-to-end train/infer/save/load milestone

### Phase B (3-4 weeks)

- Stability hardening (fuzz/soak/determinism reinforcement)
- Perf baseline governance and regression observability

### Phase C (4-6 weeks, post-milestone)

- CUDA inference MVP on top of locked CPU contracts

## Risks and Mitigations

### Risk 1: Scope Creep

- Mitigation: hard milestone boundary (CPU tiny-transformer E2E only)

### Risk 2: Numeric Drift and Flakiness

- Mitigation: tolerance policy, fixed seeds, deterministic replay, gate-backed checks

### Risk 3: Premature Optimization

- Mitigation: correctness gates must pass before perf optimization work can block roadmap

## Exit Criteria for This Design

- Approved architecture and milestone boundary are explicit
- Blocking quality gates are defined and measurable
- Phase sequence and ownership assumptions are clear
- Ready to convert into executable implementation plan

## Next Step

Generate implementation plan from this design using the planning workflow.
