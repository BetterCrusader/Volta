# Volta CUDA Inference MVP Design

## Status

Post-CPU-milestone handoff draft.

## Goal

Deliver CUDA inference MVP by reusing locked CPU runtime contracts and proving CPU/CUDA parity before any aggressive optimization.

## Scope

- Inference only (no CUDA training in this phase)
- Reuse existing verified IR and execution plan contracts
- Keep checkpoint format and model loading behavior identical to CPU path

## Contract Before Implementation

- [ ] CPU parity harness defined
- [ ] Kernel boundary API frozen
- [ ] Inference-only operator subset selected

## Architectural Rules

1. No second frontend path: CUDA executes the same graph contracts as CPU.
2. No persistence fork: checkpoint loader/writer remains format-compatible.
3. Correctness before performance: parity and determinism gates are blocking.

## Phases

### Phase 1: Parity Harness

- Add CPU vs CUDA output comparison harness over fixed fixture set.
- Gate on `max_abs_diff <= epsilon` per operator/model fixture.

### Phase 2: Kernel Bring-Up

- Implement minimal kernel subset for tiny-transformer inference path.
- Keep unsupported ops explicit and fail fast with clear diagnostics.

### Phase 3: Stability

- Add replay checks across repeated CUDA inference runs.
- Add regression baselines for latency and memory footprint.

## Exit Criteria

- Tiny-transformer inference passes CPU/CUDA parity checks.
- No contract drift in IR, checkpoint format, or execution scheduling.
- Deterministic replay gates pass for fixed seeds and fixed inputs.
