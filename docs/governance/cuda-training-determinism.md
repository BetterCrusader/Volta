# CUDA Training Determinism Policy

Training determinism is where compiler promises become product promises.

## Strict Mode Guarantees

Strict mode is a fail-fast contract for CUDA training execution.

1. Fixed reduction topology is required for every gradient accumulation step.
2. Workspace allocation order is deterministic and replay-stable.
3. optimizer state update ordering is deterministic for SGD and Adam.
4. There is no silent CPU fallback in strict mode.

## Replay Expectations

- Same graph + same parameters + same dataset order + same determinism level must reproduce bitwise-equal outputs.
- Any unsupported deterministic behavior must return an explicit runtime failure.

## Non-Goals

- Strict mode does not promise fastest throughput.
- Fast mode may relax ordering for performance, but must remain explicit policy.
