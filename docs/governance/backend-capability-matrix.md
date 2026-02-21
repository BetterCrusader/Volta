# Backend Capability Matrix

The fastest way to ship regressions is to guess backend behavior.
This matrix removes guesswork.

## Backend Capability Matrix

This matrix is the source of truth for runtime capability negotiation.

| Backend | Inference | Training | strict determinism | Default determinism |
| --- | --- | --- | --- | --- |
| CPU | yes | yes | yes | strict |
| CUDA | yes | yes | yes | balanced |
| LLVM | no | no | no | balanced |

## Why This Matters

When capabilities are explicit, runtime behavior stays predictable across CI, local runs, and releases.

## Enforcement Rules

1. Strict mode must fail fast when a backend does not support strict determinism.
2. Runtime gateway remains the only execution path for both infer and train.
3. No silent CPU fallback is allowed in strict mode.
4. Core IR contracts remain backend-neutral.
