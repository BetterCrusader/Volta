# Backend Capability Matrix

## Backend Capability Matrix

This matrix is the source of truth for runtime capability negotiation.

| Backend | Inference | Training | strict determinism | Default determinism |
| --- | --- | --- | --- | --- |
| CPU | yes | yes | yes | strict |
| CUDA | yes | yes | yes | balanced |
| LLVM | no | no | no | balanced |

## Enforcement Rules

1. Strict mode must fail fast when a backend does not support strict determinism.
2. Runtime gateway remains the only execution path for both infer and train.
3. No silent CPU fallback is allowed in strict mode.
4. Core IR contracts remain backend-neutral.
