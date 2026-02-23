# CUDA Training Determinism Policy

Training determinism is where compiler promises become product promises.

## Strict Mode Guarantees

Strict mode is a fail-fast contract for CUDA training execution.

1. Fixed reduction topology is required for every gradient accumulation step.
2. Workspace allocation order is deterministic and replay-stable.
3. optimizer state update ordering is deterministic for SGD and Adam.
4. There is no silent CPU fallback in strict mode.

## Wave Support Map (training path)

- Supported and exercised in strict training flows:
  - Add/Sub/Mul/Div
  - MatMul
  - Relu
  - Softmax
  - Sigmoid / SigmoidBackward
  - Gelu / GeluExact / GeluBackward
  - Gemm
  - ReduceSum / ReduceMean
- Supported with explicit limits:
  - ReduceMax forward path exists, but `ReduceMax` backward is not implemented and must fail fast.
  - `GemmBackward` is expected to be decomposed into primitive ops during autograd; standalone CUDA kernel execution for `GemmBackward` is not a supported contract.
- Explicitly unsupported classes must fail fast with actionable error messages.

The support map is enforced by regression tests, including no-silent-fallback checks.

## Replay Expectations

- Same graph + same parameters + same dataset order + same determinism level must reproduce bitwise-equal outputs.
- Any unsupported deterministic behavior must return an explicit runtime failure.

## Non-Goals

- Strict mode does not promise fastest throughput.
- Fast mode may relax ordering for performance, but must remain explicit policy.

## Failure Mode Policy

- Unsupported CUDA kernel classes must return explicit runtime errors.
- Error messages should identify the unsupported class or backward-construction blocker.
- Silent fallback to CPU execution is prohibited in strict mode.
