# Tier A Contracts

This document is authoritative for Tier A behavior (`src/ir`, `src/device`, tensor core, matmul tuning, KV internals).

## Tier A Invariants

- Tensor shape contracts are validated before execution.
- Type contracts are strict; mixed numeric types are rejected unless explicitly converted.
- Verifier rules (SSA, use-before-def, single producer) are hard requirements.
- Runtime fallback from unavailable accelerators must be explicit and deterministic.
- No panics in runtime paths; failures return structured `TensorError` messages.

## Numeric Tolerance Table

| Path | Metric | Tolerance |
| --- | --- | --- |
| `softmax` | row sum | `abs(sum - 1.0) <= 1e-5` |
| `cross_entropy` | finite result | `is_finite == true` |
| AVX2 matmul vs scalar | max absolute diff | `<= 1e-4` |
| deterministic replay checks | fingerprint equality | exact string match |

## Device Fallback Semantics

- `--device cpu`: hard-pin to CPU; creation failure is fatal.
- `--device cuda`: hard-pin to CUDA; if unavailable, fail fast with explicit error.
- `--device auto`: detect preferred device and fallback to CPU when supported.
- Fallback behavior must never silently change numerics outside documented tolerances.
