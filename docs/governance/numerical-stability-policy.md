# Numerical Stability Policy

> **Status**: Active  
> **Version**: 1.0  
> **Last Updated**: 2026-02-22

## Overview

This document defines the numerical stability guarantees and edge-case behavior for all mathematical operations in Volta. These policies ensure deterministic, reproducible results across CPU and GPU backends.

## Core Principles

1. **NaN Propagation**: NaN input produces NaN output (IEEE 754 compliant)
2. **Inf Propagation**: Infinity propagates through operations unless explicitly collapsed (e.g., `sum(Inf) = Inf`)
3. **No Silent Clamping**: We do not silently clamp NaN/Inf unless explicitly required by the operation's mathematical definition
4. **Determinism**: Same input → Same output, bit-exact where possible

---

## Unary Operations

### exp(x) — Exponential

**Policy**: Input is clamped to a safe range to prevent overflow.

- If `x > 88.0`: Return `Inf` (since `exp(88) ≈ 1e38` which is near f32 max)
- If `x < -126.0`: Return `0.0` (since `exp(-126) ≈ 1e-55` which is near f32 min normalized)
- Otherwise: Return `exp(x)` as computed

**Rationale**: f32 overflow produces Inf, but we document this boundary explicitly.

**Tests Required**:
- `exp(0)` → `1.0`
- `exp(1000)` → `Inf`
- `exp(-1000)` → `0.0`
- `exp(NaN)` → `NaN`
- `exp(Inf)` → `Inf`
- `exp(-Inf)` → `0.0`

---

### log(x) — Natural Logarithm

**Policy**: IEEE 754 compliant NaN for non-positive inputs.

- If `x > 0`: Return `ln(x)`
- If `x <= 0`: Return `NaN`
- If `x == NaN`: Return `NaN`

**Rationale**: Matches IEEE 754 and PyTorch/NumPy behavior.

**Tests Required**:
- `log(1)` → `0.0`
- `log(0)` → `-Inf`
- `log(-1)` → `NaN`
- `log(NaN)` → `NaN`

---

### sigmoid(x)

**Policy**: Numerically stable split formula.

We use the split formula to avoid overflow:
```
sigmoid(x) = x >= 0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x))
```

- If `x >= 0`: Use standard formula
- If `x < 0`: Use `exp(x) / (1 + exp(x))` to avoid `exp(-x)` overflow

**Tests Required**:
- `sigmoid(0)` → `0.5`
- `sigmoid(10)` → ~0.99995
- `sigmoid(-10)` → ~0.000045
- `sigmoid(Inf)` → `1.0`
- `sigmoid(-Inf)` → `0.0`
- `sigmoid(NaN)` → `NaN`

---

### gelu(x) — Gaussian Error Linear Unit

**Policy**: Use tanh approximation (BERT/GPT style).

```
gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

This is the ONNX `approximate="tanh"` variant.

**Note**: Exact erf-based gelu (`0.5 * x * (1 + erf(x/√2))`) is NOT currently supported.

**Tests Required**:
- `gelu(0)` → `0.0`
- `gelu(1)` → ~0.84119
- `gelu(-1)` → ~-0.15830
- `gelu(NaN)` → `NaN`
- `gelu(Inf)` → `Inf`

---

## Binary Operations

### add(a, b), sub(a, b), mul(a, b), div(a, b)

**Policy**: Follow IEEE 754 for all operations.

- `Inf + finite` → `Inf`
- `Inf + (-Inf)` → `NaN`
- `NaN + anything` → `NaN`
- Division by zero: Return `Inf` or `-Inf` with sign based on numerator

**Broadcasting**: When broadcasting, NaN/Inf propagates correctly through the broadcast operation.

---

### matmul(a, b) — Matrix Multiplication

**Policy**:
- If any element is NaN → output NaN
- If any element is Inf → output may be Inf (depends on values)
- No special clamping

**Tests Required**:
- Standard matmul tests
- Matmul with Inf in one input → Inf in output
- Matmul with NaN in one input → NaN in output

---

## Reduction Operations

### softmax(x, axis)

**Policy**: Numerically stable max-trick is MANDATORY.

```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

- Subtract max before exp to prevent overflow
- If all inputs are `-Inf`: Each output is `1/N` (where N is the dimension size)

**Tests Required**:
- `softmax([1, 2, 3])` → sums to 1.0
- `softmax([1000, 2000, 3000])` → finite (no overflow)
- `softmax([-1000, -2000])` → finite (no underflow)
- `softmax([NaN, 1, 2])` → `[NaN, ...]`
- `softmax([Inf, 1, 2])` → `[1.0, 0.0, 0.0]`

---

### reduce_sum(x, axis)

**Policy**:
- `sum([Inf, ...])` → `Inf`
- `sum([NaN, ...])` → `NaN`
- Empty reduction → `0.0`

---

## Gemm Operations

### gemm(a, b, alpha, beta, bias)

**Policy**: Standard BLAS semantics.

```
output = alpha * (a @ b) + beta * bias
```

- If `alpha` or `beta` is NaN → output NaN
- If `alpha` or `beta` is Inf → output may be Inf
- `alpha = 1.0` → skip multiplication (identity)
- `beta = 1.0` → skip multiplication (identity)
- `bias = None` → no bias addition

---

## Gradient (Autograd) Behavior

### Gradient Numerical Stability

Gradients should follow the same stability principles as forward operations:

- `sigmoid_backward`: Computes `grad * sigmoid(x) * (1 - sigmoid(x))`
- `gelu_backward`: Computes derivative analytically; handles edge cases
- Unbroadcast (gradient reduction): Uses `reduce_sum` which follows reduction policy

---

## Testing Requirements

Every mathematical operation MUST have tests for:

| Input | Expected Output |
|-------|-----------------|
| Normal values | Correct mathematical result |
| `0` | Correct result |
| `-0` | Correct result (may differ from `+0` in division) |
| `Inf` | Per-operation policy |
| `-Inf` | Per-operation policy |
| `NaN` | NaN propagation |
| `f32::MAX` | Per-operation policy |
| `f32::MIN` | Per-operation policy |
| Edge combinations | e.g., `exp(Inf) + exp(-Inf)` |

---

## CPU vs GPU Equivalence

For operations that run on both CPU and GPU:

- **Tolerance**: `1e-5` relative error for f32
- **NaN/Inf**: Must match exactly (not approximate)
- **Note**: Some GPU implementations may differ; document in operation-specific docs

---

## Policy Exceptions

Any deviation from this policy requires:
1. Documented rationale in the operation's docstring
2. Explicit test documenting the deviation
3. RFC or design document for significant changes

---

## Related Documents

- [Determinism Scope](../governance/determinism-scope.md)
- [CUDA Determinism Policy](../governance/cuda-determinism-policy.md)
