# ONNX Operator Coverage Matrix

This document tracks the current ONNX importer coverage in Volta and reflects what is implemented in parser + contract + runtime paths.

## Supported ONNX Ops (Wave 2 static subset)

| ONNX Operator | Volta IR | Notes |
| :--- | :--- | :--- |
| `Add` / `Sub` / `Mul` / `Div` | `Op::Add` / `Op::Sub` / `Op::Mul` / `Op::Div` | Broadcasting supported. |
| `MatMul` | `Op::MatMul` | 2D tensor path. |
| `Gemm` | `Op::Gemm` + optional `Op::Transpose` lowers | `alpha`, `beta`, `transA`, `transB`, optional bias. |
| `Relu` | `Op::Relu` | - |
| `Softmax` | `Op::Softmax` | Numerically stable CPU implementation. |
| `Log` / `Exp` / `Sigmoid` | `Op::Log` / `Op::Exp` / `Op::Sigmoid` | - |
| `Gelu` | `Op::Gelu` / `Op::GeluExact` | `approximate=tanh` -> `Gelu`; `approximate=none` -> `GeluExact`. |
| `ReduceSum` | `Op::ReduceSum` | `keepdims=0|1`, single-axis static path, negative-axis normalization when rank is known. |
| `ReduceMax` | `Op::ReduceMax` | Same reduction constraints as above. |
| `ReduceMean` | `Op::ReduceMean` | Same reduction constraints as above. |
| `Transpose` / `Reshape` | `Op::Transpose` / `Op::Reshape` | Static reshape only. |
| `Concat` / `Gather` / `Slice` | `Op::Concat` / `Op::Gather` / `Op::Slice` | Static indices/ranges required. |

## Autograd status notes

- CPU autograd covers core math and reduction paths used in training tests.
- `ReduceMax` backward is still explicit `not implemented` (fails loudly by design).
- `GeluExact` backward is explicit `not implemented` (fails loudly by design).
- `GemmBackward` is intentionally lowered/decomposed into primitive ops (for example `MatMul`), not executed as a standalone runtime kernel.

## CUDA status (honest matrix)

| Operator family | CPU Inference | CPU Train | CUDA Inference | CUDA Train |
| :--- | :---: | :---: | :---: | :---: |
| Add/Sub/Mul/Div | ✅ | ✅ | ✅ | ✅ (covered in train flows) |
| MatMul | ✅ | ✅ | ✅ | ✅ (covered in train flows) |
| Relu/Softmax | ✅ | ✅ | ✅ | ✅ (covered in train flows) |
| ReduceSum/ReduceMax/ReduceMean | ✅ | ✅ | ✅ | partial |
| Conv2D | ✅ | ✅ | ❌ (stub) | ❌ (stub) |
| Sigmoid | ✅ | ✅ | ✅ | ✅ |
| Gelu / GeluExact | ✅ | ✅ | ✅ | ✅ |
| Gemm | ✅ | ✅ | ✅ | ✅ |

`partial` means class is available for selected training paths but not full operator-family completeness.

`stub` means the operator is known to the IR/interop path, but CUDA execution currently fails fast with explicit `not implemented`.

## Known limits

| Area | Limit |
| :--- | :--- |
| Shapes | Static, rank-known wave for advanced axis normalization. |
| DTypes | Primarily `f32` tensors; integer support limited to index/shape utilities. |
| ONNX breadth | Pooling, LayerNorm, Dropout and many training-centric ops are not yet imported. |
