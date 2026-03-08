# Volta Architecture

## Overview

Volta is structured as three layers: a frontend language pipeline, a core IR engine, and a codegen backend that produces native DLLs.

CPU portability contract, at the source/runtime level:

- Tier 1: `x86_64`, `aarch64` / ARM64
- Everything else: best-effort

That is not the same thing as shipped release artifacts. Packaging coverage is narrower and is tracked separately in the release workflow/docs.

```
.vt source
    └─ Lexer (lexer.rs)
    └─ Parser (parser.rs)  →  AST (ast.rs)
    └─ SemanticAnalyzer (semantic.rs)
            │
            ├─── Executor (executor.rs)          ← interpreter path
            │
            └─── IR Builder
                    └─ Graph (graph.rs, node.rs, op.rs)
                    └─ Optimization passes
                    └─ Autograd (autograd.rs)
                    └─ Scheduler (scheduler.rs)
                    └─ Interpreter (interpreter.rs)
                    └─ Codegen
                            ├─ inner.rs           ← LLVM IR → inference .dll
                            ├─ mlp_train_codegen.rs   ← C source → training .dll
                            └─ mlp_train_rust_codegen.rs ← Rust crate → training .dll (fast path)
```

---

## Frontend (`src/frontend/`)

Processes `.vt` files in three stages:

1. **`lexer.rs`** — tokenizes source text into a flat token stream
2. **`parser.rs`** — produces an AST (`Program`, `Stmt`, `Expr`, `Property` defined in `ast.rs`)
3. **`semantic.rs`** — validates the AST: type checking, undefined variables, property validation. Emits errors with span info and "did you mean?" suggestions via `diagnostics.rs`.

The `.vt` language supports: `model` declarations with layer specs, `dataset` (CSV or synthetic), `train` blocks, `infer` blocks, `fn` declarations, `if`/`else`, `for` loops, variable assignment, `print`, `save`/`load`.

---

## Engine (`src/engine/`)

### `executor.rs`
The main `.vt` interpreter (~2400 lines). Walks the AST, maintains a scope-stack runtime (`Runtime` struct), and orchestrates training/inference. For `train` blocks, it either runs the interpreter path (through `runtime.rs`) or dispatches to codegen if `volta compile-train` is used.

### `autopilot.rs` + `rules.rs`
Rule-based hyperparameter resolution. Fills missing training config with a three-tier priority: Explicit > Rule > Default.

---

## IR (`src/engine/ir/`)

The core computation engine. A directed graph where each `Node` holds an `Op` and produces a `ValueId`. `BasicBlock`s group `Node`s; `Graph` holds blocks.

### Key types

- **`op.rs`** — all operations: arithmetic, MatMul, Conv2D, pooling, norms, activations, custom calls
- **`tensor.rs`** — `Tensor` type with stride-based views and `Arc<Vec<f32>>` shared storage
- **`graph.rs`** / **`node.rs`** / **`block.rs`** — graph construction

### Optimization passes (all implement `trait Pass`)

| Pass | File |
|---|---|
| Constant folding | `constant_folding.rs` |
| Dead code elimination | `dce.rs` |
| Common subexpression elimination | `cse.rs` |
| Algebraic simplification | `algebraic_simplification.rs` |
| Dead tensor elimination | `dead_tensor_elimination.rs` |
| Elementwise fusion | `elementwise_fusion.rs` |
| Gradient fusion | `gradient_fusion.rs` |

All passes are wrapped with a verifier guard (`pass_utils::run_with_verifier_guard`) that validates graph invariants before and after.

### Autograd (`autograd.rs`)
~1400 lines. Builds a separate backward graph from the forward graph without mutation. `GradientGraph` + `build_reverse_graph`. Produces gradient nodes for each differentiable op.

### Scheduler + Execution (`scheduler.rs`, `interpreter.rs`, `runtime.rs`)
- **`scheduler.rs`**: deterministic topological sort — no randomized ordering, no heuristics
- **`interpreter.rs`**: evaluates the graph node-by-node on CPU
- **`runtime.rs`**: ties backend + plan + interpreter together

### Memory planning
`allocation.rs`, `memory_planner.rs`, `static_memory_budget.rs` — static buffer allocation and reuse planning.

### Backend abstraction (`backend.rs`)
`Backend` trait with `CpuBackend` (always available) and `CudaBackend` (behind `cuda` feature). CPU backend executes fully sequentially for determinism.

Backends also expose explicit capability metadata through `backend_capabilities.rs`:

- backend kind
- device class (`Cpu` vs `Gpu`)
- vendor (`GenericCpu`, `Nvidia`, ...)
- maturity (`Validated` vs `Experimental`)
- phase coverage (`Inference`, `Training`)
- runtime execution support
- gradient-update support
- determinism coverage (`Strict`, `Balanced`, `Fast`)

`runtime.rs` validates requested execution mode against those capabilities before compiling or dispatching the plan. This keeps backend growth honest: adding a new backend now requires declaring what it can actually do instead of inheriting optimistic defaults.

### CUDA backend (`cuda/`, feature-gated)
Exists behind `--features cuda`. Device management, memory profiling, kernel lowering, determinism policy enforcement. Status: compiles, GPU performance has not been benchmarked.

---

## Codegen (`src/engine/ir/codegen/`)

This is the path that produces the performance numbers in the benchmarks.

Both `volta compile` and `volta compile-train` now take an explicit CPU mode:

- `--cpu-target portable` = default
- `--cpu-target native` = opt-in, host-specific tuning

Portable is the baseline contract. Native is for machine-specific binaries and benchmark chasing.

### Inference DLL (`inner.rs`)

`volta compile model.vt` → trains interpreter-side → extracts weights → generates LLVM IR → `.o` → `.dll`.

CPU target mode:

- `portable`: generic LLVM CPU target (`x86-64` on x86_64, `generic` on ARM64/other)
- `native`: LLVM `native`

Key decisions:
- Weights embedded as constant `[N x i8]` globals in LLVM IR
- Intermediate buffers: mutable zeroinit global `f32` arrays (not stack alloca — avoids stack limits)
- Optimization passes: `mem2reg, loop-vectorize, slp-vectorizer, loop-unroll, instcombine` (not `default<O3>` — IPO infers `readnone` on output param, killing output stores)
- Output copy uses volatile store loop to prevent dead store elimination
- Fused Add+ReLU: detects Add with single ReLU consumer, emits one combined loop

Exported symbol: `volta_infer(input_ptr, in_n, output_ptr, out_n)`

### Training DLL — C path (`mlp_train_codegen.rs`)

`volta compile-train model.vt` → generates C source → compiles with clang.

Scope:

- MLP-only
- SGD-only

CPU target mode:

- `portable`: `clang -O3 -ffast-math -funroll-loops`
- `native`: adds `-march=native`

- All buffers pre-allocated in `VoltaTrainHandle`, reused between steps
- GEMM via `volta_gemm_f32` from `gemm_shim.c` (tiled GEMM, GEMV fast path)
- Exports: `volta_train_init`, `volta_train_step`, `volta_train_loss`, `volta_train_free`, `volta_train_get/set_params`

### Training DLL — Rust path (`mlp_train_rust_codegen.rs`)

`volta compile-train model.vt --rust` → creates a temporary Cargo crate → generates `lib.rs` → `cargo build --release`.

Scope:

- MLP-only
- Optimizers supported today: `SGD`, `Adam`, `AdamW`, `Adagrad`

This is the path used for the published CPU benchmarks. Performance details:

**Forward pass**: `gemm` crate (Rayon-based, ~80% of MKL throughput)

**Backward pass optimizations**:
- `fast_transpose`: AVX2 8×8 kernel + scalar 32×32 fallback for non-multiples of 8
- Pre-transpose delta: `dt = delta^T`, then `tmp = W @ dt`, then `dx = tmp^T` — cache-friendly
- Alternative dX: avoids large W^T buffer

**Weight update**:
- `SGD`: `sgd_fused_tn` fuses W update with dW computation on the baseline `gemm` path
- `Adam` / `AdamW`: native builds can link MKL as an optional accelerator when it is found
- `Adagrad`: stays on the baseline `gemm` path

**Threading**: `Rayon(5)` for ops < 33M flops, `Rayon(0)` (all cores) for ≥ 33M flops. Rayon(5) is optimal on a 6-core CPU — Rayon(6) adds OS scheduling overhead.

CPU target mode:

- `portable`: default, no `target-cpu=native`
- `native`: opt-in, injects `-C target-cpu=native` unless the user already set a different `target-cpu`

MKL is not a baseline requirement. If it is absent, portable training still works and native Adam/AdamW stay on the fallback GEMM path.

**Windows note**: Rayon thread pool teardown calls `abort()` on `FreeLibrary`. Benchmark executables use `ExitProcess(0)` to bypass this. Output is written via `std::fs::write` to a temp file before `ExitProcess` (bypasses CRT buffer flush issue).

### GEMM shim (`gemm_shim.c`)

Compiled with the same portable/native clang policy as the C training path. Used by the C training path.

- GEMV fast path (m=1): 4-row unrolled, `C[j] += a0*B0[j] + a1*B1[j] + ...`
- Tiled GEMM (m>1): MC=64, KC=256, NC=1024 with pack_A/pack_B
- 4-row unroll chosen (8-row causes register pressure; 1-row misses FMA ILP)

---

## Determinism model

`DeterminismLevel` enum: `Strict`, `Balanced`, `Fast`. Flows through backends, schedulers, and CUDA policy.

- **CPU sequential**: fully deterministic (same inputs → bit-identical outputs)
- **CPU parallel** (`--features parallel`, Rayon): breaks bit-level determinism due to float operation ordering. Explicitly labeled unsafe at the contract level.
- **CUDA**: strict mode rejects operations that cannot guarantee determinism

`#![deny(unsafe_code)]` at crate root. No `unsafe` in library code.

---

## Runtime environment variables

Read at startup via `CompilerFlags::from_env()` in `compiler_flags.rs`:

| Variable | Values | Default |
|---|---|---|
| `VOLTA_STRICT` | `0\|1\|true\|false` | `true` |
| `VOLTA_DEBUG_VERIFY` | `0\|1\|true\|false` | `true` (debug), `false` (release) |
| `VOLTA_UNSAFE_OPT` | `0\|1\|true\|false` | `false` |
| `VOLTA_DETERMINISM` | `strict\|balanced\|fast` | `balanced` |
| `VOLTA_CPU_TARGET` | `portable\|native` | `portable` |
| `VOLTA_GPU_AVAILABLE` | `0\|1\|true\|false` | auto-detect |

For AOT codegen, the explicit CLI flag `--cpu-target <portable|native>` is the contract that matters. The env var is just the ambient default seen by `CompilerFlags`/`doctor`.

---

## Utils (`src/utils/`)

- **`diagnostics.rs`** — error rendering with span highlighting and suggestions
- **`surgeon.rs`** — reverse-engineers GGUF/SafeTensors into `.vt` stubs (`volta extract`)
- **`interop/`**:
  - `contract.rs` — versioned IR serialization
  - `onnx.rs` — ONNX import (behind `onnx-import` feature)
  - `plugin.rs` — `OpImportPlugin` / `PluginRegistry`
  - `python_exporter.rs` — export Volta models to PyTorch code

---

## Security limits

All external-facing allocation paths enforce hard upper bounds to prevent OOM/DoS:

| Guard | Location | Limit |
|---|---|---|
| `MAX_TENSOR_ELEMENTS` | `tensor.rs` | 512 Mi elements (2 GiB f32) |
| `MAX_ONNX_FILE_BYTES` | `onnx.rs` | 2 GiB |

These checks happen before any allocation. `Tensor::new`, `Tensor::zeros`, and `Tensor::ones` all return `TensorError` if the requested size exceeds the limit. `import_onnx_bytes` rejects oversized input before protobuf decoding begins.

---

## Known bugs fixed (non-obvious)

- `Op::Output` node was missing from the IR graph in `executor.rs` — required `lower_ctx.push_op(Op::Output(logits))`
- O3 IPO marks output param `readnone` → eliminated output stores. Fix: use targeted passes without IPO
- Output volatile store required: `build_store(dp, sv).set_volatile(true)`
- LLVM destructors segfault on Windows: `Box::leak(Context)`, `std::mem::forget(module/machine/triple)`
- `sgemm_nt` b_cols bug: must pass `c` (delta cols), not `B` (batch) — wrong strides give wrong gradients
