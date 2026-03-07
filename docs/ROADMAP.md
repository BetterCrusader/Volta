# Volta Roadmap

Honest state of the project. No aspirational lists. Only what is done, what is next, and what is genuinely uncertain.

---

## Working rules

- Volta is not trying to become "all accelerators, all models, all platforms" in one jump.
- New backend work is only valid if it improves a measured, reproducible end-to-end path.
- Claims about GPU support are invalid until correctness and benchmark parity exist against PyTorch.
- Breadth only comes after a narrow path is numerically correct, benchmarked, and regression-tested.

---

## Done

### Core pipeline
- `.vt` language: lexer, parser, AST, semantic analyzer
- IR: directed computation graph, all basic ops, op-level verifier
- Graph optimization passes: DCE, CSE, constant folding, algebraic simplification, elementwise fusion, gradient fusion
- Autograd: full backward graph construction without mutation
- Deterministic scheduler: topological sort, no randomized ordering
- CPU interpreter: sequential, bit-level deterministic
- Plan cache: keyed by graph fingerprint

### Codegen
- Inference DLL: LLVM IR → `.o` → `.dll` with embedded weights
- Training DLL (C path): generated C → clang -O3
- Training DLL (Rust path): generated Rust crate → cargo → native DLL
- AVX2 8×8 transpose kernel
- `sgd_fused_tn`: fused dW + weight update into one GEMM call
- MKL hybrid: `cblas_sgemm` for SGD weight updates
- Adaptive thread count (Rayon)

### Benchmarked and verified
- CPU training: Volta beats PyTorch eager (6T, MKL) by 35–67% at B≤64 across three MLP architectures
- Batch sweep: parity at B=128, PyTorch wins at B≥256
- Adam verified numerically correct but 1.9× slower than PyTorch

### Infrastructure
- Benchmark harness: 30 outer × 7 inner × 50 steps, 90s cooldown, p50/p95/min/max
- Python harness scripts for reproducible 30-run comparison
- Regression threshold: B=64 MLP-512 median < 2.10 ms
- CLI: `run`, `check`, `info`, `compile`, `compile-train`, `doctor`, `extract`, `init`
- CUDA backend: compiles behind `--features cuda`, GPU perf not measured

---

## Next (concrete, near-term)

### Adam optimizer fusion
Adam is 1.9× slower than PyTorch because it cannot use `sgd_fused_tn`. Need either a fused Adam-GEMM or a significantly faster element-wise update path. This is the highest-priority CPU performance item.

### AutoTuner for tile sizes
Current tile sizes (MC=64, KC=256, NC=1024) are hand-tuned for the primary benchmark machine. An autotuner that samples a few configurations at DLL compile time would improve portability.

### Cross-platform build
Currently only tested on Windows 11 x86-64. Linux/macOS support for the interpreter path is likely close; codegen path needs LLVM path configuration and testing.

### RUSTFLAGS propagation
Benchmark exes are built with `target-cpu=native` manually. The main `volta compile-train --rust` path should propagate this automatically.

### Backend contract hardening
Backend capability metadata now needs to stay explicit: device class, vendor, maturity, runtime support, gradient-update support, and determinism coverage. This is a prerequisite for any future AMD/Intel work — otherwise backend code will drift into undocumented special cases.

---

## Later (real but lower priority)

### GPU performance measurement
The CUDA backend compiles. It has never been benchmarked against PyTorch GPU. This is a significant gap — no GPU claims can be made until measurements exist.

### Broader model coverage in codegen
Only dense MLP is exercised by the codegen path. Conv2D, LayerNorm, MultiHeadAttention, LSTM exist in IR but their codegen is not implemented or not tested end-to-end. Extending the training DLL to support these would be a major effort.

### Packaging and installer
No binary releases. No `cargo install`. No Python bindings. Currently build-from-source only.

### ONNX import
The `onnx-import` feature exists. Its practical completeness is unknown — not tested on real models.

---

## Execution order

This is the order that should actually be followed. Skipping it would be dishonest engineering.

### Phase 1: CPU hardening
- Close the Adam gap
- Add autotuning or at least machine-specific tuning persistence
- Propagate `target-cpu=native` in the normal compile path
- Add stronger regression gates against existing benchmark numbers

### Phase 2: Truthful backend discipline
- Keep backend capabilities explicit and validated at runtime
- Add a backend matrix for execution mode, determinism, and maturity
- Refuse unsupported combinations early instead of silently falling back

### Phase 3: PyTorch parity
- Build repeatable correctness checks against PyTorch for forward values
- Add gradient parity checks
- Add training-curve parity checks on representative workloads

### Phase 4: Broader model coverage
- Expand beyond dense MLP only after the parity harness is stable
- Pick a small set of workloads: MLP, ConvNet, one attention block
- Measure each end-to-end before calling support "done"

### Phase 5: GPU reality check
- Benchmark CUDA against PyTorch GPU
- Publish the real numbers, even if they are bad
- Decide whether CUDA remains an active performance path or stays experimental

### Phase 6: Additional accelerators
- Evaluate ROCm/HIP or Intel oneAPI only after Phase 5 succeeds
- Choose one new accelerator path at a time
- Do not start AMD and Intel in parallel

---

## Not planned

- Distributed training (multi-node, NCCL)
- Model zoo / pretrained weights
- "One-Click Deploy" to cloud
- HuggingFace Hub integration
- Browser/WASM execution

These may be interesting eventually, but are not on any near-term horizon. Adding them here would be dishonest.
