# Volta

Volta is an experimental ML compiler and runtime written in Rust. It has its own language (`.vt` files), its own IR, its own codegen pipeline, and a CPU training backend that outperforms PyTorch eager mode on a range of MLP workloads.

This is not a framework. It is not production-ready globally. It is a focused engineering project with a determinism-first design and a codegen path that produces verifiably fast native training code.

---

## Why Volta exists

Most ML frameworks sacrifice two things for developer velocity: **determinism** and **raw performance**. PyTorch gives you flexibility, but "same inputs" does not guarantee "same output" across runs, seeds, or thread counts. And its CPU path is general-purpose — it was not built to be beaten on narrow, controlled workloads.

Volta bets on the opposite: a compiler approach where the model is compiled into a native DLL with fixed weights, fixed topology, and deterministic execution. The result, for MLP training, is measurable.

---

## Current strengths / current limitations

**Strengths (measured, reproducible):**
- CPU training codegen outperforms PyTorch eager (6 threads, MKL) by 35–67% on MLP workloads at B≤64
- Deterministic execution by design: same inputs → identical outputs, bit-for-bit
- Full codegen pipeline: `.vt` file → IR → Rust/LLVM → native `.dll` → benchmark
- AVX2 transpose kernels, fused SGD-GEMM, MKL hybrid backend for SGD weight updates
- Working `.vt` language: lexer, parser, semantic analysis, interpreter
- IR with graph optimizations: constant folding, DCE, CSE, algebraic simplification, autograd

**Limitations (honest):**
- CPU performance advantage disappears at B≥128 — PyTorch MKL wins on large GEMMs
- Adam optimizer is +25% faster than PyTorch on B≤64 MLP (4.3 ms vs 5.3 ms; SGD is 35-67% faster)
- CUDA backend: files exist and compile behind `--features cuda`, but GPU perf is uncharted
- No installer, no package, no Python bindings — build from source only
- Not tested on macOS or Linux (developed on Windows 11 x86-64)
- Many high-level layer types (`LSTM`, `MultiHeadAttention`, `LayerNorm`) exist in IR but are not exercised by the codegen path — only dense MLP is benchmarked end-to-end

---

## Quickstart

```bash
git clone https://github.com/BetterCrusader/Volta.git
cd Volta

# Build CLI (requires Rust stable + LLVM 21 for codegen)
cargo build --release

# Run XOR example (interpreter path, no LLVM needed)
cargo run --release -- run examples/xor.vt

# Check syntax
cargo run --release -- check examples/xor.vt

# Run tests
cargo test --workspace
```

**For the codegen/benchmark path**, see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#how-to-reproduce) — requires LLVM 21 and a configured Rust toolchain.

---

## Benchmark highlights

All results: Windows 11, x86-64, 6-core CPU, 30 outer runs × 7 inner × 50 steps, 90s cooldown. No trimming.

| Architecture | B | Volta | PyTorch 6T | Volta faster by |
|---|---|---|---|---|
| MLP 256→512→512→256→1 | 64 | **0.633 ms** | 0.856 ms | **+35%** |
| MLP 512→1024→1024→512→256→1 | 64 | **1.703 ms** | 2.440 ms | **+43%** |
| MLP 512→2048→2048→512→1 | 64 | **5.054 ms** | 8.457 ms | **+67%** |
| MLP 512→1024→1024→512→256→1 | 128 | 3.659 ms | 3.628 ms | **~equal** |

**Honest note**: Volta wins at B≤64. At B=128 the result is statistical parity. At B≥256 PyTorch MKL is faster. Adam optimizer is +25% faster than PyTorch at B≤64. See full tables, caveats, and reproduce instructions in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## Architecture at a glance

```
.vt source
    └─ Lexer → Parser → AST → SemanticAnalyzer
                                    └─ Executor (interpreter path)
                                    └─ IR Builder
                                            └─ Graph Optimizations (DCE, CSE, fold, ...)
                                            └─ Autograd (backward graph)
                                            └─ Scheduler (deterministic topo sort)
                                            └─ Codegen
                                                ├─ LLVM IR → .o → .dll  (inference)
                                                └─ Rust source → cargo  (training DLL)
                                                        └─ gemm crate + MKL hybrid + AVX2
```

Full architecture details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## CLI

```
volta run <file.vt>             # Interpret and execute
volta check <file.vt>           # Syntax + semantic check, no execution
volta info <file.vt>            # Show model topology info
volta compile <file.vt>         # Compile inference DLL (requires LLVM)
volta compile-train <file.vt>   # Compile training DLL
volta compile-train <file.vt> --rust  # Rust-based training DLL (faster)
volta doctor                    # Environment diagnostics
volta export-py <file.vt>       # Export model as Python/PyTorch code
volta extract <model_name>      # Reverse-engineer GGUF/SafeTensors to .vt
volta init [dir]                # Initialize new project
```

---

## Language example

```vt
model brain
    layers 512 1024 1024 512 256 1
    activation relu

dataset bench_data
    type synthetic
    batch 64

train brain on bench_data
    epochs 50
    optimizer sgd
    lr 0.01
    device cpu
```

---

## Install / platform note

- **Windows x86-64**: fully tested
- **Linux/macOS**: not tested; should compile for the interpreter path, codegen path needs LLVM 21
- **LLVM**: required only for `volta compile` (inference DLL). Path: set `LLVM_SYS_210_PREFIX` or put `clang.exe` next to the binary.
- **CUDA**: build with `--features cuda`. GPU perf not benchmarked.

---

## Roadmap snapshot

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full current state and next steps.

Short version:
- **Done (Phases 1–5)**: interpreter, IR, codegen, SGD + Adam training DLL, AVX2 kernels, MKL hybrid, benchmark harness, end-to-end PyTorch parity for MLP/ConvNet/tiny-transformer
- **In progress (Phase 6)**: CLI/help/doctor/examples/docs aligned with real behaviour
- **Planned (Phase 7)**: release artifacts, clean install story, smoke-tested shipped binary
- **Later**: autotune tile sizes, cross-platform build, broader model coverage beyond MLP

---

## License

Apache-2.0

See also: [`DISCLAIMER.md`](DISCLAIMER.md)
