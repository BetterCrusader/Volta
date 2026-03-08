# Volta

Volta is an experimental ML compiler and runtime written in Rust. It has its own language (`.vt` files), its own IR, its own codegen pipeline, and a CPU training backend that outperforms PyTorch eager mode on a range of MLP workloads.

Tier 1 source/runtime CPU support today is **x86_64 + ARM64**. Everything else is **best-effort**. That CPU support statement is separate from which prebuilt release artifacts are currently shipped.

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
- Explicit CPU target selection for AOT codegen: portable default, native opt-in
- AVX2 transpose kernels and optional MKL acceleration for native Adam/AdamW on the Rust training path
- Working `.vt` language: lexer, parser, semantic analysis, interpreter
- IR with graph optimizations: constant folding, DCE, CSE, algebraic simplification, autograd

**Limitations (honest):**
- CPU performance advantage disappears at B≥128 — PyTorch MKL wins on large GEMMs
- Adam optimizer is +25% faster than PyTorch on B≤64 MLP (4.3 ms vs 5.3 ms; SGD is 35-67% faster)
- CUDA backend: files exist and compile behind `--features cuda`, but GPU perf is uncharted
- `compile-train` is still **MLP-only**
- Default C `compile-train` path is **SGD-only**; `compile-train --rust` supports only **SGD, Adam, AdamW, Adagrad**
- Tier 1 source/runtime CPU support is x86_64 + ARM64; shipped release artifacts are narrower than that
- Prebuilt binaries are currently shipped for Linux x86_64, macOS universal, and Windows x86_64. No Python bindings.
- Many high-level layer types (`LSTM`, `MultiHeadAttention`, `LayerNorm`) exist in IR but are not exercised by the codegen path — only dense MLP is benchmarked end-to-end

---

## Install

Download the prebuilt binary for your platform from the
[latest release](https://github.com/BetterCrusader/Volta/releases/latest):

These release artifacts are the current packaged binaries, not the full CPU support matrix.

| Platform | File |
|----------|------|
| Linux x86-64 | `volta-vX.Y.Z-linux-x86_64.tar.gz` |
| Linux ARM64 | `volta-vX.Y.Z-linux-arm64.tar.gz` |
| macOS (Apple Silicon + Intel) | `volta-vX.Y.Z-macos-universal.tar.gz` |
| Windows x86-64 | `volta-vX.Y.Z-windows-x86_64.zip` |

Replace `vX.Y.Z` with the release version shown on the releases page.

**Linux / macOS:**

```bash
tar xzf volta-vX.Y.Z-*.tar.gz
chmod +x volta
mv volta /usr/local/bin/    # or any directory already on your PATH
volta --help
```

**Windows:**
Unzip `volta-vX.Y.Z-windows-x86_64.zip`, then move `volta.exe` to a
directory on your `PATH` (or add its folder to `PATH`).

**Verify the install:**

```bash
volta --help
volta doctor
volta run examples/xor.vt   # requires the examples/ directory from the repo
```

> `volta run` requires a `.vt` source file on disk. The binary is
> self-contained; example files are in the repository.

No Rust toolchain required to run prebuilt binaries.

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

These numbers are from the measured **Windows x86_64 native-path MLP benchmarks**. They are not portable-baseline numbers, and they are not ARM64 benchmark claims.

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
                                                        └─ gemm crate + optional MKL accel + AVX2
```

Full architecture details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## CLI

```
volta run <file.vt>             # Interpret and execute
volta check <file.vt>           # Syntax + semantic check, no execution
volta info <file.vt>            # Show model topology info
volta compile <file.vt> [--cpu-target portable|native]         # Compile inference DLL (requires LLVM)
volta compile-train <file.vt> [--cpu-target portable|native]   # MLP-only training DLL, C path = SGD only
volta compile-train <file.vt> --rust [--cpu-target portable|native]  # MLP-only Rust path = SGD/Adam/AdamW/Adagrad
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

## CPU and artifact scope

- **Tier 1 source/runtime CPU support**: x86_64 + ARM64
- **Other CPU architectures**: best-effort only
- **Shipped release artifacts today**: Linux x86_64, Linux ARM64, macOS universal, Windows x86_64
- **`--cpu-target`**: `portable` is the default for `compile` and `compile-train`; `native` is explicit opt-in and may emit host-specific binaries
- **`compile-train` coverage**: MLP-only today
- **C `compile-train` path**: SGD only
- **Rust `compile-train --rust` path**: SGD, Adam, AdamW, Adagrad
- **MKL**: optional accelerator for native Adam/AdamW on the Rust path, not a baseline requirement
- **LLVM**: required only for `volta compile` (inference DLL). Set `LLVM_SYS_210_PREFIX` or place `clang` next to the binary.
- **CUDA**: build with `--features cuda`. GPU perf not benchmarked.

---

## Roadmap snapshot

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full current state and next steps.

Short version:
- **Done (Phases 1–5)**: interpreter, IR, codegen, optimizer coverage split made explicit, benchmark harness, end-to-end PyTorch parity for MLP/ConvNet/tiny-transformer
- **Done (Phases 6–7)**: product-surface hardening, truthful CPU portability docs, and release/workflow verification aligned with the actually shipped artifacts on x86_64 and ARM64
- **Later**: CPU throughput/autotuning, broader model coverage beyond MLP, more packaged targets beyond the current supported set

---

## License

Apache-2.0

See also: [`DISCLAIMER.md`](DISCLAIMER.md)
