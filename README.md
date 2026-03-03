# Volta

**Volta** is an experimental, deterministic-first Machine Learning compiler and runtime written in Rust.

Most ML stacks optimize for velocity first and explain behavior later. Volta does the opposite: **same inputs, same graph, same parameters, same result** is the absolute contract. It is designed to prioritize hard replayability, deterministic scheduling, and explicit failure semantics over broad ecosystem coverage.

## Core Philosophy

- **Determinism by Design:** Replay discipline is built into the scheduler, memory allocation, and runtime behavior.
- **No Silent Fallbacks:** Unsupported operations or execution paths fail loudly, rather than silently switching semantics.
- **Verifier-First Graph:** Graph execution is strictly validated before any operations begin.

## Architecture

Volta operates on its own Intermediate Representation (IR) and supports automatic differentiation (`autograd`). It includes its own lexer and parser for defining models and execution logic.

### Backends

1. **CPU:** Fully deterministic sequential execution. Multi-core execution is optionally supported via `rayon` (note: enabling multi-threading may alter the floating-point order of operations, trading exact bit-for-bit determinism for speed).
2. **CUDA:** Hardware-accelerated execution via `cudarc`, featuring custom kernels and memory management tailored for determinism.

## Quick Start

Make sure you have [Rust](https://rustup.rs/) installed. If you intend to use the CUDA backend, ensure you have the appropriate NVIDIA drivers and CUDA toolkit installed.

```bash
# Compile and check the workspace
cargo check

# Run the test suite (CPU tests)
cargo test --workspace

# Run tests including CUDA (requires a compatible GPU)
cargo test --workspace --features cuda
```

### CLI Usage

Volta provides a built-in CLI for executing its custom DSL (`.vt` files).

```bash
volta run <file.vt> [--quiet]
volta check <file.vt> [--quiet]
volta info <file.vt>
volta doctor [--json] [--strict]
volta init [project_dir]
```

## Language Snapshot

```vt
lr 0.001

model brain
    layers 784 256 128 10
    activation relu
    optimizer adam lr

dataset mnist
    batch 32
    shuffle true

train brain on mnist
    epochs 3
    device auto

print "training complete"
```

## Disclaimer

Volta is experimental software. It is not intended to be a drop-in replacement for comprehensive frameworks like PyTorch, but rather a specialized engine for environments where execution reproducibility is a critical requirement.
