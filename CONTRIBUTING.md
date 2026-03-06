# Contributing

## Prerequisites

- Rust stable toolchain (`rustup toolchain install stable`)
- For codegen path: LLVM 21 (`LLVM_SYS_210_PREFIX` set, or `clang` in PATH)
- For benchmarks: MKL via conda (`conda create -n mkl -c conda-forge mkl`)

## Running tests

```bash
# All unit tests (interpreter path — no LLVM required)
cargo test --locked --workspace

# With all features except codegen ones
cargo test --locked --workspace --no-default-features
```

Tests must pass with `--locked`. Do not update `Cargo.lock` without a reason — pin changes go in a dedicated commit.

## Code style

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

Both must be clean before a PR is merged. CI enforces this.

## What is accepted in PRs

- Bug fixes with a failing test that now passes
- Performance improvements with before/after numbers (same protocol as `docs/BENCHMARKS.md`)
- Documentation fixes
- New `.vt` language features that pass semantic analysis and have tests

## What is not accepted

- Changes that break `cargo test --locked`
- Benchmark claims without reproducible numbers
- New dependencies without a clear justification
- Unsafe code (`#![deny(unsafe_code)]` is enforced at the crate root)

## Commit style

One logical change per commit. Message format:

```
area: short description (imperative mood)

Optional body explaining why, not what.
```

Examples: `tensor: add MAX_TENSOR_ELEMENTS guard`, `codegen: fix dX stride bug`

## Benchmarks

If your change affects performance, run the primary benchmark before and after:

```bash
cd examples/bench_real.train_rust._rust_crate
cargo build --release --examples
# 90s CPU cooldown, then:
./target/release/examples/bench_official_v2.exe
```

Regression threshold: B=64 MLP-512 median must stay **below 2.10 ms**. See `docs/BENCHMARKS.md` for full protocol.
