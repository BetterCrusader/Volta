# Volta Quality Policy

This document outlines our engineering quality standards to maintain discipline during rapid growth of the engine codebase and prevent regressions.

## 1. Core Values (Priorities)
1. **Correctness** — Math must add up, logic must not panic or crash, and the AST must parse transparently. Zero tolerance for silent failures or undefined behavior.
2. **Determinism** — Execution results (especially during parallel operations and CUDA kernel dispatch) must be strictly repeatable. No floating-point non-determinism.
3. **Measured Performance** — Execution speed, throughput, and memory footprint matter. "Faster" must be proven by benchmark metrics, not assumed.
4. **Style & Cleanliness** — Readability, architecture, and maintaining `clippy` at 100/100 to reduce cognitive load for contributors.

## 2. Zero Warnings Policy
- **Blocking in CI**: `cargo fmt --check`, `cargo test`, `cargo clippy --all-targets --all-features -- -D warnings`.
- **Non-blocking (Advisory)**: `cargo clippy --all-targets --all-features -- -W clippy::pedantic`. Used as a North Star for continuous improvement but won't block urgent hotfixes.
- **The `#[allow(...)]` Rule**: Usage must be scoped as locally as possible (to a specific function or struct) and **must include a comment** explaining the rationale (e.g., FFI quirks, false positives). Global `#![allow(...)]` declarations are strictly forbidden.

## 3. Bug Tracking & Regressions
- Every identified bug (whether found via fuzzing, code audit, or in production) **must** be accompanied by an isolated regression test that fails before the fix and passes after.
- Examples: tests for infinite recursion in the parser, validation of `epsilon` in Adam optimizer.

## 4. Benchmarking Baseline
- No performance claims are accepted at face value. Any claim of optimization must include a comparison against the baseline:
  - `Lexer / Parser / Semantic Analysis`: Syntax processing speed (`benches/benchmarks.rs`).
  - `Execution`: Inference and training time/memory.
- Benchmarks should be run systematically to track regressions over time.
