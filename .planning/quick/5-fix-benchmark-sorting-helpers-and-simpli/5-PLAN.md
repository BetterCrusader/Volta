# Quick Task 5: fix benchmark sorting helpers and simplify timing organization - Plan

**Gathered:** 2026-03-07
**Status:** In Progress

## Task 1
- files: [examples/bench_real.train_rust._rust_crate/examples/common.rs](/C:/Users/User/Desktop/Volta-main/examples/bench_real.train_rust._rust_crate/examples/common.rs)
- action: add a small shared helper for timing result sorting and median extraction so benchmark examples stop duplicating `sort_by(partial_cmp(...).unwrap())`
- verify: benchmark examples compile against the shared helper
- done: helper is the single place that defines sorting/median behavior for timing samples

## Task 2
- files: [examples/bench_real.train_rust._rust_crate/examples/bench_gemm_backends.rs](/C:/Users/User/Desktop/Volta-main/examples/bench_real.train_rust._rust_crate/examples/bench_gemm_backends.rs), [examples/bench_real.train_rust._rust_crate/examples/bench_inference.rs](/C:/Users/User/Desktop/Volta-main/examples/bench_real.train_rust._rust_crate/examples/bench_inference.rs), [examples/bench_real.train_rust._rust_crate/examples/bench_final.rs](/C:/Users/User/Desktop/Volta-main/examples/bench_real.train_rust._rust_crate/examples/bench_final.rs), [examples/bench_real.train_rust._rust_crate/examples/bench_pipeline.rs](/C:/Users/User/Desktop/Volta-main/examples/bench_real.train_rust._rust_crate/examples/bench_pipeline.rs)
- action: switch benchmark examples from ad-hoc median sorting to the shared helper and remove the most obvious duplicated timing boilerplate where touched
- verify: `cargo build --examples`
- done: touched examples use the same safe timing path and no longer panic on `partial_cmp(...).unwrap()`

## Task 3
- files: [examples/bench_real.train_rust._rust_crate/examples](/C:/Users/User/Desktop/Volta-main/examples/bench_real.train_rust._rust_crate/examples)
- action: finish the mechanical replacement for the remaining benchmark examples, then run formatter and validation so the benchmark crate stays build-clean
- verify: `cargo fmt --all`, `cargo test --quiet`, `cargo clippy --all-targets --quiet`
- done: benchmark crate builds and tests cleanly after the helper extraction, with no `partial_cmp(...).unwrap()` left in examples
