# Quality Policy

## Purpose

Define repository-wide quality gates required before claiming implementation completeness.

## Mandatory Gates

Every integration-ready change must pass:

1. `cargo fmt`
2. `cargo clippy`
3. `cargo test`
4. `cargo test --features onnx-import`

For fuzz-capable branches, the fuzz crate must at least compile:

5. `cargo check --manifest-path fuzz/Cargo.toml`

## Determinism and Safety Rules

- Determinism regressions are blockers for Tier-A paths.
- Unsupported features must fail explicitly (no silent fallback masking correctness).
- Documentation claims must match executable behavior (no aspirational matrix entries).

## Release Readiness

A branch is release-ready only when:

- all mandatory gates are green,
- coverage/status docs are updated (`ONNX_COVERAGE.md` and governance docs),
- and no pending Tier-A correctness TODO remains open.
