# Volta V1 Release Checklist

## Scope Lock

- Freeze feature scope for `v1.0.0`.
- Confirm supported interop scope for Wave 1:
  - ONNX import subset: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `MatMul`, `Transpose`, `Relu`, `Softmax`
  - static shape tensors only
  - deterministic runtime policy unchanged

## Quality Gates

- `cargo fmt --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test`
- `cargo test --release`
- `bash scripts/ci/interop_onnx_verify.sh`
- `bash scripts/ci/xl_verify.sh`
- `bash scripts/ci/release_perf_double_pass.sh`

## Governance and Policy

- RFCs for shipped features are merged.
- `docs/governance` policy docs are up to date.
- Baselines in `benchmarks/baselines/` are present and reviewed.

## Release Artifacts

- Tag format: `release-v1.0.0`
- Changelog sections:
  - IR/Runtime
  - CUDA/Perf
  - Interop/ONNX
  - Breaking changes / migration notes
- Confirm rollback procedure:
  - `bash scripts/release/rollback.sh --verify-only`

## Sign-off

- CI green on `main` at release commit.
- Release PR approved.
- Tag and publish.
