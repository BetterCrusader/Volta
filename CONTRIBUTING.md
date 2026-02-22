# Contributing to Volta

You are contributing to a deterministic compiler core, not a demo repo.
Every change should improve clarity, correctness, and replayability.

## Quick Setup

1. Install stable Rust + Cargo.
2. Clone the repository.
3. Run baseline checks:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

For ONNX interop work:

```bash
cargo test --features onnx-import --test interop_onnx_parser
```

## Branch and PR Workflow

1. Branch from `main`.
2. Keep each PR scoped to one objective.
3. Include tests for behavior changes.
4. Push branch and open PR.
5. Merge only when all CI checks are green.

Recommended branch names:

- `codex/<feature-or-fix>`
- `milestone/<milestone-name>`

## Required Quality Gates

Before opening a PR, run:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
bash scripts/ci/interop_onnx_verify.sh
bash scripts/ci/xl_verify.sh
```

Windows release parity path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/release/cut_v1.ps1
```

## Engineering Rules (Hard)

1. No silent fallback paths.
2. No verifier bypasses.
3. Preserve deterministic behavior.
4. Keep contract/version changes explicit.
5. If parsed but not lowered, fail loudly with actionable diagnostics.

## Commit Message Guidance

Use impact-oriented commits:

- `feat(interop): add wave2 op parsing guards`
- `fix(ci): align docs sync assertions`
- `docs(community): improve onboarding and guardrails`

## Reporting Issues

Always include:

1. Minimal reproduction.
2. Expected behavior.
3. Actual behavior.
4. Environment (`rustc -V`, OS, feature flags).

## Review Bar

A good PR in Volta is:

- deterministic by design,
- explicit about risk and rollback,
- easy to verify from CI output.
