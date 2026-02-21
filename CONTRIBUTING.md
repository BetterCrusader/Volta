# Contributing to Volta

Volta prioritizes determinism, verifier-backed correctness, and explicit contracts over rapid unchecked feature growth.

## Development setup

1. Install stable Rust and Cargo.
2. Clone repository.
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

## Branch and PR workflow

1. Branch from `main`.
2. Keep changes scoped to one clear objective.
3. Include tests for behavior changes.
4. Push branch and open PR.
5. Merge only when all CI checks are green.

Recommended branch naming:

- `codex/<feature-or-fix>`
- `milestone/<milestone-name>`

## Quality gates (required)

Before opening a PR, run:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
bash scripts/ci/interop_onnx_verify.sh
bash scripts/ci/xl_verify.sh
```

Windows users can run release parity flow with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/release/cut_v1.ps1
```

## Engineering rules

1. No silent fallback paths.
2. No verifier bypasses.
3. Preserve deterministic behavior.
4. Keep contract/version changes explicit.
5. If a feature is parsed but not lowered, fail loudly with actionable error text.

## Commit guidance

Write descriptive commit messages tied to impact:

- `feat(interop): add wave2 op parsing guards`
- `fix(ci): align docs sync assertions`
- `docs(community): add first-pr guide`

## Reporting issues

Include:

1. Minimal reproduction.
2. Expected behavior.
3. Actual behavior.
4. Environment (`rustc -V`, OS, feature flags).
