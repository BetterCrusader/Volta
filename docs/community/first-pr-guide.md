# Volta First PR Guide

This guide gets a new contributor from clone to merged PR with minimal friction.

## 1. Pick a small scoped task

Start with one of:

- docs clarifications
- error message improvements
- one deterministic test addition
- one parser/semantic guard improvement

Avoid first PRs that modify many subsystems at once.

## 2. Create a branch

```bash
git checkout main
git pull --ff-only origin main
git checkout -b codex/first-pr-<topic>
```

## 3. Implement with determinism-first mindset

Checklist while coding:

1. Does this preserve deterministic behavior?
2. Does this fail loudly on unsupported paths?
3. Is there an explicit test for new behavior?
4. Is the error actionable for users?

## 4. Run required checks

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

If working on ONNX interop:

```bash
bash scripts/ci/interop_onnx_verify.sh
```

## 5. Open PR

Push your branch and open a PR with:

1. Summary of behavior changes.
2. Test evidence.
3. Scope boundaries (what is intentionally not included).

## 6. Respond to review

1. Prefer precise technical responses.
2. Add tests if behavior changed.
3. Keep force-push minimal and intentional.

## 7. Merge criteria

A PR is ready when:

1. CI checks are green.
2. No unresolved review comments.
3. Scope still matches PR description.
