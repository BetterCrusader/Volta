## Summary

- Describe the change in 1-3 bullets.

## Tier Impact

- [ ] Tier A (`src/ir`, `src/device`, tensor core, matmul, KV internals)
- [ ] Tier B (`src/model`)
- [ ] Tier C (DSL/frontend/autopilot/executor)

## Quality Fortress Checklist

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test`
- [ ] `cargo test --release`
- [ ] CLI smoke checks passed (`scripts/ci/cli_smoke.sh` or `scripts/ci/cli_smoke.ps1`)

## Performance Evidence (required for Tier A/B perf-sensitive changes)

- Median delta vs baseline:
- Sigma (stddev):
- CPU signature:

## Determinism and Safety

- [ ] Determinism scope reviewed for this change
- [ ] Numeric tolerances unchanged or updated with justification
- [ ] No runtime panic path introduced

## Rollback Readiness

- Previous stable pointer/tag:
- Rollback approach:

## Governance Metadata

- RFC reference (required for policy/Tier A invariant changes):
- Incident links (if regression/hotfix):
