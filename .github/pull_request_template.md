## What This PR Changes

- Keep this to 1-3 concrete bullets.
- Say what users/runtime behavior gets better.

## Blast Radius (Tier)

- [ ] Tier A (`src/ir`, `src/device`, tensor core, matmul, KV internals)
- [ ] Tier B (`src/model`)
- [ ] Tier C (DSL/frontend/autopilot/executor)

## Quality Fortress (must be green)

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test`
- [ ] `cargo test --release`
- [ ] CLI smoke checks passed (`scripts/ci/cli_smoke.sh` or `scripts/ci/cli_smoke.ps1`)

## Performance Proof (required for Tier A/B perf-sensitive changes)

- Median delta vs baseline:
- Sigma (stddev):
- CPU signature:

## Determinism and Safety

- [ ] Determinism scope reviewed for this change
- [ ] Numeric tolerances unchanged or updated with justification
- [ ] No runtime panic path introduced

## Rollback Ready

- Previous stable pointer/tag:
- Rollback approach:

## Governance Links

- RFC reference (required for policy/Tier A invariant changes):
- Incident links (if regression/hotfix):
