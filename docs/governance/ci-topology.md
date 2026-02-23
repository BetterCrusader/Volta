# CI Topology

If you cannot explain your CI topology in one page, you do not control your release risk.

## PR Pipelines

- `pr-gates.yml` (blocking): fmt, clippy, debug tests, onnx-import tests, release tests, CLI smoke, property-fast.
- Tier A-sensitive changes activate additional governance checks.

## Release Pipelines

- `release-gates.yml` (blocking):
  - Wave 1 gates
  - fuzz smoke
  - short soak
  - double-pass perf gate
  - rollback verification

## Nightly Pipelines

- `nightly-quality.yml`:
  - heavy fuzz suite
  - long soak suite
  - perf matrix and report artifact generation
  - automatic issue update/open when nightly jobs fail

## Failure Handling

- Any blocking failure prevents merge/release.
- Nightly failures create or update a `nightly-regression` issue for triage.

## Flaky Ownership

- Flaky tests and lanes are tracked in `docs/governance/ci-flaky-registry.md`.
- Every flaky item must have an explicit owner and triage issue.
- Weekly CI health review updates status, ETA, and mitigation notes.
- Weekly review format is defined in `docs/governance/ci-weekly-health-template.md`.
- A release candidate requires one full week of stable blocker-lane signal or explicit owner-approved exception notes.

## Local Replication

- `scripts/ci/wave1_local_verify.sh` reproduces Wave 1 checks.
- `scripts/ci/wave23_local_verify.sh` reproduces Wave 2+3 checks including perf, soak, and rollback verify.
- Stale command reference audit evidence is tracked in `docs/governance/ci-stale-reference-sweep-2026-02-23.md`.
