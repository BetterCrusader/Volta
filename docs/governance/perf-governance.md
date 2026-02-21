# Performance Governance

If a release is faster on your laptop but slower in production, governance failed.

## Baseline Identity

- Perf baselines are keyed by CPU signature.
- Signature source: `scripts/perf/cpu_signature.py`.
- Baseline location: `benchmarks/baselines/<signature>.json`.

## Probe Contract

- Probe command: `cargo run --release --bin perf_probe -- ...`.
- Probe emits JSON with metric median/stdev and `lower_is_better` semantics.

## PR Policy

- Perf checks on PR are advisory-first; regressions are surfaced with metrics and deltas.
- PR comments/report artifacts include median and stdev to reduce noise-driven decisions.

## Release Policy

- Release perf gating uses a double-pass policy with two independent probe passes.
- Release fails only when both passes breach threshold.
- Default threshold: `5%` regression per metric.

## Baseline Bootstrap

- Missing baseline fails release by default.
- Controlled bootstrap is allowed via manual workflow input for first-run setup.

## Reporting

- Markdown compare output is generated per pass via `scripts/perf/baseline_compare.py`.
- Reports are uploaded as workflow artifacts for auditability.
