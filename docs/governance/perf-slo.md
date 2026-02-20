# Runtime Performance SLO

## Runtime SLOs

These SLOs apply to stable release gates and nightly quality signals.

- Compile-time P95 for cached execution requests: <= 10 ms equivalent for scaffold backend measurements.
- Compile-time P95 for uncached execution requests: <= 50 ms equivalent for scaffold backend measurements.
- Runtime plan cache hit rate on repeated inference/training workloads: >= 95% after warmup.
- Static memory budget policy compliance: 100% for required XL profiles.

## Measurement Policy

1. P95 metrics are collected from deterministic repeated runs.
2. Plan cache hit rate is measured per backend and determinism mode.
3. Memory budget regressions block release until baseline or architecture decision is updated.
4. Any SLO breach requires incident capture and corrective follow-up.
