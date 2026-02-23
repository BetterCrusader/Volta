# CI Stale Reference Sweep (2026-02-23)

Related issue: #38

## Goal

Verify that stale CI/doc command references removed in recent cleanup do not remain in the repository.

## Sweep Commands

```bash
grep pattern: scripts\.ci\.tests|python -m unittest scripts\.ci\.tests
include: *.{md,sh,yml,yaml,py}

grep pattern: schedule_optimization
include: *.{md,sh,yml,yaml,py,rs}
```

## Result

- No matches found for removed `scripts.ci.tests.*` invocation patterns.
- No stale `schedule_optimization` references found in docs/scripts/workflows/code.

## Follow-up

- Keep this sweep in weekly CI reliability review.
- Re-run after large CI/script refactors and before release cut.
