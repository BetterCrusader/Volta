# CI Flaky Registry

This registry tracks flaky tests and unreliable lanes that can degrade CI signal quality.

## Rules

1. Every flaky item must have an owner.
2. Every flaky item must have a linked triage issue.
3. Flaky items that impact blocking lanes are release blockers unless explicitly waived by release owner.
4. Weekly CI review must update status and ETA.

## Weekly CI Review Cadence

- Day: Wednesday
- Owner: CI Reliability Owner
- Inputs: PR gate history, nightly failures, rerun data, issue status
- Output: updated registry rows and mitigation progress summary

## Registry Table

| Lane | Flaky Item | Severity | Owner | Issue | First Seen | ETA | Status | Mitigation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | open | _TBD_ |

## Status Values

- `open`: known flaky behavior under active triage
- `mitigating`: mitigation in progress with active owner
- `monitoring`: mitigation landed, observing for recurrence
- `resolved`: no recurrence observed across one week of blocker-lane runs

## Release Criteria Hook

- Before release cut, all blocking-lane flaky rows must be `resolved` or have explicit release-owner exception notes.
