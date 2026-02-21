# Operational Policy

Shipping fast is easy. Shipping safely and repeatedly is the hard part.

## Protected Branches

- `main` and `release/*` are protected branches.
- Tier A changes cannot use temporary gate bypasses.

## Branch Classes

- `task/*`: implementation work branches.
- `exp/*`: sandbox-only branches for experiments; no direct merge into protected branches.

## Required Gates

- Formatting, lint, debug tests, release tests, and CLI smoke must pass before merge.
- Policy checks must pass for governance and Tier A sensitive changes.
- Release candidates must pass short soak, fuzz smoke, and double-pass perf gates.
- Nightly runs execute heavy fuzz, long soak, and perf matrix reporting.

## rollback

- Every release candidate must include a rollback pointer to previous stable tag.
- rollback procedure is scripted and validated in release workflow.
- Incident response follows severity-based escalation and documented ownership.

## Quality Freeze Trigger

- Two consecutive degraded weekly quality reports require owner review.
- Owner review can activate temporary feature freeze until regression trend is resolved.
