# Incident Playbook

When failures happen, this playbook reduces panic and speeds containment.

## Severity

- `SEV-1`: production-blocking correctness or data integrity failure.
- `SEV-2`: major regression with workaround.
- `SEV-3`: minor reliability/performance issue.

## Triage

- Record trigger, affected tier, and first observed commit/tag.
- Assign incident owner and backup owner.
- Capture reproducible command and environment metadata.

## Containment

- Block risky merges to affected tier when needed.
- Apply mitigation or temporary guard if safe.

## Rollback

- Use scripted rollback to previous stable release pointer.
- Confirm rollback with smoke and core validation tests.

## Follow-up

- Add regression test before final closure.
- Publish postmortem with root cause and prevention action.
