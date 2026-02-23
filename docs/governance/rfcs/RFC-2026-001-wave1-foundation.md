# RFC-2026-001: Quality Fortress Wave 1 Foundation

## Problem

Volta needed enforceable governance and CI quality controls before deeper release and performance hardening.
Without explicit policy automation, branch protection and review quality were inconsistent and non-auditable.

## Constraints

- Preserve backward compatibility for current CLI/API behavior.
- Keep Tier A correctness and determinism protections explicit.
- Avoid weakening existing lint/test quality requirements.

## Proposal

Implement Wave 1 governance foundation:

- Add governance policy documents under `docs/governance/`.
- Enforce ownership and PR quality rules via `CODEOWNERS` and PR template.
- Add tier-aware PR gates and policy checks in GitHub Actions.
- Add local verification script mirroring required CI checks.
- Add baseline property-fast and smoke checks.

## Impact on Invariants

- No runtime semantics changed.
- No Tier A numerical or execution invariants changed by this RFC.
- Governance process invariants become machine-enforced.

## Impact on Performance

- No runtime performance path changed.
- CI duration increases due additional policy and smoke jobs.

## Failure Analysis

- Risk: policy workflow false negatives/positives block PRs incorrectly.
- Mitigation: unit tests for policy/tier scripts and local `wave1_local_verify.sh` parity path.
- Risk: governance docs drift from enforced behavior.
- Mitigation: docs presence/content tests under `tests/`.

## Test Plan

- `bash scripts/ci/wave1_local_verify.sh`
- `python scripts/ci/detect_tiers.py --paths src/ir/tensor.rs`
- `python scripts/ci/policy_check.py --paths src/ir/tensor.rs --pr-body "RFC-004"`
- `cargo test --test governance_docs_content -- --nocapture`

## Rollback Plan

- Revert Wave 1 governance commits if policy enforcement blocks unrelated development unexpectedly.
- Restore previous branch protections and workflow required checks after revert.

## Migration Notes

- Developers must include RFC marker in PR body for Tier A/policy-sensitive changes or include an RFC document change in the PR.
- For experimental work, use `exp/*` branches with explicit hardening approval before merge.
