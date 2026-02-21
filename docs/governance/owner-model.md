# Owner Model

Clear ownership turns "someone should fix this" into "this gets fixed now".

## Roles

- Runtime Owner: Tier A runtime contracts and invariants.
- Model Owner: model-layer correctness and performance quality.
- Language Owner: DSL/frontend consistency.
- Build and Quality Owner: CI policy, gates, flake governance.

## Approval Rules

- Tier A PRs require 2 independent approvals.
- Tier A PRs require 2 independent approvals from qualified reviewers when Runtime Owner is the author.
- Tier B and Tier C changes require at least one owner-aligned reviewer.

## Accountability

- Owner review includes correctness, risk, and rollback readiness.
- Regressions are triaged to the owning domain with clear follow-up actions.
