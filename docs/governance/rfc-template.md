# RFC Template

Use this template when proposing changes that can affect correctness, determinism, performance, or release safety.

## Header

- RFC ID (`RFC-YYYY-XXX`)
- Title
- Authors
- Status (`Draft`, `Review`, `Accepted`, `Implemented`, `Verified`)

## Problem

Describe the issue and why current behavior is insufficient.

## Constraints

List non-negotiable constraints (compatibility, determinism, performance, safety).

## Proposal

Describe intended design and alternatives considered.

## Impact on Invariants

Describe what Tier A/B/C contracts are affected.

## Impact on Performance

State expected impact and how it will be measured.

## Failure Analysis

Describe what could go wrong, blast radius, and mitigation.

## Test Plan

List exact tests and verification commands.

## Rollback Plan

Describe fallback path and revert criteria.

## Migration Notes

Document any migration or operator steps.
