# Volta Quality Fortress+ Design

## Goal

Transition Volta from R&D discipline into product governance with enforceable quality gates,
backward-compatible runtime behavior, and deterministic verification policy.

## Tier Model

- Tier A: `src/ir`, `src/device`, tensor core, matmul tuner, KV internals
- Tier B: `src/model`
- Tier C: DSL/frontend/autopilot/executor

Tier A gets hardest blocking gates. Tier B and Tier C stay strict on correctness while keeping
perf governance proportional.

## Quality Bar

- Required: `fmt`, `clippy -D warnings`, debug tests, release tests, CLI smoke
- Required depth: property-fast, determinism checks, fuzz/chaos progression, soak progression
- Docs may only claim behavior that is validated by current commands/metrics

## Workstreams

1. Contract freeze (invariants, tolerances, fallback semantics)
2. Tiered gate matrix
3. Determinism and invariant reinforcement
4. Fuzz/chaos coverage expansion
5. Perf governance (PR soft, release double-pass blocking)
6. Soak/reliability policy
7. CI topology split (PR / Release / Nightly)
8. Docs as verified truth

## Operational Policy

- Protected branches: `main`, `release/*`
- `exp/*` allowed for sandbox work, no direct protected merge without hardening
- Tier A bypasses are disallowed
- Rollback is scripted and validated

## Ownership and RFC Policy

- Runtime Owner, Model Owner, Language Owner, Build & Quality Owner
- Tier A requires 2 independent approvals (including independent reviewers when author is Runtime Owner)
- RFC mandatory for Tier A policy and determinism/perf/release governance changes
- RFC includes failure analysis, test plan, and rollback plan

## Wave Plan

- Wave 1: governance docs, ownership policy, PR gates, tier detection, policy checks
- Wave 2: release-depth checks (double perf pass, short soak, fuzz smoke)
- Wave 3: nightly heavy fuzz/soak/full benchmarks with flake automation
