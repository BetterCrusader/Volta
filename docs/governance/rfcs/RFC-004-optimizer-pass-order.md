# RFC-004: Optimizer Pass Order and Determinism Contract

- Status: Accepted
- Owner: IR/Runtime team
- Date: 2026-02-23

## Context

Volta applies multiple IR passes before building an execution plan. Correctness and reproducibility depend not only on individual passes, but on the *order* in which they run.

Without a codified order, two equivalent developer changes can produce different graph fingerprints, schedules, and memory plans, which can break strict determinism guarantees.

## Decision

Volta defines a canonical pass order and treats it as a Tier-A governance contract:

1. Algebraic simplification
2. Constant folding
3. Tensor constant propagation
4. CSE
5. DCE
6. Dead tensor elimination
7. Elementwise fusion
8. Gradient fusion

Any reordering requires:

- an RFC update,
- determinism regression evidence,
- pass equivalence evidence,
- and explicit review sign-off.

## Why This Order

- **Early simplification/folding** reduces graph size before structural passes.
- **CSE after folding** maximizes duplicate-elimination opportunities.
- **DCE after CSE** removes newly orphaned values.
- **Fusion late** avoids hiding optimization opportunities from earlier semantic-preserving passes.

## Determinism Invariants

- Same input graph + same pass order => stable fingerprint.
- Stable fingerprint + stable scheduler semantics => stable schedule hash.
- Stable schedule => stable allocation and placement plans.

## Failure Analysis

If a determinism test fails:

1. Capture pass-by-pass graph snapshots.
2. Compare first divergent fingerprint.
3. Isolate pass introducing nondeterministic iteration.
4. Patch to deterministic container/iteration semantics.
5. Re-run 100-run determinism regressions.

## Rollback Plan

If a new pass-order change causes instability:

- revert to canonical order immediately,
- keep change behind an experimental branch,
- require fresh RFC evidence before re-introduction.
