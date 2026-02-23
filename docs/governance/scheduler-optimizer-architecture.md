# Scheduler and Optimizer Architecture

## Scope

This document describes how Volta turns verified SSA IR into deterministic execution order and stable optimization outcomes.

## Scheduler Design

- Input: verified acyclic SSA graph.
- Output: topological `ordered_nodes` list.
- Determinism rule: ready-set traversal must be stable for identical input graphs.

### Core Data Flow

```text
Graph -> value_to_node map
      -> indegree/edges
      -> ready queue (zero indegree)
      -> ordered_nodes
```

### Determinism Guardrails

- No hash iteration in order-sensitive branches.
- Stable node identity (`NodeId`) usage across planning.
- Schedule hash regression tests over repeated runs and threads.

## Optimizer Pass Pipeline

Canonical order is documented in RFC-004 and enforced operationally.

```text
Simplify -> Fold -> Propagate -> CSE -> DCE -> DeadTensorElim -> Fusion
```

Each pass must preserve graph semantics and verifier validity.

## Interaction Between Scheduler and Passes

1. Passes mutate graph structure.
2. Graph is re-verified.
3. Scheduler builds deterministic topological order.
4. Allocation and backend placement consume schedule.

If pass output is nondeterministic, scheduler determinism cannot recover correctness; therefore determinism starts at pass-level containers and iteration strategy.

## Risk Areas

- Hidden hash map iteration in pass internals.
- Implicit ordering from insertion side effects.
- Fused-group construction with unstable member ordering.

## Validation Checklist

- Pass equivalence tests green.
- Determinism regressions (multi-run, multi-thread) green.
- Fingerprint stability checks green.
- `cargo fmt`, `cargo clippy`, and full test suite green.
