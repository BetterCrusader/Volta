# Determinism Scope

Determinism is powerful only when its boundaries are explicit.

This document defines deterministic guarantees and explicit non-guarantees.

## Guaranteed Determinism

- Fixed seeds produce repeatable outputs for deterministic code paths.
- IR schedule/allocation/fingerprint checks are deterministic under the same inputs and flags.
- Single-threaded numeric paths are expected to be byte-stable unless otherwise documented.

## Non-Guaranteed Areas

- Multi-thread floating-point accumulation may vary in low-order bits due to reduction order.
- Hardware-dependent vectorization differences can produce small epsilon-level variance.
- CUDA-enabled execution parity is tolerance-based, not bit-identical, unless explicitly guaranteed.

## Seed and Replay Policy

- Tests that depend on pseudo-random generation must set explicit seeds.
- Replay tests must capture seed, flags, and relevant environment variables.
- Any deterministic drift requires incident triage with reproduction metadata.
