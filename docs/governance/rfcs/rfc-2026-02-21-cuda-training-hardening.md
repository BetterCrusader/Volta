# RFC-2026-02-21: CUDA Training Hardening and Pre-Merge Perf Gates

## Status

- Proposed

## Context

- The CUDA training branch introduces runtime-path hardening, memory/placement guards, and stricter pre-merge quality gates.
- Existing policy requires RFC traceability for Tier A and governance-sensitive IR/runtime changes.
- CI must remain deterministic on runners that do not provide CUDA runtime libraries.

## Decision

- Keep CUDA parity and memory guard checks as required gates, but make no-CUDA environments skip CUDA-only determinism assertions safely.
- Enforce a PR perf regression gate that compares against signature-based baselines with generic fallback resolution.
- Extend release perf double-pass reporting to use resolved baseline files from perf-gate outputs.
- Standardize generic baseline artifacts for Linux and Windows signatures to avoid false failures on first-class CI platforms.

## Scope

- Affects CI/workflows, perf scripts, CUDA determinism tests, and baseline governance artifacts.
- Does not change public language syntax or model API contracts.

## Risks

- Baseline drift may hide regressions if budgets are set too loose.
- CUDA-specific issues can still appear only on real GPU hardware.

## Mitigations

- Keep strict budgets for CUDA parity/memory checks.
- Preserve nightly/release lanes for deeper stress and perf double-pass.
- Require explicit RFC traceability for future governance-sensitive changes.

## Rollback

- Revert perf gate workflow additions and script updates.
- Revert CUDA determinism test skip guards if dedicated CUDA runners become mandatory.
- Restore previous baseline strategy if fallback resolution causes governance concerns.

## Acceptance Criteria

- Policy check passes with RFC traceability.
- PR gates pass on non-CUDA hosted runners.
- CUDA lanes still pass on GPU-capable environments.
- Perf gate emits stable JSON and markdown artifacts using resolved baselines.
