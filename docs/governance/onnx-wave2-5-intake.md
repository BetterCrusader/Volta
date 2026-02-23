# ONNX Wave 2.5 Intake and Delivery Contract

Status: Draft
Related issue: #27

## Purpose

Define exactly how new ONNX operators enter Volta 1.2.0 scope without breaking deterministic and fail-fast guarantees.

## Intake Principles

1. Static-safe subset only.
2. No silent fallback behavior.
3. Unsupported attributes fail explicitly.
4. Every supported path must have parser, contract, lowering/runtime, tests, and docs.
5. Documentation support status must reflect shipped behavior only.

## Candidate Operator Set (1.2.0)

| Operator | Priority | Contract Type | Notes |
| --- | --- | --- | --- |
| `LeakyRelu` | P0 | inference/static | include `alpha` attribute handling with deterministic default |
| `BatchNorm` | P0 | inference/static | training-mode semantics out of scope for 1.2.0 |
| `MaxPool` | P0 | inference/static | fixed static subset only |
| `AveragePool` | P0 | inference/static | fixed static subset only |
| `LayerNorm` | P1 | inference/static | ship only if contract remains stable and testable |

## Implementation Progress

- 2026-02-23: `LeakyRelu` landed with ONNX importer support and deterministic lowering to primitive IR ops.
- Verified by `interop_onnx_wave2_parser` and `interop_onnx_verify` suites.

## Acceptance Contract per Operator

An operator is considered shipped only if all boxes are complete:

| Requirement | Required |
| --- | --- |
| Parser mapping in ONNX importer | yes |
| Contract validation and clear unsupported-path errors | yes |
| Lowering/runtime path or explicit fail-fast path | yes |
| Positive tests | yes |
| Negative tests for unsupported attrs/combinations | yes |
| Coverage matrix update (`docs/ONNX_COVERAGE.md`) | yes |

## Unsupported Attribute Policy

- If attribute or mode is not supported, importer must fail with explicit message:
  - include node name
  - include attribute name/value
  - include supported alternatives when available

## Test Matrix Template

For each operator, add these groups:

1. happy-path import test
2. malformed attribute failure test
3. unsupported mode failure test
4. shape/axis constraint failure test (if applicable)
5. runtime parity test (if runtime path exists)

## Exit Criteria for Issue #27

1. At least 4 operators from candidate set are shipped with full acceptance contract.
2. Coverage matrix reflects exact support level and limits.
3. No aspirational entries remain in ONNX docs for 1.2.0 scope.
