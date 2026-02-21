# RFC: Interop Wave 1 - Stable IR Contract

- Status: Proposed
- Author: Volta maintainers
- Date: 2026-02-21
- Scope: External model import, deterministic runtime parity, plugin-based extension

## Why

Volta already has a strong internal IR and hard CI gates.
To import external models safely, we need a stable, versioned contract that is:

- deterministic by default
- strict about graph validity
- explicit about tensor shapes and dtypes
- extensible through plugins without destabilizing core runtime behavior

## Goals

1. Define a stable IR contract (`V1`) with explicit node IDs and typed tensor specs.
2. Add a minimal ONNX importer skeleton behind a feature flag (`onnx-import`).
3. Add a plugin registry contract for custom op import validation.
4. Add acceptance tests for linear import, MLP import, and import/runtime parity.

## Non-goals (Wave 1)

- Full ONNX protobuf parsing.
- Dynamic-shape control flow import.
- Vendor-specific fused op lowering.

## Contract Design

### Versioning

- `IrContractVersion::V1` is immutable after release.
- Future versions must be additive or use explicit migration steps.

### Graph constraints

- Graph must contain exactly one `Output` node.
- Node IDs must be unique and referentially valid.
- Input and parameter specs must have non-empty shapes.
- All op inputs must resolve to declared node IDs.

### Supported ops in V1

- `Input`, `Parameter`, `ConstTensor`
- `Add`, `Sub`, `Mul`, `Div`
- `MatMul`, `Relu`, `Softmax`
- `Output`

### Determinism policy

- Import must not inject implicit fallback behavior.
- Runtime determinism remains governed by existing backend and policy gates.
- Unsupported ops should fail loudly, not degrade silently.

## Plugin API

`PluginRegistry` and `OpImportPlugin` provide extension points for validating imported nodes and supporting custom op namespaces without changing core importer logic.

## Rollout Plan

1. Land `interop` module with `contract`, `plugin`, and feature-gated `onnx` importer.
2. Run acceptance tests in CI with and without `onnx-import`.
3. In Wave 2, connect real ONNX parsing to the same contract and registry.

## Risks and Mitigations

- Risk: Contract drift from internal IR.
  - Mitigation: Contract compile path targets existing `Graph`/`Op` directly and is covered by tests.
- Risk: Plugin misuse introducing invalid nodes.
  - Mitigation: Contract validation executes before graph compilation.
- Risk: Feature-gated code rot.
  - Mitigation: Add feature-enabled acceptance tests in CI lane for interop.
