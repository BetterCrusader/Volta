# Volta: Deterministic-First Compiler + ML Runtime (Experimental)

Most ML stacks optimize for velocity first and explain behavior later.
Volta does the opposite: **same inputs, same graph, same policy, same result** is the contract.

## Read This First (Honest Status)

- Volta is **experimental** and still early-stage.
- The current hardening wave in this repo was built rapidly in about **4 days** (2026-02-20 to 2026-02-23), then locked behind strict quality gates.
- This is **not** a full replacement for PyTorch. It is a deterministic-first engine for a narrower, governed scope.

If you need broad ecosystem coverage today, use PyTorch.
If you need hard replayability and explicit failure semantics for supported paths, Volta is where we are pushing harder.

## Why This Exists

When reproducibility is optional, production debugging becomes expensive guesswork.
Volta makes reproducibility a first-class runtime property:

- verifier-first graph discipline
- deterministic scheduling and allocation
- explicit unsupported-path failures (no silent fallback)
- policy-driven release gates

## Where Volta Is Better Than PyTorch (Today, Narrow Scope)

This comparison is intentionally scoped and factual.

- **Determinism as a product contract:** replay discipline is designed into scheduler/allocation/runtime behavior, not treated as best effort.
- **No silent fallback policy:** unsupported paths are expected to fail loudly, not quietly switch execution semantics.
- **Governance-backed verification lanes:** Quality Fortress gates enforce determinism and contract alignment in CI.

## Where Volta Is Not Better (Yet)

- PyTorch has vastly larger model/operator ecosystem and tooling.
- Volta ONNX import is a constrained Wave 1/2 static subset, not full ONNX breadth.
- Some paths are intentionally explicit stubs/not-implemented (for example selected backward/operator combinations), by design, to avoid fake readiness.

## What Works Right Now

- Parse and execute Volta DSL programs
- Compile to strict SSA IR
- Run deterministic CPU and CUDA validation lanes
- Import ONNX Wave 1 and Wave 2 static contracts under governance
- Train/infer through guarded runtime gateways with explicit policy checks

## Current Limits (Explicit, No Marketing Spin)

- ONNX scope is partial and static by design.
- Some autograd and CUDA paths are explicit fail-fast non-support where implementation is incomplete.
- This repo prioritizes correctness and determinism over broad operator count.

Details are tracked in `docs/ONNX_COVERAGE.md` and governance docs.

## Quick Start

```bash
cargo run -- init quickstart
cd quickstart
cargo run -- run model.vt
```

Main CLI commands:

```bash
volta run <file.vt> [--quiet]
volta check <file.vt> [--quiet]
volta info <file.vt>
volta doctor [--json] [--strict]
volta init [project_dir]
volta version
volta help
```

Command intent:

- `run`: parse + semantic + execute
- `check`: parse + semantic validation (no execution)
- `info`: structural summary (statements and topology)
- `doctor`: environment and determinism readiness report
- `init`: scaffold a Volta project quickly

## Architecture In One View

```text
Source / Model API
    -> Frontend (lexer, parser, semantic)
    -> SSA IR
    -> Verifier + Shape Inference
    -> Scheduler
    -> Allocation
    -> ExecutionPlan
    -> Backend / Runtime Execution
```

Training path:

```text
Forward Graph
    -> Reverse-mode autograd builds separate Backward Graph
    -> Build ExecutionPlan for forward and backward
    -> Execute by schedule
    -> Apply optimizer in runtime layer (SGD/Adam)
```

## Language Snapshot

```vt
lr 0.001

model brain
    layers 784 256 128 10
    activation relu
    optimizer adam lr

dataset mnist
    batch 32
    shuffle true

train brain on mnist
    epochs 3
    device auto

print "training complete"
```

## Build and Verification

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo check --manifest-path fuzz/Cargo.toml
```

## Quality Fortress Gates

### Quality Fortress Wave 1 checks

```bash
bash scripts/ci/wave1_local_verify.sh
python scripts/ci/detect_tiers.py --paths src/ir/tensor.rs
python scripts/ci/policy_check.py --paths src/ir/tensor.rs --pr-body "RFC-004"
```

### Quality Fortress Wave 2/3 checks

```bash
bash scripts/ci/wave23_local_verify.sh
bash scripts/ci/release_perf_double_pass.sh
bash scripts/ci/nightly_perf_matrix.sh
```

### tiny-transformer CPU milestone

```bash
bash scripts/ci/tiny_transformer_cpu_verify.sh
```

Includes a **deterministic replay gate** and memory planner guard.

### CUDA inference MVP

```bash
bash scripts/ci/cuda_infer_verify.sh
```

This is the authoritative verify lane for **CUDA inference MVP** behavior.

### CUDA training hardening

```bash
bash scripts/ci/cuda_train_verify.sh
```

### XL release discipline

```bash
bash scripts/ci/xl_verify.sh
```

## Governance and Contracts

Core governance docs live in `docs/governance/`.

Start here:

- `docs/governance/contracts-tier-a.md`
- `docs/governance/determinism-scope.md`
- `docs/governance/cuda-determinism-policy.md`
- `docs/governance/perf-governance.md`
- `docs/governance/ci-topology.md`
- `docs/governance/QUALITY_POLICY.md`

## Contributor Onboarding

- Contribution process: `CONTRIBUTING.md`
- Windows-native release flow: `scripts/release/cut_v1.ps1`
- Bash release flow: `scripts/release/cut_v1.sh`

## Positioning

Volta is not trying to win by feature count right now.
Volta is trying to win where determinism, replayability, and explicit contracts are non-negotiable.
