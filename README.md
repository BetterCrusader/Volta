# Volta: The Deterministic AI Compiler Core

If your model gives two different answers for the same input, Volta is built for you.

Volta is a compiler-first ML runtime in Rust focused on one promise: **same inputs, same graph, same policy, same result**.

## Why This Is Different

Most stacks optimize for speed first and explain behavior later.
Volta does the opposite:

- verifier-first graph discipline
- deterministic schedule and allocation
- strict contracts before runtime execution
- policy-driven release gates

This is the engine for people who want ML systems that are inspectable, replayable, and production-safe.

## What You Can Do Right Now

- Parse and execute the Volta DSL
- Compile to strict SSA IR
- Run deterministic CPU and CUDA validation lanes
- Import ONNX Wave 1 and Wave 2 contracts under governance
- Ship with hard CI gates instead of guesswork

## Project Status

- public release line: `release-v1.0.0` (Volta V1)
- core contract lineage started at `v0.1.0-core`
- governance hardening under **Quality Fortress**
- CUDA inference MVP in Wave 3 hardening track
- CUDA training hardening in Wave 4 hardening track

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
- `init`: scaffold a Volta project in seconds

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

## Quality Fortress Gates

### Quality Fortress Wave 1 checks

```bash
bash scripts/ci/wave1_local_verify.sh
python -m unittest scripts.ci.tests.test_detect_tiers -v
python -m unittest scripts.ci.tests.test_policy_check -v
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

## Build and Test

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
```

## Contributor Onboarding

- Contribution process: `CONTRIBUTING.md`
- Windows-native release flow: `scripts/release/cut_v1.ps1`
- Bash release flow: `scripts/release/cut_v1.sh`

## Positioning

Volta is not "yet another eager framework".

It is a deterministic compiler core for teams that want correctness and replayability as first-class product features.
