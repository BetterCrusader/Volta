# Volta

Volta is a deterministic AI compiler core written in Rust.

Volta is an experimental compiler-first ML system focused on determinism and IR discipline rather than eager ergonomics.

Project status:

- stable core contract for `v0.1.0-core`
- governance hardening in progress under **Quality Fortress**
- Wave 1 focuses on enforceable quality policy, not feature expansion

This repository focuses on compiler discipline first:

- strict SSA IR
- strict typing and shape checks
- deterministic schedule and allocation
- verifier-guarded transformations
- repeatable training pipeline through execution plans

Volta is currently at **v0.1.0-core**: experimental, but with a stable core contract.

## Why Volta?

Most ML stacks are eager-first and optimize compiler behavior later.
Volta starts from the opposite constraint: compiler correctness and determinism first.

- explicit execution plans over implicit graph walking
- strict verifier guards over permissive runtime assumptions
- deterministic scheduling/allocation as a core design requirement

---

## 1. What Volta Is

Volta is not positioned as "yet another eager ML framework".
It is a compiler-first system where model logic is represented as verified IR and executed via explicit plans.

High-level flow:

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

Training flow:

```text
Forward Graph
    -> Reverse-mode autograd builds separate Backward Graph
    -> Build ExecutionPlan for forward and backward
    -> Execute by schedule
    -> Apply optimizer in runtime layer (SGD/Adam)
```

---

## 2. Project Structure

```text
src/
  ast.rs                 # AST and Span model
  lexer.rs               # tokenizer with indentation tokens
  parser.rs              # parser -> AST
  semantic.rs            # semantic checks and warnings
  executor.rs            # DSL runtime executor
  autopilot.rs           # train config resolution
  rules.rs               # shared language/policy constants

  ir/
    op.rs                # IR operation set
    graph.rs             # Graph/Node/ValueId/Block
    lowering.rs          # AST -> IR lowering
    verifier.rs          # hard IR contract checks
    shape_inference.rs   # static shape facts
    scheduler.rs         # deterministic topological schedule
    allocation.rs        # storage classes + buffer assignment
    memory_planner.rs    # liveness/peak memory analysis
    execution_plan.rs    # Schedule + Allocation composition
    interpreter.rs       # strict IR interpreter
    autograd.rs          # reverse-mode backward graph builder
    optimizer.rs         # SGD/Adam runtime updates
    train.rs             # schedule-based train loop
    ...                  # passes, fusion, backend abstraction, stress harness

  model/
    mod.rs               # model authoring exports
    builder.rs           # deterministic model -> IR builder
    module_trait.rs      # Module trait
    layers.rs            # Sequential/Linear/Conv2D/ReLU/Softmax
    losses.rs            # MSELoss/CrossEntropyLoss
    dataset.rs           # Dataset trait + deterministic batching/shuffle
    train_api.rs         # high-level train API
    checkpoint.rs        # save/load parameters
```

---

## 3. Language Basics (DSL)

Volta DSL is indentation-based (spaces only, tabs are rejected).

### 3.1 Core statements

- variable declaration / assignment
- `model` / `dataset` / `train`
- `save` / `load`
- `print`
- `fn`, `return`, `loop`, `if` / `elif` / `else`

### 3.2 Example

```vt
lr 0.001

model brain
    layers 784 256 128 10
    activation relu
    optimizer adam lr
    precision auto

dataset mnist
    batch 32
    shuffle true

train brain on mnist
    epochs 10
    device auto

accuracy 0.92

if accuracy > 0.95
    save brain as "best.vt"
    print "done"
elif accuracy > 0.8
    print "almost"
else
    print "keep training"
```

Notes:

- example extension above is `.vt` (project convention)
- language does not allow implicit type coercion magic
- diagnostics carry location/span info

---

## 4. IR Contract (Critical)

Volta IR is single-output SSA: one node -> one `ValueId`.

Hard constraints:

- no use-before-def
- no duplicate producer for same `ValueId`
- no `Removed` node usage
- strict typing consistency
- strict shape consistency
- deterministic schedule validity
- allocation/storage safety checks

Verifier is a required guard on core paths.

---

## 5. Determinism Policy

Determinism is an explicit release criterion.

Covered by tests and harness:

- schedule hash stability
- allocation signature stability
- fingerprint stability under verified pass pipelines
- stress and fuzz checks for invariants

Compiler flags (`src/ir/compiler_flags.rs`):

- `strict`
- `debug-verify`
- `unsafe-opt` (experimental gate)

---

## 6. High-level Model API

`src/model/` provides a deterministic authoring layer above IR.

Main concepts:

- `TensorShape`
- `Parameter`
- `Module` trait
- `Sequential`
- layers: `LinearLayer`, `Conv2DLayer`, `ReLULayer`, `SoftmaxLayer`
- losses: `MSELoss`, `CrossEntropyLoss`
- dataset/batch API
- checkpoint save/load

Model build performs shape validation and emits IR only.
No eager execution shortcuts.

---

## 7. Build, Test, Run

From repository root:

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
```

### Quality Fortress Wave 1 checks

```bash
bash scripts/ci/wave1_local_verify.sh
python -m unittest scripts.ci.tests.test_detect_tiers -v
python -m unittest scripts.ci.tests.test_policy_check -v
```

Governance docs live under `docs/governance/` and are test-verified.

### Quality Fortress Wave 2/3 checks

```bash
bash scripts/ci/wave23_local_verify.sh
bash scripts/ci/release_perf_double_pass.sh
bash scripts/ci/nightly_perf_matrix.sh
```

Release/nightly workflows are defined in:

- `.github/workflows/release-gates.yml`
- `.github/workflows/nightly-quality.yml`

Run demo binary:

```bash
cargo run
```

Run selected heavy freeze tests:

```bash
cargo test --release ir::freeze_hardening::tests::fuzz_ssa_graphs_5000_heavy -- --ignored --nocapture
cargo test --release ir::freeze_hardening::tests::long_run_determinism_100x50_heavy -- --ignored --nocapture
```

---

## 8. What Exists vs Not Yet

### Implemented

- frontend pipeline (lexer/parser/semantic/executor)
- SSA IR + lowering + verifier
- shape inference
- scheduler + allocation + execution plan
- autograd (separate backward graph)
- optimizer runtime (SGD, Adam)
- deterministic train path through execution plans
- pass framework + multiple graph optimizations
- backend abstraction (`CpuBackend`, LLVM/CUDA stubs)
- stress/freeze harness

### Not implemented yet

- full production LLVM backend
- full production CUDA backend
- advanced distributed runtime
- full kernel library and low-level codegen stack

---

## 9. Release Positioning

Recommended public positioning:

> Deterministic minimal AI compiler core with strict IR discipline.

This communicates scope correctly and avoids overclaiming.

---

## 10. Safety and Contribution Rules

- do not bypass verifier
- do not mutate forward graph during autograd
- do not move optimizer logic into IR
- do not introduce implicit cast/broadcast behavior
- keep changes deterministic and test-covered

If a change violates IR contract: stop, rollback, explain.
