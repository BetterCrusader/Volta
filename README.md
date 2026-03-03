# Volta

**Volta** is an experimental, deterministic-first Machine Learning compiler and runtime written in Rust.

Most modern ML frameworks optimize primarily for developer velocity and flexibility. PyTorch, TensorFlow, JAX — they all give you incredible freedom, but they sacrifice the most important thing: **if you run the same code twice, you won't get the same result**. Floating-point arithmetic, operation ordering, parallelism, random seeds — all of these affect the output, and often the smallest changes lead to completely different numbers.

Volta is an attempt to solve this problem. Here, **determinism** is not a wish, not an option — it's a **contract**. Same inputs, same model, same parameters — you get **exactly the same result** down to the last bit.

---

## What's Inside

Volta is not just a library. It's a full-blown stack with its own components, each written from scratch:

- **Lexer & Parser** — a custom language for describing models and datasets (`.vt` files). No Python, no YAML, no JSON. Simple, concise syntax that's easy to read and parse.
- **IR (Intermediate Representation)** — a custom intermediate representation of the computation graph. The graph is built explicitly, each operation is a separate node that can be analyzed, optimized, or transformed.
- **Graph Optimizations** — a bunch of classic optimizations: constant folding, dead code elimination, common subexpression elimination, algebraic simplification. But most importantly — all of this is done in a way that doesn't break determinism.
- **Autograd** — a custom automatic differentiation system. It builds the backward graph separately from the forward graph, without mutating the original graph.
- **Scheduler** — an execution planner that builds a deterministic sequence of operations. No random orders, no "guesses" about execution time — everything is known in advance.
- **Trainer** — a full training loop with support for SGD and Adam optimizers, validation, logging, and gradient checkpointing.

---

## Backends

### CPU Backend

Fully deterministic sequential execution. Each operation executes strictly in order, no parallelism, no race conditions. The result can be predicted with mathematical precision.

Multi-threading can be optionally enabled via `rayon`, but then **bit-level determinism may be broken** due to different floating-point operation orders. This option is therefore marked as `unsafe` at the deterministic contract level.

### CUDA Backend

GPU acceleration via `cudarc`. This is much more complex because CUDA is inherently non-deterministic. But I've done everything possible:

- Special custom kernels that minimize non-determinism.
- **Strict mode** — if an operation cannot be guaranteed to be deterministic, Volta will simply refuse to execute it, instead of "quietly" using a non-deterministic backend.
- **Determinism policy** — you can choose the level of determinism: from "allow everything" to "only verified operations".

---

## The Language (.vt files)

Volta has its own language for describing models. It's simple, readable, and most importantly — combines model description, data, and training in one place:

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
```

No classes, no Python functions, no OOP. Just declarative description: here's the model, here's the data, here are the training parameters.

Supported:
- Linear/Dense layers
- Activation functions: ReLU, Sigmoid, Softmax, GELU
- Optimizers: SGD, Adam
- CSV datasets (with train/val split support)
- Save/Load models in custom binary format
- Inference with CSV export
- Custom functions
- Conditional constructs (if/elif/else)
- Loops

---

## Testing

Volta has over **250 tests** that verify:

- Parsing and language semantics
- Graph optimizations (whether they preserve semantics)
- Autograd correctness (gradcheck — are gradients computed correctly)
- Determinism (run 100 times — get identical result)
- CPU vs CUDA parity (does GPU give the same result as CPU)
- Edge cases and errors (invalid input data, incompatible tensor shapes, division by zero, etc.)

---

## Why Rust

Rust gives several advantages that are critically important for deterministic ML:

1. **Memory safety without garbage collector** — no pause-the-world moments, no unpredictable delays. Memory is controlled explicitly.

2. **Zero-cost abstractions** — abstractions don't cost at runtime. You can write high-level code that compiles to efficient machine code.

3. **Thread safety** — the borrow checker guarantees no data races. For deterministic parallelism, this is critical.

4. **No hidden state** — Rust has no "magical" global states, no implicit caches, no undefined behavior.

---

## Limitations and Disclaimers

Volta is an **experimental project**. It's not PyTorch, not TensorFlow, and doesn't intend to become one.

**What works:**
- Small models (Dense, shallow neural networks)
- CPU and CUDA backends
- Basic graph optimization
- Deterministic training and inference

**What DOESN'T work or needs work:**
- Convolutional layers (partial support)
- RNN/LSTM/Transformer architectures (partial support via tiny transformer example)
- ONNX import/export (partial, experimental)
- Distributed computing
- Model zoo and pretrained weights

**Main disclaimer:** I work on this project in my spare time. There may be bugs, unfinished parts, and things that simply aren't implemented yet. This is not production-ready software. It's a technological experiment that shows that deterministic ML is possible, and demonstrates what it might look like.

---

## How to Build and Run

```bash
# Clone
git clone https://github.com/BetterCrusader/Volta.git
cd Volta

# Check if it compiles
cargo check

# Run tests (CPU)
cargo test --workspace

# If you have NVIDIA GPU and CUDA Toolkit — run with CUDA
cargo test --workspace --features cuda

# Build CLI
cargo build --release

# Run example
volta run examples/xor.vt
```

---

## CLI Commands

```
volta run <file.vt>          # Execute script
volta check <file.vt>         # Check syntax without execution
volta info <file.vt>          # Show model info
volta doctor                  # Environment diagnostics
volta init [dir]             # Initialize new project
```

---

## Project Structure

```
src/
├── lexer.rs          # .vt language lexer
├── parser.rs        # Parser (AST)
├── semantic.rs      # Semantic analyzer
├── executor.rs      # Runtime execution
├── ir/              # Intermediate Representation
│   ├── graph.rs     # Computation graph
│   ├── tensor.rs    # Tensor mathematics
│   ├── autograd.rs  # Automatic gradients
│   ├── optimizer.rs # Optimizers (SGD, Adam)
│   ├── train.rs    # Training loop
│   ├── scheduler.rs # Scheduler
│   └── cuda/        # CUDA backend
└── main.rs          # CLI
```

---

## License

MIT. Do whatever you want with it.

---

## Contact

Want to get in touch? GitHub issues, pull requests, or just fork and change it for yourself. This is open source, and I'll be happy if it's useful to someone else.

Or if you're into ML and want to help with development — welcome, let's discuss.

---

**P.S.** If you read this far — thank you. This project was made with soul, and I hope that the idea of deterministic ML is interesting to someone else besides me.
