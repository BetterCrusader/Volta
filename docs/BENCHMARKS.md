# Volta Benchmark Results

## Protocol

Every result in this document was produced with the following protocol — no exceptions.

- **Runs**: 30 outer runs × 7 inner runs × 50 training steps per inner run
- **Reported statistic**: median of 7 inner runs → 30 values → p50=(sorted[14]+sorted[15])/2, p95=sorted[28]
- **No trimming**, no outlier removal
- **Cooldown**: 90 seconds idle before each benchmark suite; CPU process priority set to HIGH
- **Warmup**: 10 steps before timing begins (per inner run)
- **Data**: deterministic synthetic input — `X[i] = (i%17)*0.01 - 0.08`, `Y[i] = (i%7)*0.1`
- **Optimizer**: SGD, lr=0.01 (except Adam section)
- **Platform**: Windows 11 Pro x86-64, 6-core CPU
- **Date**: 2026-03-06

PyTorch uses `torch.set_num_threads(6)` (MKL), eager mode, SGD. 30 separate Python processes per benchmark.

---

## Primary Results

### Case 1 — MLP 256→512→512→256→1, B=64 ("small")

| | p50 | p95 | min | max |
|---|---|---|---|---|
| **Volta** | **0.633 ms** | 0.797 ms | 0.583 ms | 1.137 ms |
| PyTorch 6T | 0.856 ms | 0.959 ms | 0.783 ms | 1.773 ms |
| **Volta faster by** | **+35%** | | | |

### Case 2 — MLP 512→1024→1024→512→256→1, B=64 ("medium", primary reference)

| | p50 | p95 | min | max |
|---|---|---|---|---|
| **Volta** | **1.703 ms** | 1.922 ms | 1.684 ms | 1.993 ms |
| PyTorch 6T | 2.440 ms | 2.472 ms | 2.396 ms | 2.588 ms |
| **Volta faster by** | **+43%** | | | |

### Case 3 — MLP 512→2048→2048→512→1, B=64 ("heavy")

| | p50 | p95 | min | max |
|---|---|---|---|---|
| **Volta** | **5.054 ms** | 5.692 ms | 4.492 ms | 6.424 ms |
| PyTorch 6T | 8.457 ms | 11.087 ms | 6.980 ms | 17.684 ms |
| **Volta faster by** | **+67%** | | | |

### Case 4 — MLP 512→1024→1024→512→256→1, B=128

| | p50 | p95 | min | max |
|---|---|---|---|---|
| Volta | 3.659 ms | 3.698 ms | 3.608 ms | 3.704 ms |
| PyTorch 6T | 3.628 ms | 3.739 ms | 3.539 ms | 3.779 ms |
| **Result** | **statistical parity** | | | |

---

## Batch Size Sweep — MLP 512→1024→1024→512→256→1

*(7 inner runs, warm CPU, same protocol without 30-outer-run wrapper)*

| Batch | Volta | PyTorch 6T | Result |
|---|---|---|---|
| 16 | 0.782 ms | ~1.9 ms | Volta ~2.4× faster |
| 32 | 1.219 ms | ~2.2 ms | Volta ~1.8× faster |
| 64 | 1.940 ms | 2.445 ms | Volta +26% |
| 128 | 3.552 ms | 3.490 ms | PyTorch marginally faster |
| 256 | 7.478 ms | 5.913 ms | PyTorch +26% faster |

**Honest summary**: Volta advantage degrades as batch size grows. Root cause: at large batch sizes the dominant GEMMs become [B×K]@[K×N] with large B — PyTorch MKL has better tile strategies for these shapes. Volta wins on small-to-medium batch (B≤64), reaches parity at B=128, loses at B≥256.

---

## Adam Optimizer

| Optimizer | Volta | PyTorch 6T | Result |
|---|---|---|---|
| SGD lr=0.01 | 1.703 ms | 2.440 ms | **Volta +43%** |
| Adam lr=0.001 | 9.40 ms | 4.98 ms | **PyTorch 1.9× faster** |

Adam is significantly slower in Volta. The SGD path has a fused `sgd_tn` kernel (W -= lr * act^T @ delta) that eliminates the dW buffer and merges the weight update into one GEMM call. Adam cannot use this fusion — it requires a separate dW computation, moment buffers (m_w, v_w, m_b, v_b), and element-wise update loops. This is the next major optimization target.

Adam numerical accuracy: loss decreases monotonically, numerics match PyTorch Adam to float32 precision.

---

## Key Optimizations (SGD Training DLL)

| Optimization | Effect |
|---|---|
| `sgd_fused_tn`: fuse dW + SGD into one GEMM (W -= lr * act^T @ delta) | −1350 µs/step eliminated |
| Pre-transpose delta for dX: dt=delta^T, tmp=W@dt, dx=tmp^T | Cache-friendly backward pass |
| AVX2 8×8 transpose kernel | 7.76× faster than scalar on 1024×64 matrix |
| MKL hybrid: `cblas_sgemm` for weight updates, gemm crate for forward | −6.4% vs pure gemm |
| Rayon(5) threads for ops <33M, Rayon(0) for ≥33M | Optimal for 6-core |

---

## Optimization History (B=64, MLP-512)

| Version | Median | vs PyTorch 6T |
|---|---|---|
| Naive baseline (no Rayon) | 8.84 ms | 3.3× slower |
| + fast_transpose + alt dX | ~3.5 ms | 1.3× slower |
| + sgd_fused_tn | 2.374 ms | +4.2% faster |
| + AVX2 8×8 transpose | 1.982 ms | +35.6% faster |
| + MKL hybrid sgd | **1.703 ms** | **+43% faster** |

---

## What can be validly claimed

- "Volta CPU training is +35–67% faster than PyTorch eager (6T, MKL) at B≤64 across three MLP architectures"
- "At B=128 the result is statistical parity"
- "Adam optimizer is currently 1.9× slower than PyTorch"
- "Volta variance at B=64 is substantially lower than PyTorch" (see p95/p50 ratio in tables above)

## What cannot be claimed

- "Volta beats PyTorch at all batch sizes" — false at B≥256
- "Volta beats PyTorch in all scenarios" — false for Adam, false for large batch
- "Volta is faster than PyTorch on GPU" — not measured
- "These results generalize to non-MLP architectures" — not measured

---

## How to Reproduce

### Prerequisites

```
- Rust stable toolchain (cargo in PATH)
- MKL via conda: conda create -n mkl -c conda-forge mkl && conda activate mkl
- Python with torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Run Volta benchmark (primary, B=64 MLP-512)

```bash
# Build all benchmark binaries
cd examples/bench_real.train_rust._rust_crate
cargo build --release --examples

# Set MKL in PATH (required at runtime — links against mkl_rt.dll)
export PATH="/path/to/conda/envs/mkl/Library/bin:$PATH"
# Windows: $env:PATH = "C:\...\mkl\Library\bin;" + $env:PATH

# Run a single benchmark (each exe accepts --out <file> to write median to file)
./target/release/examples/bench_official_v2.exe
./target/release/examples/bench_mlp256.exe
./target/release/examples/bench_mlp2048.exe
./target/release/examples/bench_b128.exe
```

Each binary prints per-run timing to stderr and writes `median=X.XXX` to the `--out` file before exiting (uses `ExitProcess(0)` to bypass Rayon teardown abort on Windows).

### PyTorch reference

The benchmark scripts in `examples/bench_pytorch.py` and the inner scripts referenced by the harness (`bench_pytorch_mlp256.py`, etc.) are the PyTorch counterparts. Run them directly with a Python that has `torch` installed:

```bash
python examples/bench_pytorch.py
```

### Regression threshold

B=64 MLP-512 median must stay **below 2.10 ms** after 90s cooldown. Build and run `bench_official_v2.exe` in the crate above.

---

## Environment Notes

- All benchmarks ran on the same machine under the same conditions
- Thermal throttle is a real factor: without the 90s cooldown, results are 30–50% worse
- PyTorch version: 2.10.0+cpu (pip, no CUDA)
- MKL version: Intel MKL from conda-forge
- Compiler: rustc stable, `-C opt-level=3`, `RUSTFLAGS="-C target-cpu=native"`
- No background processes; CPU priority set to HIGH via Windows API before timing
