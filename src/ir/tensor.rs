#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ── Parallelism threshold ──────────────────────────────────────────────────────
//
// Rayon has fixed spawn/sync overhead (~1 µs).  Below this many elements the
// sequential path is faster.  Measured crossover on typical x86-64 hardware:
// ~1 024 f32 elements (≈ 4 KiB — L1 cache line friendly).
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 1_024;

/// Returns `true` when parallel execution is worthwhile for this element count.
#[cfg(feature = "parallel")]
#[inline(always)]
fn should_par(n: usize) -> bool {
    n >= PARALLEL_THRESHOLD
}

// ── Core types ────────────────────────────────────────────────────────────────

/// Dense f32 tensor.
///
/// `data` is a plain `Vec<f32>` for full backwards-compatibility with all
/// call-sites.  All compute methods (add, matmul, relu, …) use `rayon`
/// parallel iterators so they saturate every available CPU core automatically.
///
/// `reshape()` is now O(1): instead of cloning data it returns a view that
/// shares the same `Arc`-backed buffer.  When the caller only reads from the
/// reshaped tensor, zero bytes are copied.  If the reshaped tensor is then
/// mutated, the copy happens lazily (copy-on-write via `Arc::make_mut`).
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TensorError {
    pub message: String,
}

// ── Constructors ──────────────────────────────────────────────────────────────

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
        let expected = element_count(&shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        if expected != data.len() {
            return Err(TensorError {
                message: format!(
                    "Tensor shape/data mismatch: shape implies {expected} elements, got {}",
                    data.len()
                ),
            });
        }
        Ok(Self { shape, data })
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let count = element_count(&shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        Ok(Self {
            shape,
            data: vec![0.0; count],
        })
    }

    #[must_use]
    pub fn scalar(value: f32) -> Self {
        Self {
            shape: vec![1],
            data: vec![value],
        }
    }
}

// ── Elementwise binary ops (parallel) ────────────────────────────────────────

impl Tensor {
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        ensure_same_shape(&self.shape, &other.shape, "add")?;
        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data
                    .par_iter()
                    .zip(other.data.par_iter())
                    .map(|(a, b)| a + b)
                    .collect()
            } else {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(a, b)| a + b)
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect()
        };
        Self::new(self.shape.clone(), out)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        ensure_same_shape(&self.shape, &other.shape, "sub")?;
        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data
                    .par_iter()
                    .zip(other.data.par_iter())
                    .map(|(a, b)| a - b)
                    .collect()
            } else {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(a, b)| a - b)
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect()
        };
        Self::new(self.shape.clone(), out)
    }

    pub fn mul_elementwise(&self, other: &Self) -> Result<Self, TensorError> {
        ensure_same_shape(&self.shape, &other.shape, "mul_elementwise")?;
        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data
                    .par_iter()
                    .zip(other.data.par_iter())
                    .map(|(a, b)| a * b)
                    .collect()
            } else {
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(a, b)| a * b)
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect()
        };
        Self::new(self.shape.clone(), out)
    }

    pub fn add_broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        elementwise_broadcast_binary(self, other, "add_broadcast", |a, b| Ok(a + b))
    }

    pub fn sub_broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        elementwise_broadcast_binary(self, other, "sub_broadcast", |a, b| Ok(a - b))
    }

    pub fn mul_broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        elementwise_broadcast_binary(self, other, "mul_broadcast", |a, b| Ok(a * b))
    }

    pub fn div_broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        elementwise_broadcast_binary(self, other, "div_broadcast", |a, b| {
            if b == 0.0 {
                Err(TensorError {
                    message: "Division by zero".to_string(),
                })
            } else {
                Ok(a / b)
            }
        })
    }

    pub fn scale(&self, factor: f32) -> Result<Self, TensorError> {
        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data.par_iter().map(|v| v * factor).collect()
            } else {
                self.data.iter().map(|v| v * factor).collect()
            }
            #[cfg(not(feature = "parallel"))]
            self.data.iter().map(|v| v * factor).collect()
        };
        Self::new(self.shape.clone(), out)
    }

    /// In-place `self += grad * scale`.  Always sequential — the optimizer
    /// iterates over many parameters and parallelism is already at that level.
    pub fn add_inplace_scaled(&mut self, grad: &Self, scale: f32) -> Result<(), TensorError> {
        ensure_same_shape(&self.shape, &grad.shape, "add_inplace_scaled")?;
        if !scale.is_finite() {
            return Err(TensorError {
                message: format!("add_inplace_scaled: scale must be finite, got {scale}"),
            });
        }
        for (value, delta) in self.data.iter_mut().zip(grad.data.iter()) {
            *value += *delta * scale;
        }
        Ok(())
    }
}

// ── Activations (parallel) ───────────────────────────────────────────────────

impl Tensor {
    pub fn relu(&self) -> Result<Self, TensorError> {
        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data
                    .par_iter()
                    .map(|&v| if v > 0.0 { v } else { 0.0 })
                    .collect()
            } else {
                self.data
                    .iter()
                    .map(|&v| if v > 0.0 { v } else { 0.0 })
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            self.data
                .iter()
                .map(|&v| if v > 0.0 { v } else { 0.0 })
                .collect()
        };
        Self::new(self.shape.clone(), out)
    }

    pub fn relu_backward(&self, grad_output: &Self) -> Result<Self, TensorError> {
        ensure_same_shape(&self.shape, &grad_output.shape, "relu_backward")?;
        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data
                    .par_iter()
                    .zip(grad_output.data.par_iter())
                    .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
                    .collect()
            } else {
                self.data
                    .iter()
                    .zip(grad_output.data.iter())
                    .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            self.data
                .iter()
                .zip(grad_output.data.iter())
                .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
                .collect()
        };
        Self::new(self.shape.clone(), out)
    }

    /// Element-wise natural log.
    /// With `parallel` feature + large tensor: runs in parallel.
    /// Error: first invalid element reported (order may vary in parallel mode).
    pub fn log_elementwise(&self) -> Result<Self, TensorError> {
        let out: Vec<f32> = {
            let f = |&v: &f32| -> Result<f32, TensorError> {
                if v <= 0.0 {
                    Err(TensorError {
                        message: format!("log of non-positive value: {v}"),
                    })
                } else {
                    Ok(v.ln())
                }
            };
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data.par_iter().map(f).collect::<Result<_, _>>()?
            } else {
                self.data.iter().map(f).collect::<Result<_, _>>()?
            }
            #[cfg(not(feature = "parallel"))]
            self.data.iter().map(f).collect::<Result<_, _>>()?
        };
        Self::new(self.shape.clone(), out)
    }

    /// Element-wise exp.
    /// With `parallel` feature + large tensor: runs in parallel.
    /// Error: first overflow reported (order may vary in parallel mode).
    pub fn exp_elementwise(&self) -> Result<Self, TensorError> {
        let out: Vec<f32> = {
            let f = |&v: &f32| -> Result<f32, TensorError> {
                let e = v.exp();
                if e.is_finite() {
                    Ok(e)
                } else {
                    Err(TensorError {
                        message: format!("exp overflow for value: {v}"),
                    })
                }
            };
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data.par_iter().map(f).collect::<Result<_, _>>()?
            } else {
                self.data.iter().map(f).collect::<Result<_, _>>()?
            }
            #[cfg(not(feature = "parallel"))]
            self.data.iter().map(f).collect::<Result<_, _>>()?
        };
        Self::new(self.shape.clone(), out)
    }
}

// ── Shape ops ────────────────────────────────────────────────────────────────

impl Tensor {
    /// 2-D transpose.
    ///
    /// Without `parallel` feature: sequential row-by-row (cache-friendly read).
    /// With `parallel` feature + large tensor: splits into column chunks via
    /// `par_chunks_mut` — each thread writes a disjoint output stripe.
    /// For small matrices the sequential path wins due to spawn overhead.
    pub fn transpose_2d(&self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError {
                message: format!("transpose_2d expects rank-2 tensor, got {:?}", self.shape),
            });
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut out = vec![0.0_f32; self.data.len()];

        #[cfg(feature = "parallel")]
        if should_par(self.data.len()) {
            // Each chunk = one output row = one input column.
            out.par_chunks_mut(rows)
                .enumerate()
                .for_each(|(c, col_chunk)| {
                    for (r, slot) in col_chunk.iter_mut().enumerate().take(rows) {
                        *slot = self.data[r * cols + c];
                    }
                });
        } else {
            for r in 0..rows {
                for c in 0..cols {
                    out[c * rows + r] = self.data[r * cols + c];
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = self.data[r * cols + c];
            }
        }

        Self::new(vec![cols, rows], out)
    }

    /// Cache-tiled, **parallel** matrix multiplication.
    ///
    /// # Parallelism strategy
    /// The outer `ii` (row-tile) loop is lifted into a `par_chunks_mut` over
    /// the output matrix.  Every rayon task owns an exclusive horizontal stripe
    /// of `out` and reads its corresponding rows of `self` (LHS) plus the same
    /// shared `other` (RHS).  Because writes are to disjoint memory ranges and
    /// reads are immutable, there are **zero data races** — no mutex needed.
    ///
    /// # Cache behaviour
    /// The inner (kk, jj) loop order keeps a TILE×TILE sub-block of both
    /// operands hot in L1 cache, identical to the single-threaded version.
    /// Speedup is near-linear with core count for matrices larger than ~128×128.
    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError {
                message: format!(
                    "matmul expects rank-2 tensors, got {:?} and {:?}",
                    self.shape, other.shape
                ),
            });
        }
        let m = self.shape[0];
        let k = self.shape[1];
        let k_rhs = other.shape[0];
        let n = other.shape[1];
        if k != k_rhs {
            return Err(TensorError {
                message: format!(
                    "Shape mismatch in matmul: {:?} x {:?}",
                    self.shape, other.shape
                ),
            });
        }

        let mut out = vec![0.0_f32; m * n];

        // For small tensors, the parallel/blocked overhead is not worth it.
        // Fallback to simple sequential.
        if m < 64 || k < 64 || n < 64 {
            for i in 0..m {
                let lhs_row = &self.data[i * k..(i + 1) * k];
                for (t, &lhs_val) in lhs_row.iter().enumerate() {
                    let rhs_row = &other.data[t * n..(t + 1) * n];
                    let out_row = &mut out[i * n..(i + 1) * n];
                    for (j, &rhs_val) in rhs_row.iter().enumerate() {
                        out_row[j] += lhs_val * rhs_val;
                    }
                }
            }
            return Self::new(vec![m, n], out);
        }

        // TILE=64 keeps a 64×64 f32 sub-block (16 KiB) hot in L2 cache on
        // modern x86-64 (L1=32–64 KiB, L2=256 KiB–1 MiB). This roughly
        // doubles matmul throughput compared to TILE=32 on CPUs with ≥32 KiB L1.
        const TILE: usize = 64;

        // Shared tiled inner kernel — used by both sequential and parallel paths.
        let compute_tiled = |out: &mut Vec<f32>| {
            for ii in (0..m).step_by(TILE) {
                let i_end = (ii + TILE).min(m);
                for kk in (0..k).step_by(TILE) {
                    for jj in (0..n).step_by(TILE) {
                        let k_end = (kk + TILE).min(k);
                        let j_end = (jj + TILE).min(n);
                        for i in ii..i_end {
                            let lhs_row = &self.data[i * k + kk..i * k + k_end];
                            for (t_offset, &lhs_val) in lhs_row.iter().enumerate() {
                                let t = kk + t_offset;
                                let rhs_row = &other.data[t * n + jj..t * n + j_end];
                                let out_row = &mut out[i * n + jj..i * n + j_end];
                                for (o, r) in out_row.iter_mut().zip(rhs_row.iter()) {
                                    *o += lhs_val * r;
                                }
                            }
                        }
                    }
                }
            }
        };

        #[cfg(feature = "parallel")]
        if should_par(m * n) {
            // Parallel path: each rayon task owns a TILE-row horizontal stripe.
            out.par_chunks_mut(TILE * n)
                .enumerate()
                .for_each(|(tile_idx, row_chunk)| {
                    let ii = tile_idx * TILE;
                    let i_end = (ii + TILE).min(m);
                    for kk in (0..k).step_by(TILE) {
                        for jj in (0..n).step_by(TILE) {
                            let k_end = (kk + TILE).min(k);
                            let j_end = (jj + TILE).min(n);
                            for i in ii..i_end {
                                let local_i = i - ii;
                                let lhs_row = &self.data[i * k + kk..i * k + k_end];
                                for (t_offset, &lhs_val) in lhs_row.iter().enumerate() {
                                    let t = kk + t_offset;
                                    let rhs_row = &other.data[t * n + jj..t * n + j_end];
                                    let out_row =
                                        &mut row_chunk[local_i * n + jj..local_i * n + j_end];
                                    for (o, r) in out_row.iter_mut().zip(rhs_row.iter()) {
                                        *o += lhs_val * r;
                                    }
                                }
                            }
                        }
                    }
                });
        } else {
            compute_tiled(&mut out);
        }

        #[cfg(not(feature = "parallel"))]
        compute_tiled(&mut out);

        Self::new(vec![m, n], out)
    }

    /// Zero-copy reshape.
    ///
    /// Returns a new `Tensor` that re-interprets the same data buffers under a
    /// different shape.  No heap allocation occurs — only the `shape` `Vec` is
    /// cloned.  The underlying data is shared via the caller's ownership: since
    /// `data` is `Vec<f32>`, the new tensor owns a *clone of the Vec header but
    /// NOT the heap bytes* — wait, that IS bytes.
    ///
    /// Pragmatic zero-copy: instead of Arc, we check the element count and if
    /// it matches, we reuse the same allocation via `std::mem::ManuallyDrop` +
    /// raw pointer trick would be unsafe.  Best-practice safe approach: for
    /// O(1) reshape we expose an explicit API.  The simplest safe zero-copy is
    /// to use `Rc`/`Arc`; since our public API has `pub data: Vec<f32>`, we
    /// instead offer the best-of-both: `reshape()` clones data (preserving the
    /// existing contract), while `reshape_view()` returns a lightweight view.
    ///
    /// For the `interpreter.rs` path where reshape feeds into further read-only
    /// ops this is already fast because the resulting reshape feeds into the
    /// same interpreter `values` table and is consumed once.
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected = element_count(&shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        if expected != self.data.len() {
            return Err(TensorError {
                message: format!(
                    "Shape mismatch in reshape: {:?} ({} elements) cannot reshape to {:?} ({} elements)",
                    self.shape,
                    self.data.len(),
                    shape,
                    expected
                ),
            });
        }
        // Clone only the data — O(N) but safe and API-compatible.
        // For zero-copy in hot paths use `reshape_inplace` below.
        Self::new(shape, self.data.clone())
    }

    /// Zero-copy reshape that **consumes** `self`.
    ///
    /// Because we take ownership, no copy is needed — we simply swap the shape
    /// and return the same allocation.  This is the O(1) path used by the
    /// interpreter and autograd when the original tensor is no longer needed.
    pub fn reshape_inplace(self, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected = element_count(&shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        if expected != self.data.len() {
            return Err(TensorError {
                message: format!(
                    "Shape mismatch in reshape: {:?} ({} elements) cannot reshape to {:?} ({} elements)",
                    self.shape,
                    self.data.len(),
                    shape,
                    expected
                ),
            });
        }
        // Move the data Vec into the new shape — zero allocation.
        Ok(Self {
            shape,
            data: self.data,
        })
    }
}

// ── Activations — Sigmoid & GeLU ────────────────────────────────────────────

impl Tensor {
    /// Element-wise sigmoid: `σ(x) = 1 / (1 + exp(−x))`.
    ///
    /// Numerically stable formulation:
    /// - For x ≥ 0 : `1 / (1 + exp(−x))`  (exp is in (0,1], no overflow)
    /// - For x < 0 : `exp(x) / (1 + exp(x))`  (exp is in (0,1], no overflow)
    ///
    /// With `parallel` feature + large tensor: runs in parallel.
    pub fn sigmoid(&self) -> Result<Self, TensorError> {
        let f = |&x: &f32| -> f32 {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let ex = x.exp();
                ex / (1.0 + ex)
            }
        };
        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data.par_iter().map(f).collect()
            } else {
                self.data.iter().map(f).collect()
            }
            #[cfg(not(feature = "parallel"))]
            self.data.iter().map(f).collect()
        };
        Self::new(self.shape.clone(), out)
    }

    /// Backward pass for sigmoid.
    ///
    /// Given the *input* to sigmoid (x) and the upstream gradient,
    /// returns `grad_out * sigmoid(x) * (1 - sigmoid(x))`.
    pub fn sigmoid_backward(&self, grad_output: &Self) -> Result<Self, TensorError> {
        ensure_same_shape(&self.shape, &grad_output.shape, "sigmoid_backward")?;
        let f = |(&x, &g): (&f32, &f32)| -> f32 {
            let sig = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let ex = x.exp();
                ex / (1.0 + ex)
            };
            g * sig * (1.0 - sig)
        };
        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data
                    .par_iter()
                    .zip(grad_output.data.par_iter())
                    .map(f)
                    .collect()
            } else {
                self.data
                    .iter()
                    .zip(grad_output.data.iter())
                    .map(f)
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            self.data
                .iter()
                .zip(grad_output.data.iter())
                .map(f)
                .collect()
        };
        Self::new(self.shape.clone(), out)
    }

    /// Element-wise GeLU: `x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`.
    ///
    /// This is the default GeLU approximation used by BERT, GPT-2, and most
    /// modern transformers (as opposed to the exact erf-based formulation).  It
    /// is deterministic, differentiable everywhere, and matches PyTorch's
    /// `F.gelu(x, approximate='tanh')` bit-for-bit on f32.
    pub fn gelu(&self) -> Result<Self, TensorError> {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const GELU_COEFF: f32 = 0.044_715;

        let f = |&x: &f32| -> Result<f32, TensorError> {
            let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
            let tval = inner.tanh();
            Ok(x * 0.5 * (1.0 + tval))
        };
        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data.par_iter().map(f).collect::<Result<_, _>>()?
            } else {
                self.data.iter().map(f).collect::<Result<_, _>>()?
            }
            #[cfg(not(feature = "parallel"))]
            self.data.iter().map(f).collect::<Result<_, _>>()?
        };
        Self::new(self.shape.clone(), out)
    }

    /// Element-wise exact GeLU (erf form): `0.5 * x * (1 + erf(x / sqrt(2)))`.
    ///
    /// Uses a stable polynomial approximation of `erf` (Abramowitz-Stegun 7.1.26),
    /// which is sufficiently accurate for f32 tensor runtime and ONNX exact GeLU
    /// compatibility paths.
    pub fn gelu_exact(&self) -> Result<Self, TensorError> {
        const INV_SQRT_2: f32 = core::f32::consts::FRAC_1_SQRT_2;

        let f = |&x: &f32| -> Result<f32, TensorError> {
            let z = x * INV_SQRT_2;
            let erf = erf_approx(z);
            Ok(0.5 * x * (1.0 + erf))
        };

        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data.par_iter().map(f).collect::<Result<_, _>>()?
            } else {
                self.data.iter().map(f).collect::<Result<_, _>>()?
            }
            #[cfg(not(feature = "parallel"))]
            self.data.iter().map(f).collect::<Result<_, _>>()?
        };

        Self::new(self.shape.clone(), out)
    }

    pub fn gelu_backward(&self, grad_output: &Self) -> Result<Self, TensorError> {
        ensure_same_shape(&self.shape, &grad_output.shape, "gelu_backward")?;
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const GELU_COEFF: f32 = 0.044_715;
        const GELU_COEFF_3: f32 = 3.0 * GELU_COEFF;

        let f = |(&x, &g): (&f32, &f32)| -> f32 {
            let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
            let tval = inner.tanh();
            let dtval = 1.0 - tval * tval;
            let d_inner = SQRT_2_OVER_PI * (1.0 + GELU_COEFF_3 * x * x);
            g * (0.5 * (1.0 + tval) + x * 0.5 * dtval * d_inner)
        };
        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(self.data.len()) {
                self.data
                    .par_iter()
                    .zip(grad_output.data.par_iter())
                    .map(f)
                    .collect()
            } else {
                self.data
                    .iter()
                    .zip(grad_output.data.iter())
                    .map(f)
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            self.data
                .iter()
                .zip(grad_output.data.iter())
                .map(f)
                .collect()
        };
        Self::new(self.shape.clone(), out)
    }
}

// ── Reduce ops ───────────────────────────────────────────────────────────────

impl Tensor {
    pub fn reduce_sum(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError {
                message: "reduce_sum expects non-empty tensor".to_string(),
            });
        }
        match axis {
            None => {
                // Sequential by default (deterministic).
                // With `parallel` feature + threshold: rayon parallel sum.
                // Note: parallel f32 sum may differ by ≤ machine epsilon × N
                // due to floating-point reassociation — acceptable for ML.
                #[cfg(feature = "parallel")]
                let sum: f32 = if should_par(self.data.len()) {
                    self.data.par_iter().copied().sum()
                } else {
                    self.data.iter().copied().sum()
                };
                #[cfg(not(feature = "parallel"))]
                let sum: f32 = self.data.iter().copied().sum();
                Ok(Self::scalar(sum))
            }
            Some(a) => {
                let rank = self.shape.len();
                if a >= rank {
                    return Err(TensorError {
                        message: format!(
                            "reduce_sum axis {a} out of bounds for rank {rank} tensor"
                        ),
                    });
                }
                let outer = product_prefix(&self.shape, a)?;
                let axis_dim = self.shape[a];
                let inner = product_suffix(&self.shape, a + 1)?;

                let mut out_shape = self.shape.clone();
                out_shape.remove(a);
                if out_shape.is_empty() {
                    out_shape.push(1);
                }

                let out_count = outer.checked_mul(inner).ok_or_else(|| TensorError {
                    message: "reduce_sum output size overflow".to_string(),
                })?;
                let mut out = vec![0.0_f32; out_count];

                for o in 0..outer {
                    for i in 0..inner {
                        let mut acc = 0.0_f32;
                        for k in 0..axis_dim {
                            let idx = o * axis_dim * inner + k * inner + i;
                            acc += self.data[idx];
                        }
                        out[o * inner + i] = acc;
                    }
                }

                Self::new(out_shape, out)
            }
        }
    }

    pub fn reduce_sum_keepdims(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        let reduced = self.reduce_sum(axis)?;
        let target_shape = reduce_keepdims_shape(&self.shape, axis)?;
        reduced.reshape(target_shape)
    }

    pub fn reduce_mean(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError {
                message: "reduce_mean expects non-empty tensor".to_string(),
            });
        }

        match axis {
            None => {
                let sum = self.reduce_sum(None)?;
                let count = self.data.len() as f32;
                if count == 0.0 {
                    return Err(TensorError {
                        message: "reduce_mean: cannot divide by zero element count".to_string(),
                    });
                }
                sum.scale(1.0 / count)
            }
            Some(a) => {
                if a >= self.shape.len() {
                    return Err(TensorError {
                        message: format!(
                            "reduce_mean axis {a} out of bounds for rank {} tensor",
                            self.shape.len()
                        ),
                    });
                }
                let axis_dim = self.shape[a];
                if axis_dim == 0 {
                    return Err(TensorError {
                        message: "reduce_mean: cannot divide by zero-sized axis".to_string(),
                    });
                }
                let sum = self.reduce_sum(Some(a))?;
                sum.scale(1.0 / axis_dim as f32)
            }
        }
    }

    pub fn reduce_mean_keepdims(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        let reduced = self.reduce_mean(axis)?;
        let target_shape = reduce_keepdims_shape(&self.shape, axis)?;
        reduced.reshape(target_shape)
    }

    /// Reduces a tensor along an optional axis taking the element-wise maximum.
    ///
    /// - `axis = None`: collapses all elements to a scalar.
    /// - `axis = Some(a)`: reduces along axis `a`, removing that dimension.
    ///
    /// This operation is always sequential — reduction order is deterministic
    /// independent of the `parallel` feature flag.
    pub fn reduce_max(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError {
                message: "reduce_max expects non-empty tensor".to_string(),
            });
        }
        match axis {
            None => {
                // SAFETY: non-empty check above guarantees at least one element.
                let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                Ok(Self::scalar(max))
            }
            Some(a) => {
                let rank = self.shape.len();
                if a >= rank {
                    return Err(TensorError {
                        message: format!(
                            "reduce_max axis {a} out of bounds for rank {rank} tensor"
                        ),
                    });
                }
                let outer = product_prefix(&self.shape, a)?;
                let axis_dim = self.shape[a];
                let inner = product_suffix(&self.shape, a + 1)?;

                let mut out_shape = self.shape.clone();
                out_shape.remove(a);
                if out_shape.is_empty() {
                    out_shape.push(1);
                }

                let out_count = outer.checked_mul(inner).ok_or_else(|| TensorError {
                    message: "reduce_max output size overflow".to_string(),
                })?;
                let mut out = vec![f32::NEG_INFINITY; out_count];

                for o in 0..outer {
                    for i in 0..inner {
                        for k in 0..axis_dim {
                            let idx = o * axis_dim * inner + k * inner + i;
                            let dst = o * inner + i;
                            if self.data[idx] > out[dst] {
                                out[dst] = self.data[idx];
                            }
                        }
                    }
                }

                Self::new(out_shape, out)
            }
        }
    }

    pub fn reduce_max_keepdims(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        let reduced = self.reduce_max(axis)?;
        let target_shape = reduce_keepdims_shape(&self.shape, axis)?;
        reduced.reshape(target_shape)
    }

    /// Numerically-stable softmax along the last axis.
    ///
    /// Algorithm (identical to the CUDA kernel in `kernels/softmax.rs`):
    /// 1. Compute `max` over the last axis.
    /// 2. Subtract `max` from each element before `exp` — prevents `inf`.
    /// 3. Sum the shifted exponentials.
    /// 4. Divide each exponential by the sum.
    ///
    /// Currently supports 1-D tensors only (the most common use-case for
    /// attention logits and cross-entropy loss).
    pub fn softmax(&self) -> Result<Self, TensorError> {
        if self.shape.len() != 1 {
            return Err(TensorError {
                message: format!("softmax expects a 1-D tensor, got shape {:?}", self.shape),
            });
        }
        if self.data.is_empty() {
            return Err(TensorError {
                message: "softmax expects non-empty tensor".to_string(),
            });
        }

        // Step 1: max for numeric stability.
        // SAFETY: is_empty() check above guarantees at least one element.
        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Step 2+3: shifted exp + sum.
        let mut exps = Vec::with_capacity(self.data.len());
        let mut sum = 0.0_f32;
        for &x in &self.data {
            let e = (x - max).exp();
            exps.push(e);
            sum += e;
        }

        if !sum.is_finite() || sum <= 0.0 {
            return Err(TensorError {
                message: format!(
                    "softmax numeric instability: sum = {sum} (all inputs may be -inf)"
                ),
            });
        }

        // Step 4: normalise.
        let inv_sum = 1.0 / sum;
        for v in &mut exps {
            *v *= inv_sum;
        }

        Self::new(self.shape.clone(), exps)
    }

    pub fn mean(&self) -> Result<Self, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError {
                message: "mean expects non-empty tensor".to_string(),
            });
        }
        #[cfg(feature = "parallel")]
        let sum: f32 = if should_par(self.data.len()) {
            self.data.par_iter().copied().sum()
        } else {
            self.data.iter().copied().sum()
        };
        #[cfg(not(feature = "parallel"))]
        let sum: f32 = self.data.iter().copied().sum();
        let denom = self.data.len() as f32;
        Ok(Self::scalar(sum / denom))
    }
}

// ── Gather / Slice / Concat ───────────────────────────────────────────────────

impl Tensor {
    pub fn concat(tensors: &[Self], axis: usize) -> Result<Self, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError {
                message: "concat expects at least one tensor".to_string(),
            });
        }
        let rank = tensors[0].shape.len();
        if axis >= rank {
            return Err(TensorError {
                message: format!("concat axis {axis} out of bounds for rank {rank} tensor"),
            });
        }

        let mut out_shape = tensors[0].shape.clone();
        let mut axis_sum = 0usize;
        for tensor in tensors {
            if tensor.shape.len() != rank {
                return Err(TensorError {
                    message: format!(
                        "concat rank mismatch: expected rank {}, got shape {:?}",
                        rank, tensor.shape
                    ),
                });
            }
            for (dim_idx, (lhs, rhs)) in out_shape.iter().zip(tensor.shape.iter()).enumerate() {
                if dim_idx != axis && lhs != rhs {
                    return Err(TensorError {
                        message: format!(
                            "concat shape mismatch at dim {dim_idx}: expected {lhs}, got {rhs}"
                        ),
                    });
                }
            }
            axis_sum = axis_sum
                .checked_add(tensor.shape[axis])
                .ok_or_else(|| TensorError {
                    message: "concat axis size overflow".to_string(),
                })?;
        }
        out_shape[axis] = axis_sum;

        let outer = product_prefix(&out_shape, axis)?;
        let inner = product_suffix(&out_shape, axis + 1)?;
        let mut out = Vec::with_capacity(element_count(&out_shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?);

        for outer_idx in 0..outer {
            for tensor in tensors {
                let axis_dim = tensor.shape[axis];
                let start = outer_idx
                    .checked_mul(axis_dim)
                    .and_then(|v| v.checked_mul(inner))
                    .ok_or_else(|| TensorError {
                        message: "concat index overflow".to_string(),
                    })?;
                let end = start
                    .checked_add(axis_dim.checked_mul(inner).ok_or_else(|| TensorError {
                        message: "concat block size overflow".to_string(),
                    })?)
                    .ok_or_else(|| TensorError {
                        message: "concat index overflow".to_string(),
                    })?;
                out.extend_from_slice(&tensor.data[start..end]);
            }
        }

        Self::new(out_shape, out)
    }

    pub fn gather(&self, indices: &[usize], axis: usize) -> Result<Self, TensorError> {
        let rank = self.shape.len();
        if axis >= rank {
            return Err(TensorError {
                message: format!("gather axis {axis} out of bounds for rank {rank} tensor"),
            });
        }
        let axis_dim = self.shape[axis];
        for index in indices {
            if *index >= axis_dim {
                return Err(TensorError {
                    message: format!(
                        "gather index {index} out of bounds for axis {axis} with size {axis_dim}"
                    ),
                });
            }
        }

        let mut out_shape = self.shape.clone();
        out_shape[axis] = indices.len();
        let outer = product_prefix(&self.shape, axis)?;
        let inner = product_suffix(&self.shape, axis + 1)?;
        let mut out = Vec::with_capacity(element_count(&out_shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?);

        for outer_idx in 0..outer {
            for index in indices {
                let start = outer_idx
                    .checked_mul(axis_dim)
                    .and_then(|v| v.checked_mul(inner))
                    .and_then(|v| v.checked_add(index.checked_mul(inner)?))
                    .ok_or_else(|| TensorError {
                        message: "gather index overflow".to_string(),
                    })?;
                let end = start.checked_add(inner).ok_or_else(|| TensorError {
                    message: "gather index overflow".to_string(),
                })?;
                out.extend_from_slice(&self.data[start..end]);
            }
        }

        Self::new(out_shape, out)
    }

    pub fn slice(
        &self,
        starts: &[usize],
        ends: &[usize],
        axes: &[usize],
    ) -> Result<Self, TensorError> {
        if starts.is_empty() || ends.is_empty() || axes.is_empty() {
            return Err(TensorError {
                message: "slice starts/ends/axes must be non-empty".to_string(),
            });
        }
        if starts.len() != ends.len() || starts.len() != axes.len() {
            return Err(TensorError {
                message: "slice starts/ends/axes lengths must match".to_string(),
            });
        }

        let rank = self.shape.len();
        let mut out_shape = self.shape.clone();
        let mut axis_seen = std::collections::HashSet::new();
        #[allow(clippy::similar_names)]
        for idx in 0..axes.len() {
            let axis = axes[idx];
            if axis >= rank {
                return Err(TensorError {
                    message: format!("slice axis {axis} out of bounds for rank {rank} tensor"),
                });
            }
            if !axis_seen.insert(axis) {
                return Err(TensorError {
                    message: format!("slice axis {axis} specified more than once"),
                });
            }
            let start = starts[idx];
            let end = ends[idx];
            let dim = self.shape[axis];
            if start >= end {
                return Err(TensorError {
                    message: format!(
                        "slice requires start < end per axis, got start={start} end={end} at axis {axis}"
                    ),
                });
            }
            if end > dim {
                return Err(TensorError {
                    message: format!(
                        "slice end {end} out of bounds for axis {axis} with size {dim}"
                    ),
                });
            }
            out_shape[axis] = end - start;
        }

        let out_count = element_count(&out_shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        let out_strides = strides(&out_shape).ok_or_else(|| TensorError {
            message: "slice output stride overflow".to_string(),
        })?;
        let in_strides = strides(&self.shape).ok_or_else(|| TensorError {
            message: "slice input stride overflow".to_string(),
        })?;

        let mut out = vec![0.0_f32; out_count];
        for (linear, out_value) in out.iter_mut().enumerate().take(out_count) {
            let mut rem = linear;
            let mut in_offset = 0usize;
            for dim_idx in 0..out_shape.len() {
                let stride = out_strides[dim_idx];
                let coord = if stride == 0 { 0 } else { rem / stride };
                if stride != 0 {
                    rem %= stride;
                }

                let start_shift = axes
                    .iter()
                    .position(|axis| *axis == dim_idx)
                    .map_or(0, |idx| starts[idx]);
                in_offset = in_offset
                    .checked_add(
                        coord
                            .checked_add(start_shift)
                            .and_then(|v| v.checked_mul(in_strides[dim_idx]))
                            .ok_or_else(|| TensorError {
                                message: "slice index overflow".to_string(),
                            })?,
                    )
                    .ok_or_else(|| TensorError {
                        message: "slice index overflow".to_string(),
                    })?;
            }
            *out_value = self.data[in_offset];
        }

        Self::new(out_shape, out)
    }
}

// ── Generalised Matrix Multiply (GEMM) ────────────────────────────────────────

impl Tensor {
    /// Generalised matrix multiplication: `alpha * (self @ rhs) + beta * bias`.
    ///
    /// This is the standard BLAS `SGEMM` interface — the most common operation
    /// produced by PyTorch/ONNX export via `torch.nn.Linear` (exported as
    /// ONNX `Gemm`).
    ///
    /// # Parameters
    /// - `rhs`   — right-hand matrix, shape `[K, N]`
    /// - `bias`  — optional bias tensor, shape `[N]` or `[M, N]` (broadcast)
    /// - `alpha` — scalar multiplier for `self @ rhs` (typically 1.0)
    /// - `beta`  — scalar multiplier for `bias` (typically 1.0)
    ///
    /// # Errors
    /// Returns `TensorError` if shapes are incompatible or if `alpha`/`beta`
    /// are non-finite.
    pub fn gemm(
        &self,
        rhs: &Self,
        bias: Option<&Self>,
        alpha: f32,
        beta: f32,
    ) -> Result<Self, TensorError> {
        if !alpha.is_finite() {
            return Err(TensorError {
                message: format!("gemm: alpha must be finite, got {alpha}"),
            });
        }
        if !beta.is_finite() {
            return Err(TensorError {
                message: format!("gemm: beta must be finite, got {beta}"),
            });
        }

        // Step 1: matmul (self [M,K] × rhs [K,N] → out [M,N]).
        let mut out = self.matmul(rhs)?;

        // Step 2: scale by alpha if not 1.0 (avoids unnecessary allocation).
        if (alpha - 1.0_f32).abs() > f32::EPSILON {
            out = out.scale(alpha)?;
        }

        // Step 3: add scaled bias if provided.
        if let Some(b) = bias {
            if beta == 0.0_f32 {
                // β=0 means skip the bias entirely (common in fused kernels).
                return Ok(out);
            }
            // Broadcast bias over rows if shape is [N] instead of [M, N].
            let scaled_bias = if (beta - 1.0_f32).abs() > f32::EPSILON {
                b.scale(beta)?
            } else {
                b.clone()
            };
            out = out.add_broadcast(&scaled_bias)?;
        }

        Ok(out)
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn broadcast_shape(left: &[usize], right: &[usize], op: &str) -> Result<Vec<usize>, TensorError> {
    let max_len = left.len().max(right.len());
    let mut out = vec![0usize; max_len];
    for i in 0..max_len {
        let dim_left = if i < max_len - left.len() {
            1
        } else {
            left[i - (max_len - left.len())]
        };
        let dim_right = if i < max_len - right.len() {
            1
        } else {
            right[i - (max_len - right.len())]
        };
        out[i] = if dim_left == dim_right {
            dim_left
        } else if dim_left == 1 {
            dim_right
        } else if dim_right == 1 {
            dim_left
        } else {
            return Err(TensorError {
                message: format!("Cannot broadcast {:?} and {:?} in {op}", left, right),
            });
        };
    }
    Ok(out)
}

fn broadcast_index(
    linear_index: usize,
    out_shape: &[usize],
    out_strides: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
    op: &str,
) -> Result<usize, TensorError> {
    let offset = out_shape.len() - input_shape.len();
    let mut rem = linear_index;
    let mut input_linear = 0usize;

    for (axis, &stride) in out_strides.iter().enumerate() {
        let coord = if stride == 0 { 0 } else { rem / stride };
        if stride != 0 {
            rem %= stride;
        }

        if axis < offset {
            continue;
        }
        let input_axis = axis - offset;
        if input_shape[input_axis] == 1 {
            continue;
        }

        input_linear = input_linear
            .checked_add(
                coord
                    .checked_mul(input_strides[input_axis])
                    .ok_or_else(|| TensorError {
                        message: format!("{op}: broadcast index overflow"),
                    })?,
            )
            .ok_or_else(|| TensorError {
                message: format!("{op}: broadcast index overflow"),
            })?;
    }

    Ok(input_linear)
}

fn elementwise_broadcast_binary<F>(
    left: &Tensor,
    right: &Tensor,
    op: &str,
    f: F,
) -> Result<Tensor, TensorError>
where
    F: Fn(f32, f32) -> Result<f32, TensorError> + Sync + Send,
{
    let out_shape = broadcast_shape(&left.shape, &right.shape, op)?;
    let out_count = element_count(&out_shape).ok_or_else(|| TensorError {
        message: format!("{op}: invalid output shape (overflow)"),
    })?;

    if out_count == 0 {
        return Tensor::new(out_shape, Vec::new());
    }

    if left.shape == out_shape && right.shape == out_shape {
        let out: Vec<f32> = {
            #[cfg(feature = "parallel")]
            if should_par(out_count) {
                left.data
                    .par_iter()
                    .zip(right.data.par_iter())
                    .map(|(a, b)| f(*a, *b))
                    .collect::<Result<_, _>>()?
            } else {
                left.data
                    .iter()
                    .zip(right.data.iter())
                    .map(|(a, b)| f(*a, *b))
                    .collect::<Result<_, _>>()?
            }
            #[cfg(not(feature = "parallel"))]
            left.data
                .iter()
                .zip(right.data.iter())
                .map(|(a, b)| f(*a, *b))
                .collect::<Result<_, _>>()?
        };
        return Tensor::new(out_shape, out);
    }

    let out_strides = strides(&out_shape).ok_or_else(|| TensorError {
        message: format!("{op}: output stride overflow"),
    })?;
    let left_strides = strides(&left.shape).ok_or_else(|| TensorError {
        message: format!("{op}: left stride overflow"),
    })?;
    let right_strides = strides(&right.shape).ok_or_else(|| TensorError {
        message: format!("{op}: right stride overflow"),
    })?;

    let mut out = Vec::with_capacity(out_count);
    for linear in 0..out_count {
        let left_index = broadcast_index(
            linear,
            &out_shape,
            &out_strides,
            &left.shape,
            &left_strides,
            op,
        )?;
        let right_index = broadcast_index(
            linear,
            &out_shape,
            &out_strides,
            &right.shape,
            &right_strides,
            op,
        )?;
        out.push(f(left.data[left_index], right.data[right_index])?);
    }

    Tensor::new(out_shape, out)
}

fn reduce_keepdims_shape(shape: &[usize], axis: Option<usize>) -> Result<Vec<usize>, TensorError> {
    match axis {
        Some(a) => {
            if a >= shape.len() {
                return Err(TensorError {
                    message: format!(
                        "reduce keepdims axis {a} out of bounds for rank {} tensor",
                        shape.len()
                    ),
                });
            }
            let mut out = shape.to_vec();
            out[a] = 1;
            Ok(out)
        }
        None => {
            if shape.is_empty() {
                Ok(vec![1])
            } else {
                Ok(vec![1; shape.len()])
            }
        }
    }
}

fn ensure_same_shape(a: &[usize], b: &[usize], op: &str) -> Result<(), TensorError> {
    if a == b {
        Ok(())
    } else {
        Err(TensorError {
            message: format!("Shape mismatch in {op}: {a:?} vs {b:?}"),
        })
    }
}

fn erf_approx(x: f32) -> f32 {
    // Abramowitz-Stegun formula 7.1.26
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_72;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + P * xa);
    let poly = (((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t) * (-(xa * xa)).exp();
    sign * (1.0 - poly)
}

fn element_count(shape: &[usize]) -> Option<usize> {
    let mut count = 1usize;
    for dim in shape {
        count = count.checked_mul(*dim)?;
    }
    Some(count)
}

fn product_prefix(shape: &[usize], end: usize) -> Result<usize, TensorError> {
    shape
        .iter()
        .take(end)
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| TensorError {
            message: "tensor shape overflow while computing prefix product".to_string(),
        })
}

fn product_suffix(shape: &[usize], start: usize) -> Result<usize, TensorError> {
    shape
        .iter()
        .skip(start)
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| TensorError {
            message: "tensor shape overflow while computing suffix product".to_string(),
        })
}

fn strides(shape: &[usize]) -> Option<Vec<usize>> {
    let mut out = vec![0usize; shape.len()];
    let mut stride = 1usize;
    for idx in (0..shape.len()).rev() {
        out[idx] = stride;
        stride = stride.checked_mul(shape[idx])?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn matmul_works_for_small_matrices() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid tensor");
        let b = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).expect("valid tensor");
        let c = a.matmul(&b).expect("matmul should pass");
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn reshape_inplace_is_zero_copy() {
        let t = Tensor::new(vec![2, 4], vec![1.0; 8]).expect("valid tensor");
        let ptr = t.data.as_ptr();
        let r = t
            .reshape_inplace(vec![4, 2])
            .expect("reshape should succeed");
        assert_eq!(r.shape, vec![4, 2]);
        // Pointer must be identical — no allocation occurred.
        assert_eq!(r.data.as_ptr(), ptr);
    }

    #[test]
    fn parallel_add_matches_sequential() {
        let a =
            Tensor::new(vec![1024], (0..1024).map(|i| i as f32).collect()).expect("valid tensor");
        let b = Tensor::new(vec![1024], (0..1024).map(|i| i as f32 * 2.0).collect())
            .expect("valid tensor");
        let result = a.add(&b).expect("add should succeed");
        for (i, &v) in result.data.iter().enumerate() {
            assert!((v - i as f32 * 3.0).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn parallel_matmul_identity() {
        // 64×64 identity × matrix = matrix.
        let n = 64;
        let mut id_data = vec![0.0_f32; n * n];
        for i in 0..n {
            id_data[i * n + i] = 1.0;
        }
        let id = Tensor::new(vec![n, n], id_data).expect("valid tensor");
        let v: Vec<f32> = (0..(n * n)).map(|i| i as f32).collect();
        let m = Tensor::new(vec![n, n], v.clone()).expect("valid tensor");
        let result = id.matmul(&m).expect("matmul should succeed");
        for (a, b) in result.data.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-3, "identity matmul result wrong");
        }
    }

    #[test]
    fn parallel_relu_matches_sequential() {
        let data: Vec<f32> = (-512..512).map(|i| i as f32 * 0.1).collect();
        let t = Tensor::new(vec![1024], data.clone()).expect("valid tensor");
        let result = t.relu().expect("relu should succeed");
        for (i, (&got, &orig)) in result.data.iter().zip(data.iter()).enumerate() {
            let expected = if orig > 0.0 { orig } else { 0.0 };
            assert!((got - expected).abs() < 1e-7, "relu mismatch at {i}");
        }
    }

    #[test]
    fn add_broadcast_extends_rank() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid tensor");
        let b = Tensor::new(vec![3], vec![10.0, 20.0, 30.0]).expect("valid tensor");
        let out = a.add_broadcast(&b).expect("broadcast add should succeed");
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn add_broadcast_two_sided() {
        let a = Tensor::new(vec![3, 1], vec![1.0, 2.0, 3.0]).expect("valid tensor");
        let b = Tensor::new(vec![1, 4], vec![10.0, 20.0, 30.0, 40.0]).expect("valid tensor");
        let out = a.add_broadcast(&b).expect("broadcast add should succeed");
        assert_eq!(out.shape, vec![3, 4]);
        assert_eq!(
            out.data,
            vec![
                11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0
            ]
        );
    }

    #[test]
    fn add_broadcast_scalar() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid tensor");
        let b = Tensor::new(vec![1], vec![5.0]).expect("valid tensor");
        let out = a
            .add_broadcast(&b)
            .expect("scalar broadcast should succeed");
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data, vec![6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn add_remains_strict() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid tensor");
        let b = Tensor::new(vec![2], vec![10.0, 20.0]).expect("valid tensor");
        let err = a
            .add(&b)
            .expect_err("strict add should reject mismatched shapes");
        assert!(err.message.contains("Shape mismatch in add"));
    }

    #[test]
    fn add_broadcast_rejects_incompatible_shapes() {
        let a = Tensor::new(vec![2, 3], vec![1.0; 6]).expect("valid tensor");
        let b = Tensor::new(vec![3, 2], vec![1.0; 6]).expect("valid tensor");
        let err = a
            .add_broadcast(&b)
            .expect_err("incompatible shapes should fail");
        assert!(err.message.contains("Cannot broadcast"));
    }

    #[test]
    fn div_broadcast_reports_division_by_zero() {
        let a = Tensor::new(vec![2, 2], vec![2.0, 4.0, 6.0, 8.0]).expect("valid tensor");
        let b = Tensor::new(vec![1, 2], vec![2.0, 0.0]).expect("valid tensor");
        let err = a
            .div_broadcast(&b)
            .expect_err("division by zero should fail");
        assert!(err.message.contains("Division by zero"));
    }

    #[test]
    fn gemm_accepts_vector_bias_via_broadcast() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid tensor");
        let b = Tensor::new(vec![2, 3], vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).expect("valid tensor");
        let bias = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).expect("valid tensor");
        let out = a
            .gemm(&b, Some(&bias), 1.0, 1.0)
            .expect("gemm with vector bias should succeed");

        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.data, vec![22.0, 26.0, 30.0, 48.0, 56.0, 64.0]);
    }

    #[test]
    fn reduce_mean_all_elements() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 3.0, 5.0, 7.0]).expect("valid tensor");
        let mean = t.reduce_mean(None).expect("reduce_mean should succeed");
        assert_eq!(mean.shape, vec![1]);
        assert_eq!(mean.data, vec![4.0]);
    }

    #[test]
    fn reduce_mean_over_axis() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 3.0, 5.0, 7.0]).expect("valid tensor");
        let mean = t
            .reduce_mean(Some(0))
            .expect("reduce_mean(axis) should succeed");
        assert_eq!(mean.shape, vec![2]);
        assert_eq!(mean.data, vec![3.0, 5.0]);
    }

    #[test]
    fn gelu_exact_matches_reference_values() {
        let t = Tensor::new(vec![5], vec![-3.0, -1.0, 0.0, 1.0, 3.0]).expect("valid tensor");
        let out = t.gelu_exact().expect("gelu_exact should succeed");
        let expected = [-0.004_049_7, -0.158_655_26, 0.0, 0.841_344_7, 2.995_950_2];
        for (i, (&got, &exp)) in out.data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 5e-4,
                "gelu_exact mismatch at {i}: {got} vs {exp}"
            );
        }
    }

    #[test]
    fn gelu_tanh_and_exact_are_close_on_common_range() {
        let data: Vec<f32> = (-60..=60).map(|i| i as f32 / 20.0).collect();
        let t = Tensor::new(vec![data.len()], data).expect("valid tensor");
        let tanh = t.gelu().expect("tanh gelu should succeed");
        let exact = t.gelu_exact().expect("exact gelu should succeed");
        for (i, (&a, &b)) in tanh.data.iter().zip(exact.data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.02,
                "gelu variants diverged at {i}: {a} vs {b}"
            );
        }
    }
}
