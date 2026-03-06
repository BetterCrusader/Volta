use crate::engine::ir::tensor::{Tensor, TensorError};

/// Sinusoidal positional encoding.
/// Input: [batch, seq_len, d_model] or [seq_len, d_model].
/// Returns: input + PE (same shape).
pub fn sinusoidal_pe(input: &Tensor) -> Result<Tensor, TensorError> {
    let inp = input.make_contiguous()?;
    let rank = inp.shape.len();
    if rank < 2 {
        return Err(TensorError {
            message: "SinusoidalPE expects rank >= 2".to_string(),
        });
    }
    let d_model = *inp.shape.last().unwrap();
    let seq_len = inp.shape[rank - 2];
    let outer: usize = inp.shape[..rank - 2].iter().product::<usize>().max(1);

    if d_model % 2 != 0 {
        return Err(TensorError {
            message: format!("SinusoidalPE: d_model must be even, got {d_model}"),
        });
    }

    // Precompute PE table [seq_len, d_model]
    let mut pe = vec![0.0_f32; seq_len * d_model];
    for pos in 0..seq_len {
        for i in 0..(d_model / 2) {
            let denom = 10000.0_f64.powf(2.0 * i as f64 / d_model as f64);
            let angle = pos as f64 / denom;
            pe[pos * d_model + 2 * i] = angle.sin() as f32;
            pe[pos * d_model + 2 * i + 1] = angle.cos() as f32;
        }
    }

    let mut out = inp.data.to_vec();
    for b in 0..outer {
        for s in 0..seq_len {
            for d in 0..d_model {
                out[(b * seq_len + s) * d_model + d] += pe[s * d_model + d];
            }
        }
    }
    Tensor::new(inp.shape.clone(), out)
}

/// Rotary Position Embedding (RoPE).
/// input: [batch, seq_len, head_dim] or [batch, heads, seq_len, head_dim].
/// Rotates pairs of dimensions in the last dim by position-dependent angles.
/// offset: starting position (for KV cache).
pub fn rope(input: &Tensor, offset: usize) -> Result<Tensor, TensorError> {
    let inp = input.make_contiguous()?;
    let rank = inp.shape.len();
    if rank < 2 {
        return Err(TensorError {
            message: "RoPE expects rank >= 2".to_string(),
        });
    }
    let head_dim = *inp.shape.last().unwrap();
    let seq_len = inp.shape[rank - 2];
    if head_dim % 2 != 0 {
        return Err(TensorError {
            message: format!("RoPE: head_dim must be even, got {head_dim}"),
        });
    }
    let outer: usize = inp.shape[..rank - 2].iter().product::<usize>().max(1);

    let mut out = inp.data.to_vec();
    for b in 0..outer {
        for s in 0..seq_len {
            let pos = (s + offset) as f64;
            let base = (b * seq_len + s) * head_dim;
            for i in 0..(head_dim / 2) {
                let theta = pos / 10000.0_f64.powf(2.0 * i as f64 / head_dim as f64);
                let (sin_t, cos_t) = (theta.sin() as f32, theta.cos() as f32);
                let x0 = inp.data[base + 2 * i];
                let x1 = inp.data[base + 2 * i + 1];
                out[base + 2 * i] = x0 * cos_t - x1 * sin_t;
                out[base + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    Tensor::new(inp.shape.clone(), out)
}

/// RoPE backward: same rotation with transposed rotation matrix (cos stays same, sin negated).
pub fn rope_backward(upstream: &Tensor, offset: usize) -> Result<Tensor, TensorError> {
    let up = upstream.make_contiguous()?;
    let rank = up.shape.len();
    let head_dim = *up.shape.last().unwrap();
    let seq_len = up.shape[rank - 2];
    let outer: usize = up.shape[..rank - 2].iter().product::<usize>().max(1);

    let mut grad = up.data.to_vec();
    for b in 0..outer {
        for s in 0..seq_len {
            let pos = (s + offset) as f64;
            let base = (b * seq_len + s) * head_dim;
            for i in 0..(head_dim / 2) {
                let theta = pos / 10000.0_f64.powf(2.0 * i as f64 / head_dim as f64);
                let (sin_t, cos_t) = (theta.sin() as f32, theta.cos() as f32);
                // Inverse rotation: multiply by R^T (negate sin)
                let dy0 = up.data[base + 2 * i];
                let dy1 = up.data[base + 2 * i + 1];
                grad[base + 2 * i] = dy0 * cos_t + dy1 * sin_t;
                grad[base + 2 * i + 1] = -dy0 * sin_t + dy1 * cos_t;
            }
        }
    }
    Tensor::new(up.shape.clone(), grad)
}
