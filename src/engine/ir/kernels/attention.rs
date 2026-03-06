use crate::engine::ir::tensor::{Tensor, TensorError};

/// Scaled dot-product attention for a single head.
/// q: [batch, seq_q, head_dim]
/// k: [batch, seq_k, head_dim]
/// v: [batch, seq_k, head_dim]
/// mask: optional [seq_q, seq_k] (additive, -inf for masked positions)
/// Returns: (output [batch, seq_q, head_dim], attn_weights [batch, seq_q, seq_k])
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<(Tensor, Tensor), TensorError> {
    let q_c = q.make_contiguous()?;
    let k_c = k.make_contiguous()?;
    let v_c = v.make_contiguous()?;
    if q_c.shape.len() != 3 || k_c.shape.len() != 3 || v_c.shape.len() != 3 {
        return Err(TensorError { message: "SDPA expects rank-3 tensors".to_string() });
    }
    let (batch, seq_q, head_dim) = (q_c.shape[0], q_c.shape[1], q_c.shape[2]);
    let seq_k = k_c.shape[1];
    if k_c.shape != vec![batch, seq_k, head_dim] || v_c.shape[..2] != [batch, seq_k] {
        return Err(TensorError { message: "SDPA shape mismatch between Q, K, V".to_string() });
    }
    let v_dim = v_c.shape[2];

    let mut attn_logits = vec![0.0_f32; batch * seq_q * seq_k];
    let mut output = vec![0.0_f32; batch * seq_q * v_dim];
    let mut attn_weights_data = vec![0.0_f32; batch * seq_q * seq_k];

    for b in 0..batch {
        // Compute Q @ K^T * scale → [seq_q, seq_k]
        for i in 0..seq_q {
            for j in 0..seq_k {
                let mut dot = 0.0_f64;
                for d in 0..head_dim {
                    dot += q_c.data[(b * seq_q + i) * head_dim + d] as f64
                        * k_c.data[(b * seq_k + j) * head_dim + d] as f64;
                }
                let mut logit = dot as f32 * scale;
                if let Some(m) = mask {
                    let m_c = m.make_contiguous()?;
                    logit += m_c.data[i * seq_k + j];
                }
                attn_logits[(b * seq_q + i) * seq_k + j] = logit;
            }
        }

        // Softmax over seq_k for each query position
        for i in 0..seq_q {
            let row_start = (b * seq_q + i) * seq_k;
            let row = &attn_logits[row_start..row_start + seq_k];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max_v).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            let inv_sum = 1.0 / sum_e;
            for j in 0..seq_k {
                attn_weights_data[row_start + j] = exps[j] * inv_sum;
            }
        }

        // Output = attn_weights @ V → [seq_q, v_dim]
        for i in 0..seq_q {
            for d in 0..v_dim {
                let mut acc = 0.0_f64;
                for j in 0..seq_k {
                    acc += attn_weights_data[(b * seq_q + i) * seq_k + j] as f64
                        * v_c.data[(b * seq_k + j) * v_dim + d] as f64;
                }
                output[(b * seq_q + i) * v_dim + d] = acc as f32;
            }
        }
    }

    Ok((
        Tensor::new(vec![batch, seq_q, v_dim], output)?,
        Tensor::new(vec![batch, seq_q, seq_k], attn_weights_data)?,
    ))
}

pub struct SdpaGrads {
    pub dq: Tensor,
    pub dk: Tensor,
    pub dv: Tensor,
}

/// Backward through scaled dot-product attention.
pub fn scaled_dot_product_attention_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_weights: &Tensor,
    d_output: &Tensor,
    scale: f32,
) -> Result<SdpaGrads, TensorError> {
    let q_c = q.make_contiguous()?;
    let k_c = k.make_contiguous()?;
    let v_c = v.make_contiguous()?;
    let aw = attn_weights.make_contiguous()?;
    let do_c = d_output.make_contiguous()?;

    let (batch, seq_q, head_dim) = (q_c.shape[0], q_c.shape[1], q_c.shape[2]);
    let seq_k = k_c.shape[1];
    let v_dim = v_c.shape[2];

    let mut dq = vec![0.0_f32; batch * seq_q * head_dim];
    let mut dk = vec![0.0_f32; batch * seq_k * head_dim];
    let mut dv = vec![0.0_f32; batch * seq_k * v_dim];

    for b in 0..batch {
        // dV = attn_weights^T @ d_output  [seq_k, v_dim]
        for j in 0..seq_k {
            for d in 0..v_dim {
                let mut acc = 0.0_f64;
                for i in 0..seq_q {
                    acc += aw.data[(b * seq_q + i) * seq_k + j] as f64
                        * do_c.data[(b * seq_q + i) * v_dim + d] as f64;
                }
                dv[(b * seq_k + j) * v_dim + d] = acc as f32;
            }
        }

        // d_attn_weights = d_output @ V^T  [seq_q, seq_k]
        let mut d_aw = vec![0.0_f32; seq_q * seq_k];
        for i in 0..seq_q {
            for j in 0..seq_k {
                let mut acc = 0.0_f64;
                for d in 0..v_dim {
                    acc += do_c.data[(b * seq_q + i) * v_dim + d] as f64
                        * v_c.data[(b * seq_k + j) * v_dim + d] as f64;
                }
                d_aw[i * seq_k + j] = acc as f32;
            }
        }

        // Softmax backward: d_logits = attn_weights * (d_aw - sum(d_aw * attn_weights))
        let mut d_logits = vec![0.0_f32; seq_q * seq_k];
        for i in 0..seq_q {
            let aw_row = &aw.data[(b * seq_q + i) * seq_k..(b * seq_q + i) * seq_k + seq_k];
            let da_row = &d_aw[i * seq_k..(i + 1) * seq_k];
            let s: f32 = aw_row.iter().zip(da_row.iter()).map(|(&a, &da)| a * da).sum();
            for j in 0..seq_k {
                d_logits[i * seq_k + j] = aw_row[j] * (da_row[j] - s);
            }
        }

        // Scale backward
        for v in d_logits.iter_mut() { *v *= scale; }

        // dQ = d_logits @ K  [seq_q, head_dim]
        for i in 0..seq_q {
            for d in 0..head_dim {
                let mut acc = 0.0_f64;
                for j in 0..seq_k {
                    acc += d_logits[i * seq_k + j] as f64
                        * k_c.data[(b * seq_k + j) * head_dim + d] as f64;
                }
                dq[(b * seq_q + i) * head_dim + d] += acc as f32;
            }
        }

        // dK = d_logits^T @ Q  [seq_k, head_dim]
        for j in 0..seq_k {
            for d in 0..head_dim {
                let mut acc = 0.0_f64;
                for i in 0..seq_q {
                    acc += d_logits[i * seq_k + j] as f64
                        * q_c.data[(b * seq_q + i) * head_dim + d] as f64;
                }
                dk[(b * seq_k + j) * head_dim + d] += acc as f32;
            }
        }
    }

    Ok(SdpaGrads {
        dq: Tensor::new(vec![batch, seq_q, head_dim], dq)?,
        dk: Tensor::new(vec![batch, seq_k, head_dim], dk)?,
        dv: Tensor::new(vec![batch, seq_k, v_dim], dv)?,
    })
}

/// Multi-Head Attention forward.
/// q_input, k_input, v_input: [batch, seq, d_model]
/// w_q, w_k, w_v: [d_model, d_model]   (output projections)
/// w_o: [d_model, d_model]
/// bias_q, bias_k, bias_v, bias_o: [d_model] (optional, pass zeros)
/// num_heads: number of attention heads
/// Returns: (output [batch, seq, d_model], attn_weights [batch*heads, seq_q, seq_k])
pub struct MhaOutput {
    pub output: Tensor,
    /// Attention weights for all heads: [batch*num_heads, seq_q, seq_k]
    pub attn_weights: Tensor,
    /// Q projected [batch, seq, d_model]
    pub q_proj: Tensor,
    /// K projected [batch, seq_k, d_model]
    pub k_proj: Tensor,
    /// V projected [batch, seq_k, d_model]
    pub v_proj: Tensor,
    /// Context (pre-output-projection) [batch, seq_q, d_model]
    pub context: Tensor,
}

pub fn multi_head_attention(
    q_input: &Tensor,
    k_input: &Tensor,
    v_input: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
    w_o: &Tensor,
    bias_q: &Tensor,
    bias_k: &Tensor,
    bias_v: &Tensor,
    bias_o: &Tensor,
    num_heads: usize,
    causal: bool,
) -> Result<MhaOutput, TensorError> {
    let q_c = q_input.make_contiguous()?;
    if q_c.shape.len() != 3 {
        return Err(TensorError { message: "MHA expects rank-3 inputs [batch, seq, d_model]".to_string() });
    }
    let (batch, seq_q, d_model) = (q_c.shape[0], q_c.shape[1], q_c.shape[2]);
    let k_c = k_input.make_contiguous()?;
    let seq_k = k_c.shape[1];

    if d_model % num_heads != 0 {
        return Err(TensorError { message: format!("d_model {d_model} not divisible by num_heads {num_heads}") });
    }
    let head_dim = d_model / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Helper: linear projection [batch, seq, d_model] @ [d_model, d_model] + [d_model] → [batch, seq, d_model]
    let linear = |input: &Tensor, w: &Tensor, b: &Tensor| -> Result<Tensor, TensorError> {
        let inp = input.make_contiguous()?;
        let wc = w.make_contiguous()?;
        let bc = b.make_contiguous()?;
        let (bs, sq, d) = (inp.shape[0], inp.shape[1], inp.shape[2]);
        let d_out = wc.shape[0];
        let mut out = vec![0.0_f32; bs * sq * d_out];
        for n in 0..bs {
            for s in 0..sq {
                for o in 0..d_out {
                    let mut v = bc.data[o] as f64;
                    for k in 0..d {
                        v += inp.data[(n * sq + s) * d + k] as f64 * wc.data[o * d + k] as f64;
                    }
                    out[(n * sq + s) * d_out + o] = v as f32;
                }
            }
        }
        Tensor::new(vec![bs, sq, d_out], out)
    };

    let q_proj = linear(q_input, w_q, bias_q)?;
    let k_proj = linear(k_input, w_k, bias_k)?;
    let v_proj = linear(v_input, w_v, bias_v)?;

    // Reshape: [batch, seq, d_model] → [batch*num_heads, seq, head_dim]
    let reshape_heads = |t: &Tensor, sq: usize| -> Result<Tensor, TensorError> {
        let tc = t.make_contiguous()?;
        let mut out = vec![0.0_f32; batch * num_heads * sq * head_dim];
        for b in 0..batch {
            for s in 0..sq {
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        out[(b * num_heads + h) * sq * head_dim + s * head_dim + d] =
                            tc.data[(b * sq + s) * d_model + h * head_dim + d];
                    }
                }
            }
        }
        Tensor::new(vec![batch * num_heads, sq, head_dim], out)
    };

    let q_heads = reshape_heads(&q_proj, seq_q)?;
    let k_heads = reshape_heads(&k_proj, seq_k)?;
    let v_heads = reshape_heads(&v_proj, seq_k)?;

    // Optional causal mask: [seq_q, seq_k], -inf for future positions
    let mask_tensor;
    let mask_ref = if causal {
        let mut mask_data = vec![0.0_f32; seq_q * seq_k];
        for i in 0..seq_q {
            for j in 0..seq_k {
                if j > i {
                    mask_data[i * seq_k + j] = f32::NEG_INFINITY;
                }
            }
        }
        mask_tensor = Tensor::new(vec![seq_q, seq_k], mask_data)?;
        Some(&mask_tensor)
    } else {
        None
    };

    let (attn_out, attn_weights) = scaled_dot_product_attention(
        &q_heads, &k_heads, &v_heads, mask_ref, scale
    )?;

    // Reshape back: [batch*num_heads, seq_q, head_dim] → [batch, seq_q, d_model]
    let attn_out_c = attn_out.make_contiguous()?;
    let mut context_data = vec![0.0_f32; batch * seq_q * d_model];
    for b in 0..batch {
        for s in 0..seq_q {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    context_data[(b * seq_q + s) * d_model + h * head_dim + d] =
                        attn_out_c.data[(b * num_heads + h) * seq_q * head_dim + s * head_dim + d];
                }
            }
        }
    }
    let context = Tensor::new(vec![batch, seq_q, d_model], context_data)?;

    // Output projection
    let output = linear(&context, w_o, bias_o)?;

    Ok(MhaOutput {
        output,
        attn_weights,
        q_proj,
        k_proj,
        v_proj,
        context,
    })
}
