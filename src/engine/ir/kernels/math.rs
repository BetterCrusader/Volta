use crate::engine::ir::kernels::utils::{
    ensure_same_shape, product_prefix, product_suffix, should_par,
};
use crate::engine::ir::tensor::{Tensor, TensorError};
#[cfg(feature = "parallel")]
use gemm::Parallelism;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn add(left: &Tensor, right: &Tensor) -> Result<Tensor, TensorError> {
    ensure_same_shape(&left.shape, &right.shape, "add")?;
    let left_c = left.make_contiguous()?;
    let right_c = right.make_contiguous()?;

    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if should_par(left_c.data.len()) {
            left_c
                .data
                .par_iter()
                .zip(right_c.data.par_iter())
                .map(|(a, b)| a + b)
                .collect()
        } else {
            left_c
                .data
                .iter()
                .zip(right_c.data.iter())
                .map(|(a, b)| a + b)
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        left_c
            .data
            .iter()
            .zip(right_c.data.iter())
            .map(|(a, b)| a + b)
            .collect()
    };
    Tensor::new(left.shape.clone(), out)
}

pub fn sub(left: &Tensor, right: &Tensor) -> Result<Tensor, TensorError> {
    ensure_same_shape(&left.shape, &right.shape, "sub")?;
    let left_c = left.make_contiguous()?;
    let right_c = right.make_contiguous()?;

    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if should_par(left_c.data.len()) {
            left_c
                .data
                .par_iter()
                .zip(right_c.data.par_iter())
                .map(|(a, b)| a - b)
                .collect()
        } else {
            left_c
                .data
                .iter()
                .zip(right_c.data.iter())
                .map(|(a, b)| a - b)
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        left_c
            .data
            .iter()
            .zip(right_c.data.iter())
            .map(|(a, b)| a - b)
            .collect()
    };
    Tensor::new(left.shape.clone(), out)
}

pub fn mul_elementwise(left: &Tensor, right: &Tensor) -> Result<Tensor, TensorError> {
    ensure_same_shape(&left.shape, &right.shape, "mul_elementwise")?;
    let left_c = left.make_contiguous()?;
    let right_c = right.make_contiguous()?;

    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if should_par(left_c.data.len()) {
            left_c
                .data
                .par_iter()
                .zip(right_c.data.par_iter())
                .map(|(a, b)| a * b)
                .collect()
        } else {
            left_c
                .data
                .iter()
                .zip(right_c.data.iter())
                .map(|(a, b)| a * b)
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        left_c
            .data
            .iter()
            .zip(right_c.data.iter())
            .map(|(a, b)| a * b)
            .collect()
    };
    Tensor::new(left.shape.clone(), out)
}

pub fn matmul(left: &Tensor, right: &Tensor) -> Result<Tensor, TensorError> {
    if left.shape.len() != 2 || right.shape.len() != 2 {
        return Err(TensorError {
            message: "Matmul expects rank-2".to_string(),
        });
    }
    let (m, k, n) = (left.shape[0], left.shape[1], right.shape[1]);
    if k != right.shape[0] {
        return Err(TensorError {
            message: format!(
                "Shape mismatch in matmul: [{m}x{k}] @ [{k_right}x{n}]",
                k_right = right.shape[0]
            ),
        });
    }

    // Use the tensor's actual strides — avoids a full O(N) make_contiguous() copy
    // for transposed inputs (common in backward passes: grad_weight = input^T @ grad).
    let left_data = &left.data;
    let right_data = &right.data;
    // Strides are in elements. Offset into the underlying data buffer.
    let left_ptr_offset = left.offset;
    let right_ptr_offset = right.offset;

    // Row and column strides for the gemm call.
    // For a 2D tensor with strides [row_stride, col_stride]:
    //   strides[0] = row stride (elements between rows)
    //   strides[1] = col stride (elements between columns, usually 1)
    let (lhs_rs, lhs_cs) = (left.strides[0] as isize, left.strides[1] as isize);
    let (rhs_rs, rhs_cs) = (right.strides[0] as isize, right.strides[1] as isize);

    let mut out = vec![0.0_f32; m * n];

    // gemm crate: cache-aware SIMD GEMM with native Rayon threading.
    // API: dst = alpha * dst + beta * lhs @ rhs
    //   alpha=0, beta=1 → dst = lhs @ rhs (no accumulation)
    // Output is always row-major (stride = n for rows, 1 for cols).
    #[cfg(feature = "parallel")]
    let parallelism = {
        let flops = 2 * m * k * n;
        const PAR_FLOP_THRESHOLD: usize = 1 << 17; // 128k FLOPs
        if flops >= PAR_FLOP_THRESHOLD {
            Parallelism::Rayon(rayon::current_num_threads())
        } else {
            Parallelism::None
        }
    };
    #[cfg(not(feature = "parallel"))]
    let parallelism = gemm::Parallelism::None;

    // Safety: pointers are valid, lengths verified above; out is zero-initialised.
    #[allow(unsafe_code)]
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,          // dst_cs: col stride = 1 (row-major output)
            n as isize, // dst_rs: row stride = n
            false,
            left_data.as_ptr().add(left_ptr_offset),
            lhs_cs, // lhs col stride (from tensor metadata)
            lhs_rs, // lhs row stride
            right_data.as_ptr().add(right_ptr_offset),
            rhs_cs, // rhs col stride
            rhs_rs, // rhs row stride
            0.0_f32,
            1.0_f32,
            false,
            false,
            false,
            parallelism,
        );
    }

    Tensor::new(vec![m, n], out)
}

pub fn concat(tensors: &[Tensor], axis: usize) -> Result<Tensor, TensorError> {
    if tensors.is_empty() {
        return Err(TensorError {
            message: "Empty list in concat".to_string(),
        });
    }
    let mut out_shape = tensors[0].shape.clone();
    let mut axis_sum = 0;
    let mut contigs = Vec::with_capacity(tensors.len());
    for t in tensors {
        axis_sum += t.shape[axis];
        contigs.push(t.make_contiguous()?);
    }
    out_shape[axis] = axis_sum;
    let (outer, inner) = (
        product_prefix(&out_shape, axis)?,
        product_suffix(&out_shape, axis + 1)?,
    );
    let total_elements: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total_elements);
    for o in 0..outer {
        for t in &contigs {
            let dim = t.shape[axis];
            out.extend_from_slice(&t.data[o * dim * inner..(o + 1) * dim * inner]);
        }
    }
    Tensor::new(out_shape, out)
}

pub fn gather(tensor: &Tensor, indices: &[usize], axis: usize) -> Result<Tensor, TensorError> {
    let contig = tensor.make_contiguous()?;
    let axis_dim = contig.shape[axis];
    let mut out_shape = contig.shape.clone();
    out_shape[axis] = indices.len();
    let (outer, inner) = (
        product_prefix(&contig.shape, axis)?,
        product_suffix(&contig.shape, axis + 1)?,
    );
    let total_elements: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total_elements);
    for o in 0..outer {
        for &idx in indices {
            let start = o * axis_dim * inner + idx * inner;
            out.extend_from_slice(&contig.data[start..start + inner]);
        }
    }
    Tensor::new(out_shape, out)
}

pub fn log_elementwise(tensor: &Tensor) -> Result<Tensor, TensorError> {
    let contig = tensor.make_contiguous()?;
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
        if should_par(contig.data.len()) {
            contig.data.par_iter().map(f).collect::<Result<_, _>>()?
        } else {
            contig.data.iter().map(f).collect::<Result<_, _>>()?
        }
        #[cfg(not(feature = "parallel"))]
        contig.data.iter().map(f).collect::<Result<_, _>>()?
    };
    Tensor::new(tensor.shape.clone(), out)
}

pub fn exp_elementwise(tensor: &Tensor) -> Result<Tensor, TensorError> {
    let contig = tensor.make_contiguous()?;
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
        if crate::engine::ir::kernels::utils::should_par(contig.data.len()) {
            contig.data.par_iter().map(f).collect::<Result<_, _>>()?
        } else {
            contig.data.iter().map(f).collect::<Result<_, _>>()?
        }
        #[cfg(not(feature = "parallel"))]
        contig.data.iter().map(f).collect::<Result<_, _>>()?
    };
    Tensor::new(tensor.shape.clone(), out)
}

pub fn neg_elementwise(tensor: &Tensor) -> Result<Tensor, TensorError> {
    let contig = tensor.make_contiguous()?;
    let out: Vec<f32> = {
        let f = |&v: &f32| -> f32 { -v };
        #[cfg(feature = "parallel")]
        if crate::engine::ir::kernels::utils::should_par(contig.data.len()) {
            contig.data.par_iter().map(f).collect()
        } else {
            contig.data.iter().map(f).collect()
        }
        #[cfg(not(feature = "parallel"))]
        contig.data.iter().map(f).collect()
    };
    Tensor::new(tensor.shape.clone(), out)
}

/// Embedding lookup: weight [vocab_size, embed_dim], indices [batch] or [batch, seq_len] (stored as f32 integers).
/// Output shape: [*indices_shape, embed_dim].
pub fn embedding(weight: &Tensor, indices: &Tensor) -> Result<Tensor, TensorError> {
    let w = weight.make_contiguous()?;
    let idx = indices.make_contiguous()?;
    if w.shape.len() != 2 {
        return Err(TensorError {
            message: "Embedding weight must be rank-2 [vocab_size, embed_dim]".to_string(),
        });
    }
    let (vocab_size, embed_dim) = (w.shape[0], w.shape[1]);
    let num_tokens = idx.data.len();
    let mut out = vec![0.0_f32; num_tokens * embed_dim];
    for (t, &raw_idx) in idx.data.iter().enumerate() {
        let i = raw_idx as usize;
        if i >= vocab_size {
            return Err(TensorError {
                message: format!("Embedding index {i} out of vocab_size {vocab_size}"),
            });
        }
        out[t * embed_dim..(t + 1) * embed_dim]
            .copy_from_slice(&w.data[i * embed_dim..(i + 1) * embed_dim]);
    }
    let mut out_shape = idx.shape.clone();
    out_shape.push(embed_dim);
    Tensor::new(out_shape, out)
}

/// Sparse embedding backward: accumulates upstream gradient into weight rows indicated by indices.
/// Returns dense gradient of shape [vocab_size, embed_dim].
pub fn embedding_backward(
    weight: &Tensor,
    indices: &Tensor,
    upstream: &Tensor,
) -> Result<Tensor, TensorError> {
    let w = weight.make_contiguous()?;
    let idx = indices.make_contiguous()?;
    let up = upstream.make_contiguous()?;
    if w.shape.len() != 2 {
        return Err(TensorError {
            message: "Embedding weight must be rank-2".to_string(),
        });
    }
    let (vocab_size, embed_dim) = (w.shape[0], w.shape[1]);
    let num_tokens = idx.data.len();
    if up.data.len() != num_tokens * embed_dim {
        return Err(TensorError {
            message: format!(
                "EmbeddingBackward upstream size mismatch: expected {} got {}",
                num_tokens * embed_dim,
                up.data.len()
            ),
        });
    }
    let mut dw = vec![0.0_f32; vocab_size * embed_dim];
    for (t, &raw_idx) in idx.data.iter().enumerate() {
        let i = raw_idx as usize;
        if i >= vocab_size {
            return Err(TensorError {
                message: format!("Embedding index {i} out of vocab_size {vocab_size}"),
            });
        }
        for d in 0..embed_dim {
            dw[i * embed_dim + d] += up.data[t * embed_dim + d];
        }
    }
    Tensor::new(vec![vocab_size, embed_dim], dw)
}
