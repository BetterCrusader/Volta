use crate::engine::ir::kernels::utils::{product_prefix, product_suffix, should_par};
use crate::engine::ir::tensor::{Tensor, TensorError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn reduce_sum(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, TensorError> {
    let contig = tensor.make_contiguous()?;
    if contig.data.is_empty() {
        return Err(TensorError {
            message: "Empty tensor in reduce_sum".to_string(),
        });
    }
    match axis {
        None => {
            #[cfg(feature = "parallel")]
            let sum: f32 = if should_par(contig.data.len()) {
                contig.data.par_iter().copied().sum()
            } else {
                contig.data.iter().copied().sum()
            };
            #[cfg(not(feature = "parallel"))]
            let sum: f32 = contig.data.iter().copied().sum();
            Ok(Tensor::scalar(sum))
        }
        Some(a) => {
            let rank = contig.shape.len();
            if a >= rank {
                return Err(TensorError {
                    message: "Axis out of bounds in reduce_sum".to_string(),
                });
            }
            let outer = product_prefix(&contig.shape, a)?;
            let axis_dim = contig.shape[a];
            let inner = product_suffix(&contig.shape, a + 1)?;
            let mut out_shape = contig.shape.clone();
            out_shape.remove(a);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            let mut out = vec![0.0_f32; outer * inner];
            for o in 0..outer {
                for i in 0..inner {
                    let mut acc = 0.0_f32;
                    for k in 0..axis_dim {
                        acc += contig.data[o * axis_dim * inner + k * inner + i];
                    }
                    out[o * inner + i] = acc;
                }
            }
            Tensor::new(out_shape, out)
        }
    }
}

pub fn reduce_mean(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, TensorError> {
    let contig = tensor.make_contiguous()?;
    if contig.data.is_empty() {
        return Err(TensorError {
            message: "Empty tensor in reduce_mean".to_string(),
        });
    }
    match axis {
        None => {
            let sum = reduce_sum(&contig, None)?;
            sum.scale(1.0 / contig.logical_len() as f32)
        }
        Some(a) => {
            let axis_dim = contig.shape[a];
            if axis_dim == 0 {
                return Err(TensorError {
                    message: "Zero-sized axis in reduce_mean".to_string(),
                });
            }
            reduce_sum(&contig, Some(a))?.scale(1.0 / axis_dim as f32)
        }
    }
}

pub fn reduce_max(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, TensorError> {
    let contig = tensor.make_contiguous()?;
    if contig.data.is_empty() {
        return Err(TensorError {
            message: "Empty tensor in reduce_max".to_string(),
        });
    }
    match axis {
        None => Ok(Tensor::scalar(
            contig
                .data
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
        )),
        Some(a) => {
            let outer = product_prefix(&contig.shape, a)?;
            let axis_dim = contig.shape[a];
            let inner = product_suffix(&contig.shape, a + 1)?;
            let mut out_shape = contig.shape.clone();
            out_shape.remove(a);
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            let mut out = vec![f32::NEG_INFINITY; outer * inner];
            for o in 0..outer {
                for i in 0..inner {
                    for k in 0..axis_dim {
                        let val = contig.data[o * axis_dim * inner + k * inner + i];
                        if val > out[o * inner + i] {
                            out[o * inner + i] = val;
                        }
                    }
                }
            }
            Tensor::new(out_shape, out)
        }
    }
}

pub fn reduce_max_backward(
    input: &Tensor,
    output_max: &Tensor,
    upstream: &Tensor,
    axis: Option<usize>,
) -> Result<Tensor, TensorError> {
    let input_c = input.make_contiguous()?;
    let output_max_c = output_max.make_contiguous()?;
    let upstream_c = upstream.make_contiguous()?;

    let mut grad = vec![0.0_f32; input_c.logical_len()];

    match axis {
        None => {
            let max_val = output_max_c.data[0];
            let up = upstream_c.data[0];
            let mut count = 0;
            for &v in input_c.data.iter() {
                if v == max_val {
                    count += 1;
                }
            }
            if count > 0 {
                let g = up / count as f32;
                for (i, &v) in input_c.data.iter().enumerate() {
                    if v == max_val {
                        grad[i] = g;
                    }
                }
            }
        }
        Some(a) => {
            let rank = input_c.shape.len();
            if a >= rank {
                return Err(TensorError {
                    message: "Axis out of bounds in reduce_max_backward".to_string(),
                });
            }
            let outer = product_prefix(&input_c.shape, a)?;
            let axis_dim = input_c.shape[a];
            let inner = product_suffix(&input_c.shape, a + 1)?;

            for o in 0..outer {
                for i in 0..inner {
                    let max_val = output_max_c.data[o * inner + i];
                    let up = upstream_c.data[o * inner + i];

                    let mut count = 0;
                    for k in 0..axis_dim {
                        if input_c.data[o * axis_dim * inner + k * inner + i] == max_val {
                            count += 1;
                        }
                    }

                    if count > 0 {
                        let g = up / count as f32;
                        for k in 0..axis_dim {
                            let idx = o * axis_dim * inner + k * inner + i;
                            if input_c.data[idx] == max_val {
                                grad[idx] = g;
                            }
                        }
                    }
                }
            }
        }
    }
    Tensor::new(input_c.shape.clone(), grad)
}

pub fn softmax(tensor: &Tensor) -> Result<Tensor, TensorError> {
    let contig = tensor.make_contiguous()?;
    if contig.shape.len() == 1 {
        let max = contig
            .data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = contig.data.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for v in &mut exps {
            *v /= sum;
        }
        Tensor::new(contig.shape.clone(), exps)
    } else if contig.shape.len() == 2 {
        let row_len = contig.shape[1];
        let mut out = Vec::with_capacity(contig.logical_len());
        for row in contig.data.chunks(row_len) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let row_exps: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = row_exps.iter().sum();
            for e in row_exps {
                out.push(e / sum);
            }
        }
        Tensor::new(contig.shape.clone(), out)
    } else {
        Err(TensorError {
            message: "Softmax expects 1D or 2D".to_string(),
        })
    }
}
