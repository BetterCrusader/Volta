use crate::engine::ir::tensor::{Tensor, TensorError};

pub fn layer_norm(input: &Tensor, weight: &Tensor, bias: &Tensor, epsilon: f32) -> Result<Tensor, TensorError> {
    let input_c = input.make_contiguous()?;
    let w = weight.make_contiguous()?;
    let b = bias.make_contiguous()?;
    if input_c.shape.len() != 2 { return Err(TensorError { message: "Layer_norm expects [N, D]".to_string() }); }
    let (n, d) = (input_c.shape[0], input_c.shape[1]);
    if d == 0 { return Err(TensorError { message: "D must be > 0".to_string() }); }
    if w.shape != vec![d] || b.shape != vec![d] { return Err(TensorError { message: "Weight/bias mismatch in layer_norm".to_string() }); }

    let mut out = vec![0.0_f32; n * d];
    for i in 0..n {
        let row = &input_c.data[i * d..(i + 1) * d];
        let mean: f64 = row.iter().copied().map(|x| x as f64).sum::<f64>() / d as f64;
        let var: f64 = row.iter().copied().map(|x| { let diff = x as f64 - mean; diff * diff }).sum::<f64>() / d as f64;
        let inv_std = 1.0_f64 / (var + epsilon as f64).sqrt();
        for j in 0..d {
            let x_hat = (row[j] as f64 - mean) * inv_std;
            out[i * d + j] = (w.data[j] as f64 * x_hat + b.data[j] as f64) as f32;
        }
    }
    Tensor::new(input_c.shape.clone(), out)
}

pub fn batch_norm_nchw(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) -> Result<Tensor, TensorError> {
    if input.shape.len() != 4 {
        return Err(TensorError { message: format!("BatchNorm expects rank-4 NCHW input, got {:?}", input.shape) });
    }
    let input_c = input.make_contiguous()?;
    let weight_c = weight.make_contiguous()?;
    let bias_c = bias.make_contiguous()?;
    let mean_c = mean.make_contiguous()?;
    let var_c = var.make_contiguous()?;

    let (n, c, h, w) = (input_c.shape[0], input_c.shape[1], input_c.shape[2], input_c.shape[3]);
    if weight_c.shape != vec![c] || bias_c.shape != vec![c] || mean_c.shape != vec![c] || var_c.shape != vec![c] {
        return Err(TensorError { message: "BatchNorm weight/bias/mean/var shape mismatch".to_string() });
    }

    let mut out = vec![0.0_f32; input_c.logical_len()];
    for ni in 0..n {
        for ci in 0..c {
            let gamma = weight_c.data[ci] as f64;
            let beta = bias_c.data[ci] as f64;
            let m = mean_c.data[ci] as f64;
            let v = var_c.data[ci] as f64;
            let inv_std = 1.0 / (v + epsilon as f64).sqrt();

            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    let x_hat = (input_c.data[idx] as f64 - m) * inv_std;
                    out[idx] = (gamma * x_hat + beta) as f32;
                }
            }
        }
    }
    Tensor::new(input_c.shape.clone(), out)
}

pub fn batch_norm_backward_input_nchw(
    input: &Tensor,
    upstream: &Tensor,
    weight: &Tensor,
    var: &Tensor,
    epsilon: f32,
) -> Result<Tensor, TensorError> {
    if input.shape.len() != 4 { return Err(TensorError { message: "BatchNormBackwardInput expects rank-4".to_string() }); }
    if upstream.shape != input.shape { return Err(TensorError { message: "Shape mismatch in BatchNormBackwardInput".to_string() }); }
    let input_c = input.make_contiguous()?;
    let upstream_c = upstream.make_contiguous()?;
    let weight_c = weight.make_contiguous()?;
    let var_c = var.make_contiguous()?;

    let (n, c, h, w) = (input_c.shape[0], input_c.shape[1], input_c.shape[2], input_c.shape[3]);
    let mut grad = vec![0.0_f32; input_c.logical_len()];
    for ni in 0..n {
        for ci in 0..c {
            let scale = weight_c.data[ci] / (var_c.data[ci] + epsilon).sqrt();
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    grad[idx] = upstream_c.data[idx] * scale;
                }
            }
        }
    }
    Tensor::new(input_c.shape.clone(), grad)
}

pub fn batch_norm_backward_weight_nchw(
    input: &Tensor,
    upstream: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) -> Result<Tensor, TensorError> {
    if input.shape.len() != 4 { return Err(TensorError { message: "BatchNormBackwardWeight expects rank-4".to_string() }); }
    if upstream.shape != input.shape { return Err(TensorError { message: "Shape mismatch in BatchNormBackwardWeight".to_string() }); }
    let input_c = input.make_contiguous()?;
    let upstream_c = upstream.make_contiguous()?;
    let mean_c = mean.make_contiguous()?;
    let var_c = var.make_contiguous()?;

    let (n, c, h, w) = (input_c.shape[0], input_c.shape[1], input_c.shape[2], input_c.shape[3]);
    let mut grad = vec![0.0_f32; c];
    for ci in 0..c {
        let inv_std = 1.0_f64 / (var_c.data[ci] as f64 + epsilon as f64).sqrt();
        let mut acc = 0.0_f64;
        let m = mean_c.data[ci] as f64;
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    acc += upstream_c.data[idx] as f64 * (input_c.data[idx] as f64 - m) * inv_std;
                }
            }
        }
        grad[ci] = acc as f32;
    }
    Tensor::new(vec![c], grad)
}

pub fn batch_norm_backward_bias_nchw(
    upstream: &Tensor,
) -> Result<Tensor, TensorError> {
    if upstream.shape.len() != 4 { return Err(TensorError { message: "BatchNormBackwardBias expects rank-4".to_string() }); }
    let upstream_c = upstream.make_contiguous()?;
    let (n, c, h, w) = (upstream_c.shape[0], upstream_c.shape[1], upstream_c.shape[2], upstream_c.shape[3]);
    let mut grad = vec![0.0_f32; c];
    for ci in 0..c {
        let mut acc = 0.0_f64;
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    acc += upstream_c.data[idx] as f64;
                }
            }
        }
        grad[ci] = acc as f32;
    }
    Tensor::new(vec![c], grad)
}

pub fn layer_norm_backward_input(input: &Tensor, dy: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor, TensorError> {
    let input_c = input.make_contiguous()?;
    let dy_c = dy.make_contiguous()?;
    let w = weight.make_contiguous()?;
    if input_c.shape.len() != 2 || input_c.shape != dy_c.shape { return Err(TensorError { message: "Shape mismatch in layer_norm_backward_input".to_string() }); }
    let (n, d) = (input_c.shape[0], input_c.shape[1]);
    let inv_d = 1.0_f64 / d as f64;
    let mut out = vec![0.0_f32; n * d];

    for i in 0..n {
        let x_row = &input_c.data[i * d..(i + 1) * d];
        let dy_row = &dy_c.data[i * d..(i + 1) * d];
        let mean: f64 = x_row.iter().copied().map(|v| v as f64).sum::<f64>() * inv_d;
        let var: f64 = x_row.iter().copied().map(|v| { let diff = v as f64 - mean; diff * diff }).sum::<f64>() * inv_d;
        let inv_std = 1.0_f64 / (var + epsilon as f64).sqrt();
        let g: Vec<f64> = dy_row.iter().zip(w.data.iter()).map(|(&dy_j, &gam_j)| dy_j as f64 * gam_j as f64).collect();
        let x_hat: Vec<f64> = x_row.iter().map(|&x_j| (x_j as f64 - mean) * inv_std).collect();
        let sum_g: f64 = g.iter().sum();
        let sum_g_x_hat: f64 = g.iter().zip(x_hat.iter()).map(|(g_j, xh_j)| g_j * xh_j).sum();
        for j in 0..d {
            out[i * d + j] = (inv_d * inv_std * (d as f64 * g[j] - sum_g - x_hat[j] * sum_g_x_hat)) as f32;
        }
    }
    Tensor::new(input_c.shape.clone(), out)
}

pub fn layer_norm_backward_weight(input: &Tensor, dy: &Tensor, epsilon: f32) -> Result<Tensor, TensorError> {
    let input_c = input.make_contiguous()?;
    let dy_c = dy.make_contiguous()?;
    let (n, d) = (input_c.shape[0], input_c.shape[1]);
    let inv_d = 1.0_f64 / d as f64;
    let mut d_gamma = vec![0.0_f64; d];
    for i in 0..n {
        let x_row = &input_c.data[i * d..(i + 1) * d];
        let dy_row = &dy_c.data[i * d..(i + 1) * d];
        let mean: f64 = x_row.iter().copied().map(|v| v as f64).sum::<f64>() * inv_d;
        let var: f64 = x_row.iter().copied().map(|v| { let diff = v as f64 - mean; diff * diff }).sum::<f64>() * inv_d;
        let inv_std = 1.0_f64 / (var + epsilon as f64).sqrt();
        for j in 0..d { d_gamma[j] += dy_row[j] as f64 * ((x_row[j] as f64 - mean) * inv_std); }
    }
    Tensor::new(vec![d], d_gamma.into_iter().map(|v| v as f32).collect())
}

pub fn layer_norm_backward_bias(dy: &Tensor) -> Result<Tensor, TensorError> {
    let dy_c = dy.make_contiguous()?;
    let (n, d) = (dy_c.shape[0], dy_c.shape[1]);
    let mut d_beta = vec![0.0_f32; d];
    for i in 0..n { for j in 0..d { d_beta[j] += dy_c.data[i * d + j]; } }
    Tensor::new(vec![d], d_beta)
}

/// GroupNorm: input [N, C, *spatial], num_groups divides C.
/// weight [C], bias [C]. Normalizes within each (n, g) group of C/num_groups channels.
pub fn group_norm(input: &Tensor, weight: &Tensor, bias: &Tensor, num_groups: usize, epsilon: f32) -> Result<Tensor, TensorError> {
    let inp = input.make_contiguous()?;
    let w = weight.make_contiguous()?;
    let b = bias.make_contiguous()?;
    if inp.shape.len() < 2 {
        return Err(TensorError { message: "GroupNorm expects rank >= 2".to_string() });
    }
    let n = inp.shape[0];
    let c = inp.shape[1];
    if c % num_groups != 0 {
        return Err(TensorError { message: format!("GroupNorm: C={c} not divisible by num_groups={num_groups}") });
    }
    if w.shape != vec![c] || b.shape != vec![c] {
        return Err(TensorError { message: "GroupNorm weight/bias must have shape [C]".to_string() });
    }
    let spatial: usize = inp.shape[2..].iter().product::<usize>().max(1);
    let channels_per_group = c / num_groups;
    let group_size = channels_per_group * spatial;
    let mut out = vec![0.0_f32; inp.data.len()];

    for ni in 0..n {
        for g in 0..num_groups {
            // Compute mean and variance over the group
            let mut sum = 0.0_f64;
            for ci in 0..channels_per_group {
                let ch = g * channels_per_group + ci;
                let base = (ni * c + ch) * spatial;
                for s in 0..spatial {
                    sum += inp.data[base + s] as f64;
                }
            }
            let mean = sum / group_size as f64;
            let mut var_sum = 0.0_f64;
            for ci in 0..channels_per_group {
                let ch = g * channels_per_group + ci;
                let base = (ni * c + ch) * spatial;
                for s in 0..spatial {
                    let diff = inp.data[base + s] as f64 - mean;
                    var_sum += diff * diff;
                }
            }
            let inv_std = 1.0_f64 / (var_sum / group_size as f64 + epsilon as f64).sqrt();
            // Normalize and apply affine transform
            for ci in 0..channels_per_group {
                let ch = g * channels_per_group + ci;
                let base = (ni * c + ch) * spatial;
                let gamma = w.data[ch] as f64;
                let beta = b.data[ch] as f64;
                for s in 0..spatial {
                    let x_hat = (inp.data[base + s] as f64 - mean) * inv_std;
                    out[base + s] = (gamma * x_hat + beta) as f32;
                }
            }
        }
    }
    Tensor::new(inp.shape.clone(), out)
}

/// GroupNorm backward for input gradient.
pub fn group_norm_backward_input(
    input: &Tensor, dy: &Tensor, weight: &Tensor, num_groups: usize, epsilon: f32
) -> Result<Tensor, TensorError> {
    let inp = input.make_contiguous()?;
    let dy_c = dy.make_contiguous()?;
    let w = weight.make_contiguous()?;
    if inp.shape.len() < 2 || inp.shape != dy_c.shape {
        return Err(TensorError { message: "Shape mismatch in group_norm_backward_input".to_string() });
    }
    let n = inp.shape[0];
    let c = inp.shape[1];
    let spatial: usize = inp.shape[2..].iter().product::<usize>().max(1);
    let channels_per_group = c / num_groups;
    let group_size = channels_per_group * spatial;
    let mut grad = vec![0.0_f32; inp.data.len()];

    for ni in 0..n {
        for g in 0..num_groups {
            // Recompute mean and var
            let mut sum = 0.0_f64;
            for ci in 0..channels_per_group {
                let ch = g * channels_per_group + ci;
                let base = (ni * c + ch) * spatial;
                for s in 0..spatial { sum += inp.data[base + s] as f64; }
            }
            let mean = sum / group_size as f64;
            let mut var_sum = 0.0_f64;
            for ci in 0..channels_per_group {
                let ch = g * channels_per_group + ci;
                let base = (ni * c + ch) * spatial;
                for s in 0..spatial {
                    let diff = inp.data[base + s] as f64 - mean;
                    var_sum += diff * diff;
                }
            }
            let inv_std = 1.0_f64 / (var_sum / group_size as f64 + epsilon as f64).sqrt();
            // x_hat for each element
            // Compute sum_dy_gamma and sum_dy_gamma_x_hat
            let mut sum_dg = 0.0_f64;
            let mut sum_dg_xhat = 0.0_f64;
            for ci in 0..channels_per_group {
                let ch = g * channels_per_group + ci;
                let base = (ni * c + ch) * spatial;
                let gamma = w.data[ch] as f64;
                for s in 0..spatial {
                    let x_hat = (inp.data[base + s] as f64 - mean) * inv_std;
                    let dg = dy_c.data[base + s] as f64 * gamma;
                    sum_dg += dg;
                    sum_dg_xhat += dg * x_hat;
                }
            }
            let inv_m = 1.0_f64 / group_size as f64;
            for ci in 0..channels_per_group {
                let ch = g * channels_per_group + ci;
                let base = (ni * c + ch) * spatial;
                let gamma = w.data[ch] as f64;
                for s in 0..spatial {
                    let x_hat = (inp.data[base + s] as f64 - mean) * inv_std;
                    let dg = dy_c.data[base + s] as f64 * gamma;
                    grad[base + s] = (inv_std * (dg - inv_m * sum_dg - x_hat * inv_m * sum_dg_xhat)) as f32;
                }
            }
        }
    }
    Tensor::new(inp.shape.clone(), grad)
}

/// GroupNorm backward for weight (gamma) gradient: shape [C].
pub fn group_norm_backward_weight(input: &Tensor, dy: &Tensor, num_groups: usize, epsilon: f32) -> Result<Tensor, TensorError> {
    let inp = input.make_contiguous()?;
    let dy_c = dy.make_contiguous()?;
    if inp.shape.len() < 2 || inp.shape != dy_c.shape {
        return Err(TensorError { message: "Shape mismatch in group_norm_backward_weight".to_string() });
    }
    let n = inp.shape[0];
    let c = inp.shape[1];
    let spatial: usize = inp.shape[2..].iter().product::<usize>().max(1);
    let channels_per_group = c / num_groups;
    let group_size = channels_per_group * spatial;
    let mut d_gamma = vec![0.0_f64; c];

    for ni in 0..n {
        for g in 0..num_groups {
            let mut sum = 0.0_f64;
            for ci in 0..channels_per_group {
                let ch = g * channels_per_group + ci;
                let base = (ni * c + ch) * spatial;
                for s in 0..spatial { sum += inp.data[base + s] as f64; }
            }
            let mean = sum / group_size as f64;
            let mut var_sum = 0.0_f64;
            for ci in 0..channels_per_group {
                let ch = g * channels_per_group + ci;
                let base = (ni * c + ch) * spatial;
                for s in 0..spatial {
                    let d = inp.data[base + s] as f64 - mean;
                    var_sum += d * d;
                }
            }
            let inv_std = 1.0_f64 / (var_sum / group_size as f64 + epsilon as f64).sqrt();
            for ci in 0..channels_per_group {
                let ch = g * channels_per_group + ci;
                let base = (ni * c + ch) * spatial;
                for s in 0..spatial {
                    let x_hat = (inp.data[base + s] as f64 - mean) * inv_std;
                    d_gamma[ch] += dy_c.data[base + s] as f64 * x_hat;
                }
            }
        }
    }
    Tensor::new(vec![c], d_gamma.into_iter().map(|v| v as f32).collect())
}

/// GroupNorm backward for bias (beta) gradient: shape [C].
pub fn group_norm_backward_bias(dy: &Tensor) -> Result<Tensor, TensorError> {
    let dy_c = dy.make_contiguous()?;
    if dy_c.shape.len() < 2 {
        return Err(TensorError { message: "GroupNorm backward bias expects rank >= 2".to_string() });
    }
    let n = dy_c.shape[0];
    let c = dy_c.shape[1];
    let spatial: usize = dy_c.shape[2..].iter().product::<usize>().max(1);
    let mut d_beta = vec![0.0_f64; c];
    for ni in 0..n {
        for ch in 0..c {
            let base = (ni * c + ch) * spatial;
            for s in 0..spatial { d_beta[ch] += dy_c.data[base + s] as f64; }
        }
    }
    Tensor::new(vec![c], d_beta.into_iter().map(|v| v as f32).collect())
}

/// InstanceNorm: input [N, C, *spatial], weight [C], bias [C].
/// Normalizes each (n, c) slice independently across spatial dims.
pub fn instance_norm(input: &Tensor, weight: &Tensor, bias: &Tensor, epsilon: f32) -> Result<Tensor, TensorError> {
    // InstanceNorm = GroupNorm with num_groups = C (one group per channel)
    let c = if input.shape.len() >= 2 { input.shape[1] } else {
        return Err(TensorError { message: "InstanceNorm expects rank >= 2".to_string() });
    };
    group_norm(input, weight, bias, c, epsilon)
}

/// InstanceNorm backward for input gradient.
pub fn instance_norm_backward_input(input: &Tensor, dy: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor, TensorError> {
    let c = if input.shape.len() >= 2 { input.shape[1] } else {
        return Err(TensorError { message: "InstanceNorm expects rank >= 2".to_string() });
    };
    group_norm_backward_input(input, dy, weight, c, epsilon)
}

/// InstanceNorm backward for weight gradient.
pub fn instance_norm_backward_weight(input: &Tensor, dy: &Tensor, epsilon: f32) -> Result<Tensor, TensorError> {
    let c = if input.shape.len() >= 2 { input.shape[1] } else {
        return Err(TensorError { message: "InstanceNorm expects rank >= 2".to_string() });
    };
    group_norm_backward_weight(input, dy, c, epsilon)
}

/// InstanceNorm backward for bias gradient.
pub fn instance_norm_backward_bias(dy: &Tensor) -> Result<Tensor, TensorError> {
    group_norm_backward_bias(dy)
}
