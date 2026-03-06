use crate::engine::ir::tensor::{Tensor, TensorError};

pub fn conv2d(input: &Tensor, weight: &Tensor) -> Result<Tensor, TensorError> {
    if input.shape.len() != 2 || weight.shape.len() != 2 {
        return Err(TensorError {
            message: "Conv2D expects 2D input and 2D kernel".to_string(),
        });
    }

    let input_c = input.make_contiguous()?;
    let weight_c = weight.make_contiguous()?;

    let in_h = input_c.shape[0];
    let in_w = input_c.shape[1];
    let k_h = weight_c.shape[0];
    let k_w = weight_c.shape[1];
    if k_h == 0 || k_w == 0 || k_h > in_h || k_w > in_w {
        return Err(TensorError {
            message: format!(
                "Shape mismatch in Conv2D: input {:?}, kernel {:?}",
                input_c.shape, weight_c.shape
            ),
        });
    }

    let out_h = in_h - k_h + 1;
    let out_w = in_w - k_w + 1;
    let mut out = vec![0.0_f32; out_h * out_w];
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut acc = 0.0_f32;
            for ky in 0..k_h {
                for kx in 0..k_w {
                    let in_index = (oy + ky) * in_w + (ox + kx);
                    let kernel_index = ky * k_w + kx;
                    acc += input_c.data[in_index] * weight_c.data[kernel_index];
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }

    Tensor::new(vec![out_h, out_w], out)
}

pub fn pool2d_nchw(
    input: &Tensor,
    kernel_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    is_max: bool,
) -> Result<Tensor, TensorError> {
    if input.shape.len() != 4 {
        return Err(TensorError {
            message: format!("Pool expects rank-4 NCHW input, got {:?}", input.shape),
        });
    }
    if kernel_shape.len() != 2 {
        return Err(TensorError {
            message: "Pool kernel_shape must have 2 values".to_string(),
        });
    }

    let input_c = input.make_contiguous()?;
    let (n, c, h, w) = (
        input_c.shape[0],
        input_c.shape[1],
        input_c.shape[2],
        input_c.shape[3],
    );
    let kh = kernel_shape[0];
    let kw = kernel_shape[1];
    if kh == 0 || kw == 0 {
        return Err(TensorError {
            message: "Pool kernel values must be > 0".to_string(),
        });
    }
    let sh = strides.first().copied().unwrap_or(1);
    let sw = strides.get(1).copied().unwrap_or(1);
    if sh == 0 || sw == 0 {
        return Err(TensorError {
            message: "Pool strides must be > 0".to_string(),
        });
    }
    let (pt, pl, pb, pr) = if pads.is_empty() {
        (0_usize, 0_usize, 0_usize, 0_usize)
    } else if pads.len() == 4 {
        (pads[0], pads[1], pads[2], pads[3])
    } else {
        return Err(TensorError {
            message: "Pool pads must be empty or [top,left,bottom,right]".to_string(),
        });
    };

    let h_padded = h + pt + pb;
    let w_padded = w + pl + pr;
    if h_padded < kh || w_padded < kw {
        return Err(TensorError {
            message: format!(
                "Pool kernel {:?} too large for padded input {:?}",
                kernel_shape,
                [n, c, h_padded, w_padded]
            ),
        });
    }

    let out_h = (h_padded - kh) / sh + 1;
    let out_w = (w_padded - kw) / sw + 1;
    let mut out = vec![0.0_f32; n * c * out_h * out_w];

    for ni in 0..n {
        for ci in 0..c {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let y0 = oy * sh;
                    let x0 = ox * sw;
                    let mut acc = if is_max { f32::NEG_INFINITY } else { 0.0 };
                    let mut count = 0_usize;

                    for ky in 0..kh {
                        for kx in 0..kw {
                            let py = y0 + ky;
                            let px = x0 + kx;
                            if py < pt || py >= pt + h || px < pl || px >= pl + w {
                                continue;
                            }
                            let iy = py - pt;
                            let ix = px - pl;
                            let idx = ((ni * c + ci) * h + iy) * w + ix;
                            let v = input_c.data[idx];
                            if is_max {
                                if v > acc {
                                    acc = v;
                                }
                            } else {
                                acc += v;
                            }
                            count += 1;
                        }
                    }

                    let out_val = if is_max {
                        if count == 0 { 0.0 } else { acc }
                    } else if count == 0 {
                        0.0
                    } else {
                        acc / (count as f32)
                    };

                    let out_idx = ((ni * c + ci) * out_h + oy) * out_w + ox;
                    out[out_idx] = out_val;
                }
            }
        }
    }

    Tensor::new(vec![n, c, out_h, out_w], out)
}

pub fn max_pool2d_backward_nchw(
    input: &Tensor,
    upstream: &Tensor,
    kernel_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
) -> Result<Tensor, TensorError> {
    if input.shape.len() != 4 {
        return Err(TensorError {
            message: "MaxPoolBackward expects rank-4".to_string(),
        });
    }
    let input_c = input.make_contiguous()?;
    let upstream_c = upstream.make_contiguous()?;

    let (n, c, h, w) = (
        input_c.shape[0],
        input_c.shape[1],
        input_c.shape[2],
        input_c.shape[3],
    );
    let kh = kernel_shape[0];
    let kw = kernel_shape[1];
    let sh = strides.first().copied().unwrap_or(1);
    let sw = strides.get(1).copied().unwrap_or(1);
    let (pt, pl, _pb, _pr) = if pads.is_empty() {
        (0, 0, 0, 0)
    } else {
        (pads[0], pads[1], pads[2], pads[3])
    };

    let h_padded = h + pt + (if pads.len() == 4 { pads[2] } else { 0 });
    let w_padded = w + pl + (if pads.len() == 4 { pads[3] } else { 0 });
    let out_h = (h_padded - kh) / sh + 1;
    let out_w = (w_padded - kw) / sw + 1;

    let mut grad = vec![0.0_f32; input_c.logical_len()];
    for ni in 0..n {
        for ci in 0..c {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let dy = upstream_c.data[((ni * c + ci) * out_h + oy) * out_w + ox];
                    let y0 = oy * sh;
                    let x0 = ox * sw;
                    let (mut max_v, mut max_idx) = (f32::NEG_INFINITY, None);

                    for ky in 0..kh {
                        for kx in 0..kw {
                            let (py, px) = (y0 + ky, x0 + kx);
                            if py >= pt && py < pt + h && px >= pl && px < pl + w {
                                let (iy, ix) = (py - pt, px - pl);
                                let idx = ((ni * c + ci) * h + iy) * w + ix;
                                let v = input_c.data[idx];
                                if v > max_v {
                                    max_v = v;
                                    max_idx = Some(idx);
                                }
                            }
                        }
                    }
                    if let Some(idx) = max_idx {
                        grad[idx] += dy;
                    }
                }
            }
        }
    }
    Tensor::new(input_c.shape.clone(), grad)
}

pub fn avg_pool2d_backward_nchw(
    input: &Tensor,
    upstream: &Tensor,
    kernel_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
) -> Result<Tensor, TensorError> {
    if input.shape.len() != 4 {
        return Err(TensorError {
            message: "AvgPoolBackward expects rank-4".to_string(),
        });
    }
    let input_c = input.make_contiguous()?;
    let upstream_c = upstream.make_contiguous()?;

    let (n, c, h, w) = (
        input_c.shape[0],
        input_c.shape[1],
        input_c.shape[2],
        input_c.shape[3],
    );
    let kh = kernel_shape[0];
    let kw = kernel_shape[1];
    let sh = strides.first().copied().unwrap_or(1);
    let sw = strides.get(1).copied().unwrap_or(1);
    let (pt, pl, _pb, _pr) = if pads.is_empty() {
        (0, 0, 0, 0)
    } else {
        (pads[0], pads[1], pads[2], pads[3])
    };

    let h_padded = h + pt + (if pads.len() == 4 { pads[2] } else { 0 });
    let w_padded = w + pl + (if pads.len() == 4 { pads[3] } else { 0 });
    let out_h = (h_padded - kh) / sh + 1;
    let out_w = (w_padded - kw) / sw + 1;

    let mut grad = vec![0.0_f32; input_c.logical_len()];
    for ni in 0..n {
        for ci in 0..c {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let dy = upstream_c.data[((ni * c + ci) * out_h + oy) * out_w + ox];
                    let y0 = oy * sh;
                    let x0 = ox * sw;
                    let mut count = 0;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let (py, px) = (y0 + ky, x0 + kx);
                            if py >= pt && py < pt + h && px >= pl && px < pl + w {
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        let val = dy / (count as f32);
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let (py, px) = (y0 + ky, x0 + kx);
                                if py >= pt && py < pt + h && px >= pl && px < pl + w {
                                    let (iy, ix) = (py - pt, px - pl);
                                    grad[((ni * c + ci) * h + iy) * w + ix] += val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Tensor::new(input_c.shape.clone(), grad)
}

pub fn conv2d_backward_input(
    input: &Tensor,
    kernel: &Tensor,
    grad_output: &Tensor,
) -> Result<Tensor, TensorError> {
    let (input_c, k_c, go_c) = (
        input.make_contiguous()?,
        kernel.make_contiguous()?,
        grad_output.make_contiguous()?,
    );
    if input_c.shape.len() != 2 || k_c.shape.len() != 2 || go_c.shape.len() != 2 {
        return Err(TensorError {
            message: "Conv2d_backward_input expects 2D tensors".to_string(),
        });
    }
    let (in_h, in_w) = (input_c.shape[0], input_c.shape[1]);
    let (kh, kw) = (k_c.shape[0], k_c.shape[1]);
    let (gh, gw) = (go_c.shape[0], go_c.shape[1]);

    let mut grad_input = vec![0.0; in_h * in_w];
    for i in 0..gh {
        for j in 0..gw {
            let g = go_c.data[i * gw + j];
            for ki in 0..kh {
                for kj in 0..kw {
                    let (h, w) = (i + ki, j + kj);
                    if h < in_h && w < in_w {
                        grad_input[h * in_w + w] += g * k_c.data[ki * kw + kj];
                    }
                }
            }
        }
    }
    Tensor::new(input_c.shape.clone(), grad_input)
}

pub fn conv2d_backward_weight(
    input: &Tensor,
    kernel_shape: Vec<usize>,
    grad_output: &Tensor,
) -> Result<Tensor, TensorError> {
    let (input_c, go_c) = (input.make_contiguous()?, grad_output.make_contiguous()?);
    if input_c.shape.len() != 2 || kernel_shape.len() != 2 || go_c.shape.len() != 2 {
        return Err(TensorError {
            message: "Conv2d_backward_weight expects 2D tensors".to_string(),
        });
    }
    let (in_h, in_w) = (input_c.shape[0], input_c.shape[1]);
    let (kh, kw) = (kernel_shape[0], kernel_shape[1]);
    let (gh, gw) = (go_c.shape[0], go_c.shape[1]);

    let mut grad_weight = vec![0.0; kh * kw];
    for ki in 0..kh {
        for kj in 0..kw {
            let mut sum = 0.0;
            for i in 0..gh {
                for j in 0..gw {
                    let (h, w) = (i + ki, j + kj);
                    if h < in_h && w < in_w {
                        sum += input_c.data[h * in_w + w] * go_c.data[i * gw + j];
                    }
                }
            }
            grad_weight[ki * kw + kj] = sum;
        }
    }
    Tensor::new(kernel_shape, grad_weight)
}

/// ConvTranspose2D (deconvolution / transposed convolution) for single channel.
/// input: [in_h, in_w], weight: [kh, kw], stride: (sh, sw), padding: (ph, pw)
/// Output size: out_h = (in_h-1)*sh - 2*ph + kh, out_w = (in_w-1)*sw - 2*pw + kw
pub fn conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor, TensorError> {
    let inp = input.make_contiguous()?;
    let w = weight.make_contiguous()?;
    if inp.shape.len() != 2 || w.shape.len() != 2 {
        return Err(TensorError {
            message: "ConvTranspose2D expects 2D input and kernel".to_string(),
        });
    }
    let (in_h, in_w) = (inp.shape[0], inp.shape[1]);
    let (kh, kw) = (w.shape[0], w.shape[1]);
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let out_h = (in_h - 1) * sh + kh - 2 * ph;
    let out_w = (in_w - 1) * sw + kw - 2 * pw;
    let mut out = vec![0.0_f32; out_h * out_w];

    for i in 0..in_h {
        for j in 0..in_w {
            let x_val = inp.data[i * in_w + j];
            for ki in 0..kh {
                for kj in 0..kw {
                    let out_i = i * sh + ki;
                    let out_j = j * sw + kj;
                    // Check padding bounds
                    if out_i >= ph && out_j >= pw {
                        let r = out_i - ph;
                        let c = out_j - pw;
                        if r < out_h && c < out_w {
                            out[r * out_w + c] += x_val * w.data[ki * kw + kj];
                        }
                    }
                }
            }
        }
    }
    Tensor::new(vec![out_h, out_w], out)
}

/// ConvTranspose2D NCHW variant: input [N, C, H, W], weight [C_in, C_out, kH, kW].
/// For simplicity, channel grouping: each input channel convolves with corresponding output channels.
pub fn conv_transpose2d_nchw(
    input: &Tensor,
    weight: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor, TensorError> {
    let inp = input.make_contiguous()?;
    let w = weight.make_contiguous()?;
    if inp.shape.len() != 4 || w.shape.len() != 4 {
        return Err(TensorError {
            message: "ConvTranspose2DNCHW expects rank-4 input and weight [C_in,C_out,kH,kW]"
                .to_string(),
        });
    }
    let (n, c_in, in_h, in_w) = (inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]);
    if w.shape[0] != c_in {
        return Err(TensorError {
            message: format!(
                "ConvTranspose2D weight C_in={} != input C={}",
                w.shape[0], c_in
            ),
        });
    }
    let (c_out, kh, kw) = (w.shape[1], w.shape[2], w.shape[3]);
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let out_h = (in_h - 1) * sh + kh - 2 * ph;
    let out_w = (in_w - 1) * sw + kw - 2 * pw;

    let mut out = vec![0.0_f32; n * c_out * out_h * out_w];

    for ni in 0..n {
        for ci in 0..c_in {
            for co in 0..c_out {
                for ih in 0..in_h {
                    for iw in 0..in_w {
                        let val = inp.data[((ni * c_in + ci) * in_h + ih) * in_w + iw];
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let out_i = ih * sh + ki;
                                let out_j = iw * sw + kj;
                                if out_i >= ph && out_j >= pw {
                                    let r = out_i - ph;
                                    let oc = out_j - pw;
                                    if r < out_h && oc < out_w {
                                        out[((ni * c_out + co) * out_h + r) * out_w + oc] += val
                                            * w.data[ci * c_out * kh * kw
                                                + co * kh * kw
                                                + ki * kw
                                                + kj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Tensor::new(vec![n, c_out, out_h, out_w], out)
}

/// Upsample 2D — Nearest neighbor interpolation.
/// input: [N, C, H, W], scale_h and scale_w are integer scale factors.
pub fn upsample_nearest2d(
    input: &Tensor,
    scale_h: usize,
    scale_w: usize,
) -> Result<Tensor, TensorError> {
    let inp = input.make_contiguous()?;
    if inp.shape.len() != 4 {
        return Err(TensorError {
            message: "Upsample expects rank-4 NCHW input".to_string(),
        });
    }
    let (n, c, h, w) = (inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]);
    let (out_h, out_w) = (h * scale_h, w * scale_w);
    let mut out = vec![0.0_f32; n * c * out_h * out_w];
    for ni in 0..n {
        for ci in 0..c {
            for i in 0..out_h {
                for j in 0..out_w {
                    let src_i = i / scale_h;
                    let src_j = j / scale_w;
                    out[((ni * c + ci) * out_h + i) * out_w + j] =
                        inp.data[((ni * c + ci) * h + src_i) * w + src_j];
                }
            }
        }
    }
    Tensor::new(vec![n, c, out_h, out_w], out)
}

/// Upsample 2D — Bilinear interpolation.
/// scale_h, scale_w: float scale factors.
pub fn upsample_bilinear2d(
    input: &Tensor,
    scale_h: f32,
    scale_w: f32,
) -> Result<Tensor, TensorError> {
    let inp = input.make_contiguous()?;
    if inp.shape.len() != 4 {
        return Err(TensorError {
            message: "UpsampleBilinear expects rank-4 NCHW input".to_string(),
        });
    }
    let (n, c, h, w) = (inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]);
    let out_h = (h as f32 * scale_h).round() as usize;
    let out_w = (w as f32 * scale_w).round() as usize;
    let mut out = vec![0.0_f32; n * c * out_h * out_w];

    for ni in 0..n {
        for ci in 0..c {
            for i in 0..out_h {
                for j in 0..out_w {
                    // Map output coordinates to input coordinates (align_corners=false)
                    let src_y = ((i as f32 + 0.5) / scale_h) - 0.5;
                    let src_x = ((j as f32 + 0.5) / scale_w) - 0.5;
                    let y0 = src_y.floor() as isize;
                    let x0 = src_x.floor() as isize;
                    let y1 = y0 + 1;
                    let x1 = x0 + 1;
                    let dy = src_y - y0 as f32;
                    let dx = src_x - x0 as f32;

                    let clamp = |v: isize, max: usize| v.max(0).min(max as isize - 1) as usize;
                    let y0c = clamp(y0, h);
                    let y1c = clamp(y1, h);
                    let x0c = clamp(x0, w);
                    let x1c = clamp(x1, w);

                    let base = (ni * c + ci) * h;
                    let v00 = inp.data[(base + y0c) * w + x0c];
                    let v01 = inp.data[(base + y0c) * w + x1c];
                    let v10 = inp.data[(base + y1c) * w + x0c];
                    let v11 = inp.data[(base + y1c) * w + x1c];

                    let val = v00 * (1.0 - dy) * (1.0 - dx)
                        + v01 * (1.0 - dy) * dx
                        + v10 * dy * (1.0 - dx)
                        + v11 * dy * dx;

                    out[((ni * c + ci) * out_h + i) * out_w + j] = val;
                }
            }
        }
    }
    Tensor::new(vec![n, c, out_h, out_w], out)
}

/// Upsample backward (nearest neighbor) — scatter gradient back to source pixels.
pub fn upsample_nearest2d_backward(
    upstream: &Tensor,
    orig_h: usize,
    orig_w: usize,
    scale_h: usize,
    scale_w: usize,
) -> Result<Tensor, TensorError> {
    let up = upstream.make_contiguous()?;
    if up.shape.len() != 4 {
        return Err(TensorError {
            message: "UpsampleNearestBackward expects rank-4 input".to_string(),
        });
    }
    let (n, c, out_h, out_w) = (up.shape[0], up.shape[1], up.shape[2], up.shape[3]);
    let mut grad = vec![0.0_f32; n * c * orig_h * orig_w];
    for ni in 0..n {
        for ci in 0..c {
            for i in 0..out_h {
                for j in 0..out_w {
                    let src_i = i / scale_h;
                    let src_j = j / scale_w;
                    grad[((ni * c + ci) * orig_h + src_i) * orig_w + src_j] +=
                        up.data[((ni * c + ci) * out_h + i) * out_w + j];
                }
            }
        }
    }
    Tensor::new(vec![n, c, orig_h, orig_w], grad)
}

/// Depthwise separable convolution (MobileNet-style) for NCHW tensors.
///
/// Performs:
///   1. Depthwise conv: each input channel convolved with its own kernel.
///      input: [N, C, H, W], dw_weight: [C, kH, kW] (one kernel per channel)
///   2. Pointwise conv: 1×1 convolution to mix channels.
///      pw_weight: [C_out, C]
///
/// Returns output: [N, C_out, H_out, W_out]
pub fn depthwise_separable_conv_nchw(
    input: &Tensor,
    dw_weight: &Tensor,
    pw_weight: &Tensor,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<Tensor, TensorError> {
    if input.shape.len() != 4 {
        return Err(TensorError {
            message: format!(
                "DepthwiseSeparableConv expects rank-4 NCHW input, got {:?}",
                input.shape
            ),
        });
    }
    if dw_weight.shape.len() != 3 {
        return Err(TensorError {
            message: format!(
                "DepthwiseSeparableConv dw_weight must be rank-3 [C, kH, kW], got {:?}",
                dw_weight.shape
            ),
        });
    }
    if pw_weight.shape.len() != 2 {
        return Err(TensorError {
            message: format!(
                "DepthwiseSeparableConv pw_weight must be rank-2 [C_out, C], got {:?}",
                pw_weight.shape
            ),
        });
    }

    let n = input.shape[0];
    let c = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];

    let kh = dw_weight.shape[1];
    let kw = dw_weight.shape[2];
    let c_out = pw_weight.shape[0];

    let pad_h = padding[0];
    let pad_w = padding[1];
    let str_h = stride[0].max(1);
    let str_w = stride[1].max(1);

    let out_h = (h + 2 * pad_h).saturating_sub(kh) / str_h + 1;
    let out_w = (w + 2 * pad_w).saturating_sub(kw) / str_w + 1;

    // Step 1: depthwise conv → [N, C, out_h, out_w]
    let mut dw_out = vec![0.0f32; n * c * out_h * out_w];

    for ni in 0..n {
        for ci in 0..c {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = 0.0f32;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = oy * str_h + ky;
                            let ix = ox * str_w + kx;
                            if iy < pad_h || ix < pad_w {
                                continue;
                            }
                            let iy = iy - pad_h;
                            let ix = ix - pad_w;
                            if iy >= h || ix >= w {
                                continue;
                            }
                            let inp_idx = ((ni * c + ci) * h + iy) * w + ix;
                            let ker_idx = (ci * kh + ky) * kw + kx;
                            acc += input.data[inp_idx] * dw_weight.data[ker_idx];
                        }
                    }
                    dw_out[((ni * c + ci) * out_h + oy) * out_w + ox] = acc;
                }
            }
        }
    }

    // Step 2: pointwise 1×1 conv → [N, C_out, out_h, out_w]
    let mut pw_out = vec![0.0f32; n * c_out * out_h * out_w];

    for ni in 0..n {
        for co in 0..c_out {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = 0.0f32;
                    for ci in 0..c {
                        let dw_idx = ((ni * c + ci) * out_h + oy) * out_w + ox;
                        let pw_idx = co * c + ci;
                        acc += dw_out[dw_idx] * pw_weight.data[pw_idx];
                    }
                    pw_out[((ni * c_out + co) * out_h + oy) * out_w + ox] = acc;
                }
            }
        }
    }

    Tensor::new(vec![n, c_out, out_h, out_w], pw_out)
}
