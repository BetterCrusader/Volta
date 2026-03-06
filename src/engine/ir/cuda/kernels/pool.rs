#![allow(unsafe_code)]

use crate::ir::DeterminismLevel;
use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::memory::DeviceBuffer;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::CompileOptions;

pub fn max_pool2d_f32(
    device: &CudaDevice,
    input: &[f32],
    shape: [usize; 4],
    cfg: &Pool2dConfig,
    determinism: DeterminismLevel,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    pool2d_f32(device, input, shape, cfg, determinism, true)
}

pub fn avg_pool2d_f32(
    device: &CudaDevice,
    input: &[f32],
    shape: [usize; 4],
    cfg: &Pool2dConfig,
    determinism: DeterminismLevel,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    pool2d_f32(device, input, shape, cfg, determinism, false)
}

pub fn max_pool2d_backward_f32(
    device: &CudaDevice,
    input: &[f32],
    input_shape: [usize; 4],
    upstream: &[f32],
    cfg: &Pool2dConfig,
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    pool2d_backward_f32(device, input, input_shape, upstream, cfg, determinism, true)
}

pub fn avg_pool2d_backward_f32(
    device: &CudaDevice,
    input: &[f32],
    input_shape: [usize; 4],
    upstream: &[f32],
    cfg: &Pool2dConfig,
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    pool2d_backward_f32(
        device,
        input,
        input_shape,
        upstream,
        cfg,
        determinism,
        false,
    )
}

#[derive(Debug, Clone)]
pub struct Pool2dConfig {
    pub kernel_shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub pads: Vec<usize>,
}

fn pool2d_f32(
    device: &CudaDevice,
    input: &[f32],
    shape: [usize; 4],
    cfg: &Pool2dConfig,
    determinism: DeterminismLevel,
    is_max: bool,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    let [n, c, h, w] = shape;
    if input.len() != n * c * h * w {
        return Err(format!(
            "pool input length mismatch: got {}, expected {}",
            input.len(),
            n * c * h * w
        ));
    }
    if cfg.kernel_shape.len() != 2 {
        return Err("pool kernel_shape must have 2 values".to_string());
    }
    let kh = cfg.kernel_shape[0];
    let kw = cfg.kernel_shape[1];
    if kh == 0 || kw == 0 {
        return Err("pool kernel values must be > 0".to_string());
    }
    let sh = cfg.strides.first().copied().unwrap_or(1);
    let sw = cfg.strides.get(1).copied().unwrap_or(1);
    if sh == 0 || sw == 0 {
        return Err("pool strides must be > 0".to_string());
    }
    let (pt, pl, pb, pr) = if cfg.pads.is_empty() {
        (0_usize, 0_usize, 0_usize, 0_usize)
    } else if cfg.pads.len() == 4 {
        (cfg.pads[0], cfg.pads[1], cfg.pads[2], cfg.pads[3])
    } else {
        return Err("pool pads must be empty or [top,left,bottom,right]".to_string());
    };

    let h_padded = h + pt + pb;
    let w_padded = w + pl + pr;
    if h_padded < kh || w_padded < kw {
        return Err(format!(
            "pool kernel {:?} too large for padded input [{},{}]",
            cfg.kernel_shape, h_padded, w_padded
        ));
    }
    let out_h = (h_padded - kh) / sh + 1;
    let out_w = (w_padded - kw) / sw + 1;
    let out_len = n * c * out_h * out_w;
    if out_len == 0 {
        return Ok((vec![n, c, out_h, out_w], Vec::new()));
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, out_len).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "pool2d",
            if is_max {
                "max_pool2d_nchw_kernel"
            } else {
                "avg_pool2d_nchw_kernel"
            },
            POOL_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let to_i32 = |name: &str, v: usize| -> Result<i32, String> {
        i32::try_from(v).map_err(|_| format!("{name} value out of i32 range: {v}"))
    };

    let n_i = to_i32("n", n)?;
    let c_i = to_i32("c", c)?;
    let h_i = to_i32("h", h)?;
    let w_i = to_i32("w", w)?;
    let kh_i = to_i32("kh", kh)?;
    let kw_i = to_i32("kw", kw)?;
    let sh_i = to_i32("sh", sh)?;
    let sw_i = to_i32("sw", sw)?;
    let pt_i = to_i32("pt", pt)?;
    let pl_i = to_i32("pl", pl)?;
    let out_h_i = to_i32("out_h", out_h)?;
    let out_w_i = to_i32("out_w", out_w)?;
    let out_len_i = to_i32("out_len", out_len)?;

    let cfg = LaunchConfig {
        grid_dim: ((out_len as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(input_device.cuda_slice())
            .arg(&n_i)
            .arg(&c_i)
            .arg(&h_i)
            .arg(&w_i)
            .arg(&kh_i)
            .arg(&kw_i)
            .arg(&sh_i)
            .arg(&sw_i)
            .arg(&pt_i)
            .arg(&pl_i)
            .arg(&out_h_i)
            .arg(&out_w_i)
            .arg(&out_len_i)
            .launch(cfg)
            .map_err(|err| format!("CUDA pool kernel launch failed: {err}"))?;
    }

    let out = out_device.copy_to_host(device).map_err(|err| err.message)?;
    Ok((vec![n, c, out_h, out_w], out))
}

fn pool2d_backward_f32(
    device: &CudaDevice,
    input: &[f32],
    input_shape: [usize; 4],
    upstream: &[f32],
    cfg: &Pool2dConfig,
    determinism: DeterminismLevel,
    is_max: bool,
) -> Result<Vec<f32>, String> {
    let [n, c, h, w] = input_shape;
    if input.len() != n * c * h * w {
        return Err(format!(
            "pool backward input length mismatch: got {}, expected {}",
            input.len(),
            n * c * h * w
        ));
    }
    if cfg.kernel_shape.len() != 2 {
        return Err("pool backward kernel_shape must have 2 values".to_string());
    }
    let kh = cfg.kernel_shape[0];
    let kw = cfg.kernel_shape[1];
    if kh == 0 || kw == 0 {
        return Err("pool backward kernel values must be > 0".to_string());
    }
    let sh = cfg.strides.first().copied().unwrap_or(1);
    let sw = cfg.strides.get(1).copied().unwrap_or(1);
    if sh == 0 || sw == 0 {
        return Err("pool backward strides must be > 0".to_string());
    }
    let (pt, pl, pb, pr) = if cfg.pads.is_empty() {
        (0_usize, 0_usize, 0_usize, 0_usize)
    } else if cfg.pads.len() == 4 {
        (cfg.pads[0], cfg.pads[1], cfg.pads[2], cfg.pads[3])
    } else {
        return Err("pool backward pads must be empty or [top,left,bottom,right]".to_string());
    };

    let h_padded = h + pt + pb;
    let w_padded = w + pl + pr;
    if h_padded < kh || w_padded < kw {
        return Err(format!(
            "pool backward kernel {:?} too large for padded input [{},{}]",
            cfg.kernel_shape, h_padded, w_padded
        ));
    }
    let out_h = (h_padded - kh) / sh + 1;
    let out_w = (w_padded - kw) / sw + 1;
    let out_len = n * c * out_h * out_w;
    if upstream.len() != out_len {
        return Err(format!(
            "pool backward upstream length mismatch: got {}, expected {}",
            upstream.len(),
            out_len
        ));
    }

    if input.is_empty() {
        return Ok(Vec::new());
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let upstream_device = DeviceBuffer::from_host(device, upstream).map_err(|err| err.message)?;
    let mut out_grad_device =
        DeviceBuffer::zeros(device, input.len()).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "pool2d",
            if is_max {
                "max_pool2d_backward_nchw_kernel"
            } else {
                "avg_pool2d_backward_nchw_kernel"
            },
            POOL_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let to_i32 = |name: &str, v: usize| -> Result<i32, String> {
        i32::try_from(v).map_err(|_| format!("{name} value out of i32 range: {v}"))
    };

    let n_i = to_i32("n", n)?;
    let c_i = to_i32("c", c)?;
    let h_i = to_i32("h", h)?;
    let w_i = to_i32("w", w)?;
    let kh_i = to_i32("kh", kh)?;
    let kw_i = to_i32("kw", kw)?;
    let sh_i = to_i32("sh", sh)?;
    let sw_i = to_i32("sw", sw)?;
    let pt_i = to_i32("pt", pt)?;
    let pl_i = to_i32("pl", pl)?;
    let out_h_i = to_i32("out_h", out_h)?;
    let out_w_i = to_i32("out_w", out_w)?;
    let in_len_i = to_i32("in_len", input.len())?;

    let cfg_launch = LaunchConfig {
        grid_dim: ((input.len() as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_grad_device.cuda_slice_mut())
            .arg(input_device.cuda_slice())
            .arg(upstream_device.cuda_slice())
            .arg(&n_i)
            .arg(&c_i)
            .arg(&h_i)
            .arg(&w_i)
            .arg(&kh_i)
            .arg(&kw_i)
            .arg(&sh_i)
            .arg(&sw_i)
            .arg(&pt_i)
            .arg(&pl_i)
            .arg(&out_h_i)
            .arg(&out_w_i)
            .arg(&in_len_i)
            .launch(cfg_launch)
            .map_err(|err| format!("CUDA pool backward kernel launch failed: {err}"))?;
    }

    out_grad_device
        .copy_to_host(device)
        .map_err(|err| err.message)
}

fn compile_options(device: &CudaDevice, determinism: DeterminismLevel) -> CompileOptions {
    let arch = format!(
        "--gpu-architecture=compute_{}{}",
        device.compute_capability_major, device.compute_capability_minor
    );
    let mut options = vec![arch];
    if determinism == DeterminismLevel::Strict {
        options.push("--fmad=false".to_string());
    }
    CompileOptions {
        use_fast_math: Some(false),
        options,
        ..Default::default()
    }
}

const POOL_KERNELS_CU: &str = r#"
extern "C" __global__ void max_pool2d_nchw_kernel(
    float* out,
    const float* input,
    int n,
    int c,
    int h,
    int w,
    int kh,
    int kw,
    int sh,
    int sw,
    int pt,
    int pl,
    int out_h,
    int out_w,
    int out_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_len) return;

    int ox = idx % out_w;
    int t1 = idx / out_w;
    int oy = t1 % out_h;
    int t2 = t1 / out_h;
    int ci = t2 % c;
    int ni = t2 / c;

    int y0 = oy * sh;
    int x0 = ox * sw;
    float best = -3.402823466e+38F;
    bool found = false;

    for (int ky = 0; ky < kh; ++ky) {
        for (int kx = 0; kx < kw; ++kx) {
            int py = y0 + ky;
            int px = x0 + kx;
            if (py < pt || py >= pt + h || px < pl || px >= pl + w) continue;
            int iy = py - pt;
            int ix = px - pl;
            int in_idx = ((ni * c + ci) * h + iy) * w + ix;
            float v = input[in_idx];
            if (!found || v > best) {
                best = v;
                found = true;
            }
        }
    }
    out[idx] = found ? best : 0.0f;
}

extern "C" __global__ void avg_pool2d_nchw_kernel(
    float* out,
    const float* input,
    int n,
    int c,
    int h,
    int w,
    int kh,
    int kw,
    int sh,
    int sw,
    int pt,
    int pl,
    int out_h,
    int out_w,
    int out_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_len) return;

    int ox = idx % out_w;
    int t1 = idx / out_w;
    int oy = t1 % out_h;
    int t2 = t1 / out_h;
    int ci = t2 % c;
    int ni = t2 / c;

    int y0 = oy * sh;
    int x0 = ox * sw;
    float sum = 0.0f;
    int count = 0;

    for (int ky = 0; ky < kh; ++ky) {
        for (int kx = 0; kx < kw; ++kx) {
            int py = y0 + ky;
            int px = x0 + kx;
            if (py < pt || py >= pt + h || px < pl || px >= pl + w) continue;
            int iy = py - pt;
            int ix = px - pl;
            int in_idx = ((ni * c + ci) * h + iy) * w + ix;
            sum += input[in_idx];
            count += 1;
        }
    }
    out[idx] = count > 0 ? (sum / (float)count) : 0.0f;
}

extern "C" __global__ void max_pool2d_backward_nchw_kernel(
    float* grad_input,
    const float* input,
    const float* upstream,
    int n,
    int c,
    int h,
    int w,
    int kh,
    int kw,
    int sh,
    int sw,
    int pt,
    int pl,
    int out_h,
    int out_w,
    int in_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_len) return;

    int ix = idx % w;
    int t1 = idx / w;
    int iy = t1 % h;
    int t2 = t1 / h;
    int ci = t2 % c;
    int ni = t2 / c;

    float grad = 0.0f;
    for (int oy = 0; oy < out_h; ++oy) {
        int y0 = oy * sh;
        int py = iy + pt;
        if (py < y0 || py >= y0 + kh) continue;
        for (int ox = 0; ox < out_w; ++ox) {
            int x0 = ox * sw;
            int px = ix + pl;
            if (px < x0 || px >= x0 + kw) continue;

            float best = -3.402823466e+38F;
            int best_idx = -1;
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    int pyy = y0 + ky;
                    int pxx = x0 + kx;
                    if (pyy < pt || pyy >= pt + h || pxx < pl || pxx >= pl + w) continue;
                    int iyy = pyy - pt;
                    int ixx = pxx - pl;
                    int in_idx = ((ni * c + ci) * h + iyy) * w + ixx;
                    float v = input[in_idx];
                    if (best_idx < 0 || v > best) {
                        best = v;
                        best_idx = in_idx;
                    }
                }
            }

            if (best_idx == idx) {
                int up_idx = ((ni * c + ci) * out_h + oy) * out_w + ox;
                grad += upstream[up_idx];
            }
        }
    }
    grad_input[idx] = grad;
}

extern "C" __global__ void avg_pool2d_backward_nchw_kernel(
    float* grad_input,
    const float* input,
    const float* upstream,
    int n,
    int c,
    int h,
    int w,
    int kh,
    int kw,
    int sh,
    int sw,
    int pt,
    int pl,
    int out_h,
    int out_w,
    int in_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_len) return;

    int ix = idx % w;
    int t1 = idx / w;
    int iy = t1 % h;
    int t2 = t1 / h;
    int ci = t2 % c;
    int ni = t2 / c;

    float grad = 0.0f;
    for (int oy = 0; oy < out_h; ++oy) {
        int y0 = oy * sh;
        int py = iy + pt;
        if (py < y0 || py >= y0 + kh) continue;
        for (int ox = 0; ox < out_w; ++ox) {
            int x0 = ox * sw;
            int px = ix + pl;
            if (px < x0 || px >= x0 + kw) continue;

            int count = 0;
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    int pyy = y0 + ky;
                    int pxx = x0 + kx;
                    if (pyy < pt || pyy >= pt + h || pxx < pl || pxx >= pl + w) continue;
                    count += 1;
                }
            }
            if (count > 0) {
                int up_idx = ((ni * c + ci) * out_h + oy) * out_w + ox;
                grad += upstream[up_idx] / (float)count;
            }
        }
    }
    grad_input[idx] = grad;
}
"#;
