#![allow(unsafe_code)]

use crate::ir::DeterminismLevel;
use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::memory::DeviceBuffer;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::CompileOptions;

pub fn batch_norm_nchw_f32(
    device: &CudaDevice,
    input: &[f32],
    shape: [usize; 4],
    params: &BatchNormParams<'_>,
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    let [n, c, h, w] = shape;
    if input.len() != n * c * h * w {
        return Err(format!(
            "batchnorm input length mismatch: got {}, expected {}",
            input.len(),
            n * c * h * w
        ));
    }
    if params.weight.len() != c
        || params.bias.len() != c
        || params.mean.len() != c
        || params.var.len() != c
    {
        return Err(format!(
            "batchnorm channel vector size mismatch: expected {c}, got w={} b={} mean={} var={}",
            params.weight.len(),
            params.bias.len(),
            params.mean.len(),
            params.var.len()
        ));
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let weight_device =
        DeviceBuffer::from_host(device, params.weight).map_err(|err| err.message)?;
    let bias_device = DeviceBuffer::from_host(device, params.bias).map_err(|err| err.message)?;
    let mean_device = DeviceBuffer::from_host(device, params.mean).map_err(|err| err.message)?;
    let var_device = DeviceBuffer::from_host(device, params.var).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, input.len()).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "batchnorm",
            "batch_norm_nchw_kernel",
            BATCHNORM_KERNELS_CU,
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
    let len_i = to_i32("len", input.len())?;
    let eps = 1e-5_f32;

    let cfg = LaunchConfig {
        grid_dim: ((input.len() as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(input_device.cuda_slice())
            .arg(weight_device.cuda_slice())
            .arg(bias_device.cuda_slice())
            .arg(mean_device.cuda_slice())
            .arg(var_device.cuda_slice())
            .arg(&n_i)
            .arg(&c_i)
            .arg(&h_i)
            .arg(&w_i)
            .arg(&eps)
            .arg(&len_i)
            .launch(cfg)
            .map_err(|err| format!("CUDA batchnorm kernel launch failed: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
}

pub fn batch_norm_backward_input_nchw_f32(
    device: &CudaDevice,
    input_shape: [usize; 4],
    upstream: &[f32],
    weight: &[f32],
    var: &[f32],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    let [n, c, h, w] = input_shape;
    let input_len = n * c * h * w;
    if upstream.len() != input_len {
        return Err(format!(
            "batchnorm backward input upstream length mismatch: got {}, expected {}",
            upstream.len(),
            input_len
        ));
    }
    if weight.len() != c || var.len() != c {
        return Err(format!(
            "batchnorm backward input channel mismatch: expected {c}, got weight={} var={}",
            weight.len(),
            var.len()
        ));
    }
    if upstream.is_empty() {
        return Ok(Vec::new());
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let upstream_device = DeviceBuffer::from_host(device, upstream).map_err(|err| err.message)?;
    let weight_device = DeviceBuffer::from_host(device, weight).map_err(|err| err.message)?;
    let var_device = DeviceBuffer::from_host(device, var).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, input_len).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "batchnorm",
            "batch_norm_backward_input_nchw_kernel",
            BATCHNORM_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let to_i32 = |name: &str, v: usize| -> Result<i32, String> {
        i32::try_from(v).map_err(|_| format!("{name} value out of i32 range: {v}"))
    };
    let c_i = to_i32("c", c)?;
    let h_i = to_i32("h", h)?;
    let w_i = to_i32("w", w)?;
    let len_i = to_i32("len", input_len)?;
    let eps = 1e-5_f32;

    let cfg = LaunchConfig {
        grid_dim: ((input_len as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(upstream_device.cuda_slice())
            .arg(weight_device.cuda_slice())
            .arg(var_device.cuda_slice())
            .arg(&c_i)
            .arg(&h_i)
            .arg(&w_i)
            .arg(&eps)
            .arg(&len_i)
            .launch(cfg)
            .map_err(|err| format!("CUDA batchnorm backward input launch failed: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
}

pub fn batch_norm_backward_weight_nchw_f32(
    device: &CudaDevice,
    input: &[f32],
    input_shape: [usize; 4],
    upstream: &[f32],
    mean: &[f32],
    var: &[f32],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    let [n, c, h, w] = input_shape;
    let input_len = n * c * h * w;
    if input.len() != input_len || upstream.len() != input_len {
        return Err(format!(
            "batchnorm backward weight input/upstream mismatch: input={} upstream={} expected={}",
            input.len(),
            upstream.len(),
            input_len
        ));
    }
    if mean.len() != c || var.len() != c {
        return Err(format!(
            "batchnorm backward weight channel mismatch: expected {c}, got mean={} var={}",
            mean.len(),
            var.len()
        ));
    }
    if c == 0 {
        return Ok(Vec::new());
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let upstream_device = DeviceBuffer::from_host(device, upstream).map_err(|err| err.message)?;
    let mean_device = DeviceBuffer::from_host(device, mean).map_err(|err| err.message)?;
    let var_device = DeviceBuffer::from_host(device, var).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, c).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "batchnorm",
            "batch_norm_backward_weight_nchw_kernel",
            BATCHNORM_KERNELS_CU,
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
    let eps = 1e-5_f32;

    let cfg = LaunchConfig {
        grid_dim: (c as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(input_device.cuda_slice())
            .arg(upstream_device.cuda_slice())
            .arg(mean_device.cuda_slice())
            .arg(var_device.cuda_slice())
            .arg(&n_i)
            .arg(&c_i)
            .arg(&h_i)
            .arg(&w_i)
            .arg(&eps)
            .launch(cfg)
            .map_err(|err| format!("CUDA batchnorm backward weight launch failed: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
}

pub fn batch_norm_backward_bias_nchw_f32(
    device: &CudaDevice,
    upstream: &[f32],
    input_shape: [usize; 4],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    let [n, c, h, w] = input_shape;
    let input_len = n * c * h * w;
    if upstream.len() != input_len {
        return Err(format!(
            "batchnorm backward bias upstream length mismatch: got {}, expected {}",
            upstream.len(),
            input_len
        ));
    }
    if c == 0 {
        return Ok(Vec::new());
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let upstream_device = DeviceBuffer::from_host(device, upstream).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, c).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "batchnorm",
            "batch_norm_backward_bias_nchw_kernel",
            BATCHNORM_KERNELS_CU,
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

    let cfg = LaunchConfig {
        grid_dim: (c as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(upstream_device.cuda_slice())
            .arg(&n_i)
            .arg(&c_i)
            .arg(&h_i)
            .arg(&w_i)
            .launch(cfg)
            .map_err(|err| format!("CUDA batchnorm backward bias launch failed: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
}

#[derive(Debug, Clone, Copy)]
pub struct BatchNormParams<'a> {
    pub weight: &'a [f32],
    pub bias: &'a [f32],
    pub mean: &'a [f32],
    pub var: &'a [f32],
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

const BATCHNORM_KERNELS_CU: &str = r#"
extern "C" __global__ void batch_norm_nchw_kernel(
    float* out,
    const float* input,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* var,
    int n,
    int c,
    int h,
    int w,
    float eps,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;

    int spatial = h * w;
    int cidx = (idx / spatial) % c;
    float scale = weight[cidx] / sqrtf(var[cidx] + eps);
    float shift = bias[cidx] - mean[cidx] * scale;
    out[idx] = input[idx] * scale + shift;
}

extern "C" __global__ void batch_norm_backward_input_nchw_kernel(
    float* grad_input,
    const float* upstream,
    const float* weight,
    const float* var,
    int c,
    int h,
    int w,
    float eps,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    int spatial = h * w;
    int cidx = (idx / spatial) % c;
    float scale = weight[cidx] / sqrtf(var[cidx] + eps);
    grad_input[idx] = upstream[idx] * scale;
}

extern "C" __global__ void batch_norm_backward_weight_nchw_kernel(
    float* grad_weight,
    const float* input,
    const float* upstream,
    const float* mean,
    const float* var,
    int n,
    int c,
    int h,
    int w,
    float eps
) {
    int ci = blockIdx.x;
    if (ci >= c) return;
    float inv_std = rsqrtf(var[ci] + eps);
    float m_val = mean[ci];
    float acc = 0.0f;
    int spatial = h * w;
    int num_elements = n * spatial;

    for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
        int ni = i / spatial;
        int s = i % spatial;
        int idx = (ni * c + ci) * spatial + s;
        acc += upstream[idx] * (input[idx] - m_val) * inv_std;
    }

    __shared__ float sdata[256];
    sdata[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        grad_weight[ci] = sdata[0];
    }
}

extern "C" __global__ void batch_norm_backward_bias_nchw_kernel(
    float* grad_bias,
    const float* upstream,
    int n,
    int c,
    int h,
    int w
) {
    int ci = blockIdx.x;
    if (ci >= c) return;
    float acc = 0.0f;
    int spatial = h * w;
    int num_elements = n * spatial;

    for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
        int ni = i / spatial;
        int s = i % spatial;
        int idx = (ni * c + ci) * spatial + s;
        acc += upstream[idx];
    }

    __shared__ float sdata[256];
    sdata[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        grad_bias[ci] = sdata[0];
    }
}
"#;
