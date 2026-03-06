#![allow(unsafe_code)]

use crate::ir::DeterminismLevel;
use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::memory::DeviceBuffer;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::CompileOptions;

pub fn layer_norm_f32(
    device: &CudaDevice,
    input: &[f32],
    shape: &[usize],
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    if shape.is_empty() {
        return Err("LayerNorm requires at least a 1D tensor".to_string());
    }
    let d = *shape.last().unwrap();
    if weight.len() != d || bias.len() != d {
        return Err(format!(
            "LayerNorm expected weight/bias len {d}, got {}/{}",
            weight.len(),
            bias.len()
        ));
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let weight_device = DeviceBuffer::from_host(device, weight).map_err(|err| err.message)?;
    let bias_device = DeviceBuffer::from_host(device, bias).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, input.len()).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "layernorm",
            "layer_norm_forward_kernel",
            LAYERNORM_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let num_rows = input.len() / d;
    let d_i = d as i32;
    let eps = epsilon;

    let cfg = LaunchConfig {
        grid_dim: ((num_rows as u32), 1, 1),
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
            .arg(&d_i)
            .arg(&eps)
            .launch(cfg)
            .map_err(|err| format!("CUDA layernorm kernel launch failed: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
}

pub fn layer_norm_backward_input_f32(
    device: &CudaDevice,
    input: &[f32],
    upstream: &[f32],
    shape: &[usize],
    weight: &[f32],
    epsilon: f32,
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    if shape.is_empty() {
        return Err("LayerNorm backward input requires at least a 1D tensor".to_string());
    }
    let d = *shape.last().unwrap();
    if input.len() != upstream.len() {
        return Err("LayerNorm backward input length mismatch".to_string());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let upstream_device = DeviceBuffer::from_host(device, upstream).map_err(|err| err.message)?;
    let weight_device = DeviceBuffer::from_host(device, weight).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, input.len()).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "layernorm",
            "layer_norm_backward_input_kernel",
            LAYERNORM_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let num_rows = input.len() / d;
    let d_i = d as i32;
    let eps = epsilon;

    let cfg = LaunchConfig {
        grid_dim: ((num_rows as u32), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(input_device.cuda_slice())
            .arg(upstream_device.cuda_slice())
            .arg(weight_device.cuda_slice())
            .arg(&d_i)
            .arg(&eps)
            .launch(cfg)
            .map_err(|err| format!("CUDA layernorm backward input launch failed: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
}

pub fn layer_norm_backward_weight_f32(
    device: &CudaDevice,
    input: &[f32],
    upstream: &[f32],
    shape: &[usize],
    epsilon: f32,
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    if shape.is_empty() {
        return Err("LayerNorm backward weight requires at least a 1D tensor".to_string());
    }
    let d = *shape.last().unwrap();
    if input.len() != upstream.len() {
        return Err("LayerNorm backward weight length mismatch".to_string());
    }
    if input.is_empty() {
        return Ok(vec![0.0; d]);
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let upstream_device = DeviceBuffer::from_host(device, upstream).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, d).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "layernorm",
            "layer_norm_backward_weight_kernel",
            LAYERNORM_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let num_rows = input.len() / d;
    let d_i = d as i32;
    let num_rows_i = num_rows as i32;
    let eps = epsilon;

    let cfg = LaunchConfig {
        grid_dim: ((d as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(input_device.cuda_slice())
            .arg(upstream_device.cuda_slice())
            .arg(&num_rows_i)
            .arg(&d_i)
            .arg(&eps)
            .launch(cfg)
            .map_err(|err| format!("CUDA layernorm backward weight launch failed: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
}

pub fn layer_norm_backward_bias_f32(
    device: &CudaDevice,
    upstream: &[f32],
    shape: &[usize],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    if shape.is_empty() {
        return Err("LayerNorm backward bias requires at least a 1D tensor".to_string());
    }
    let d = *shape.last().unwrap();
    if upstream.is_empty() {
        return Ok(vec![0.0; d]);
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let upstream_device = DeviceBuffer::from_host(device, upstream).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, d).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "layernorm",
            "layer_norm_backward_bias_kernel",
            LAYERNORM_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let num_rows = upstream.len() / d;
    let d_i = d as i32;
    let num_rows_i = num_rows as i32;

    let cfg = LaunchConfig {
        grid_dim: ((d as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(upstream_device.cuda_slice())
            .arg(&num_rows_i)
            .arg(&d_i)
            .launch(cfg)
            .map_err(|err| format!("CUDA layernorm backward bias launch failed: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
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

const LAYERNORM_KERNELS_CU: &str = r#"
// Forward Pass Kernel
extern "C" __global__ void layer_norm_forward_kernel(
    float* out,
    const float* input,
    const float* weight,
    const float* bias,
    int d,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_input = input + row * d;
    float* row_out = out + row * d;

    // Phase 1: Mean
    float sum = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        sum += row_input[i];
    }

    // Block-level reduction for sum
    // using warp reduction & shared memory
    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / d;

    // Phase 2: Variance
    float var_sum = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float var = sdata[0] / d;
    float rstd = rsqrtf(var + eps);

    // Phase 3: Normalize and Apply affine
    for (int i = tid; i < d; i += blockDim.x) {
        float x_hat = (row_input[i] - mean) * rstd;
        row_out[i] = x_hat * weight[i] + bias[i];
    }
}

// Backward Input Kernel
extern "C" __global__ void layer_norm_backward_input_kernel(
    float* dx,
    const float* input,
    const float* dy,
    const float* weight,
    int d,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_input = input + row * d;
    const float* row_dy = dy + row * d;
    float* row_dx = dx + row * d;

    __shared__ float sdata[256];

    // Calculate mean
    float sum = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        sum += row_input[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / d;

    // Calculate variance
    float var_sum = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float var = sdata[0] / d;
    float rstd = rsqrtf(var + eps);

    // Calculate sum_dy_gamma and sum_dy_gamma_xhat
    float sum_dy_g = 0.0f;
    float sum_dy_g_xh = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float gamma = weight[i];
        float dy_val = row_dy[i];
        float x_hat = (row_input[i] - mean) * rstd;
        float dy_g = dy_val * gamma;
        sum_dy_g += dy_g;
        sum_dy_g_xh += dy_g * x_hat;
    }

    __shared__ float sdata_d[2][256]; // 2 elements
    sdata_d[0][tid] = sum_dy_g;
    sdata_d[1][tid] = sum_dy_g_xh;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_d[0][tid] += sdata_d[0][tid + s];
            sdata_d[1][tid] += sdata_d[1][tid + s];
        }
        __syncthreads();
    }
    float final_sum_dy_g = sdata_d[0][0];
    float final_sum_dy_g_xh = sdata_d[1][0];

    // Compute dx
    float d_f32 = (float)d;
    for (int i = tid; i < d; i += blockDim.x) {
        float gamma = weight[i];
        float x_hat = (row_input[i] - mean) * rstd;
        float dx_val = (d_f32 * row_dy[i] * gamma - final_sum_dy_g - x_hat * final_sum_dy_g_xh) / d_f32;
        row_dx[i] = dx_val * rstd;
    }
}

// Backward Weight Kernel
extern "C" __global__ void layer_norm_backward_weight_kernel(
    float* dw,
    const float* input,
    const float* dy,
    int num_rows,
    int d,
    float eps
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= d) return;

    float dw_val = 0.0f;
    for (int row = 0; row < num_rows; ++row) {
        const float* row_input = input + row * d;
        const float* row_dy = dy + row * d;

        // Compute mean and var for this row explicitly to keep thread divergence simple
        // (Could be precomputed and passed in memory for performance, but this guarantees determinism)
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += row_input[i];
        }
        float mean = sum / d;

        float var_sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            float diff = row_input[i] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / d;
        float rstd = rsqrtf(var + eps);

        float x_hat = (row_input[col] - mean) * rstd;
        dw_val += row_dy[col] * x_hat;
    }
    dw[col] = dw_val;
}

// Backward Bias Kernel
extern "C" __global__ void layer_norm_backward_bias_kernel(
    float* db,
    const float* dy,
    int num_rows,
    int d
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= d) return;

    float db_val = 0.0f;
    for (int row = 0; row < num_rows; ++row) {
        db_val += dy[row * d + col];
    }
    db[col] = db_val;
}
"#;
