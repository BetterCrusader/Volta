use crate::ir::DeterminismLevel;
use crate::ir::NodeId;
use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::memory::DeviceBuffer;

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::CompileOptions;

pub fn run(nodes: &[NodeId]) -> Result<(), String> {
    if nodes.is_empty() {
        return Err("add kernel dispatch received no nodes".to_string());
    }
    Ok(())
}

pub fn add_f32(
    device: &CudaDevice,
    left: &[f32],
    right: &[f32],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    binary_kernel_f32(device, left, right, determinism, "add_kernel", "add")
}

pub fn sub_f32(
    device: &CudaDevice,
    left: &[f32],
    right: &[f32],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    binary_kernel_f32(device, left, right, determinism, "sub_kernel", "sub")
}

pub fn mul_f32(
    device: &CudaDevice,
    left: &[f32],
    right: &[f32],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    binary_kernel_f32(device, left, right, determinism, "mul_kernel", "mul")
}

pub fn div_f32(
    device: &CudaDevice,
    left: &[f32],
    right: &[f32],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    binary_kernel_f32(device, left, right, determinism, "div_kernel", "div")
}

pub fn neg_f32(
    device: &CudaDevice,
    input: &[f32],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, input.len()).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "elementwise_add",
            "neg_kernel",
            ADD_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let numel = i32::try_from(input.len())
        .map_err(|_| format!("neg element count does not fit i32: {}", input.len()))?;
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
            .arg(&numel)
            .launch(cfg)
            .map_err(|err| format!("CUDA kernel launch failed for neg: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
}

pub fn scale_f32(
    device: &CudaDevice,
    input: &[f32],
    alpha: f32,
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    unary_scalar_kernel_f32(device, input, alpha, determinism, "scale_kernel", "scale")
}

pub fn add_scalar_f32(
    device: &CudaDevice,
    input: &[f32],
    alpha: f32,
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    unary_scalar_kernel_f32(
        device,
        input,
        alpha,
        determinism,
        "add_scalar_kernel",
        "add_scalar",
    )
}

pub fn div_scalar_f32(
    device: &CudaDevice,
    input: &[f32],
    alpha: f32,
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    unary_scalar_kernel_f32(
        device,
        input,
        alpha,
        determinism,
        "div_scalar_kernel",
        "div_scalar",
    )
}

pub fn sqrt_f32(
    device: &CudaDevice,
    input: &[f32],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, input.len()).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "elementwise_add",
            "sqrt_kernel",
            ADD_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let numel = i32::try_from(input.len())
        .map_err(|_| format!("sqrt element count does not fit i32: {}", input.len()))?;
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
            .arg(&numel)
            .launch(cfg)
            .map_err(|err| format!("CUDA kernel launch failed for sqrt: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
}

fn binary_kernel_f32(
    device: &CudaDevice,
    left: &[f32],
    right: &[f32],
    determinism: DeterminismLevel,
    kernel_name: &str,
    op_name: &str,
) -> Result<Vec<f32>, String> {
    if left.len() != right.len() {
        return Err(format!(
            "{op_name} shape mismatch: left_len={} right_len={}",
            left.len(),
            right.len()
        ));
    }
    if left.is_empty() {
        return Ok(Vec::new());
    }

    let stream = device.stream().map_err(|err| err.message)?;

    let left_device = DeviceBuffer::from_host(device, left).map_err(|err| err.message)?;
    let right_device = DeviceBuffer::from_host(device, right).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, left.len()).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "elementwise_add",
            kernel_name,
            ADD_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let numel = i32::try_from(left.len())
        .map_err(|_| format!("{op_name} element count does not fit i32: {}", left.len()))?;
    let cfg = LaunchConfig {
        grid_dim: ((left.len() as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(left_device.cuda_slice())
            .arg(right_device.cuda_slice())
            .arg(&numel)
            .launch(cfg)
            .map_err(|err| format!("CUDA kernel launch failed for {op_name}: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
}

fn unary_scalar_kernel_f32(
    device: &CudaDevice,
    input: &[f32],
    alpha: f32,
    determinism: DeterminismLevel,
    kernel_name: &str,
    op_name: &str,
) -> Result<Vec<f32>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, input.len()).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "elementwise_add",
            kernel_name,
            ADD_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let numel = i32::try_from(input.len())
        .map_err(|_| format!("{op_name} element count does not fit i32: {}", input.len()))?;
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
            .arg(&alpha)
            .arg(&numel)
            .launch(cfg)
            .map_err(|err| format!("CUDA kernel launch failed for {op_name}: {err}"))?;
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

const ADD_KERNELS_CU: &str = r#"
extern "C" __global__ void add_kernel(float* out, const float* left, const float* right, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = left[i] + right[i];
    }
}

extern "C" __global__ void sub_kernel(float* out, const float* left, const float* right, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = left[i] - right[i];
    }
}

extern "C" __global__ void mul_kernel(float* out, const float* left, const float* right, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = left[i] * right[i];
    }
}

extern "C" __global__ void div_kernel(float* out, const float* left, const float* right, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = left[i] / right[i];
    }
}

extern "C" __global__ void neg_kernel(float* out, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = -input[i];
    }
}

extern "C" __global__ void scale_kernel(float* out, const float* input, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = input[i] * alpha;
    }
}

extern "C" __global__ void add_scalar_kernel(float* out, const float* input, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = input[i] + alpha;
    }
}

extern "C" __global__ void div_scalar_kernel(float* out, const float* input, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = input[i] / alpha;
    }
}

extern "C" __global__ void sqrt_kernel(float* out, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sqrtf(input[i]);
    }
}
"#;
