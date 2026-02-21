use crate::ir::DeterminismLevel;
use crate::ir::NodeId;
use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::memory::DeviceBuffer;

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::CompileOptions;

pub fn run(nodes: &[NodeId]) -> Result<(), String> {
    if nodes.is_empty() {
        return Err("relu kernel dispatch received no nodes".to_string());
    }
    Ok(())
}

pub fn relu_f32(
    device: &CudaDevice,
    input: &[f32],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    unary_kernel_f32(device, input, determinism, "relu_kernel", "relu")
}

pub fn relu_backward_f32(
    device: &CudaDevice,
    input: &[f32],
    grad: &[f32],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    if input.len() != grad.len() {
        return Err(format!(
            "relu_backward shape mismatch: input_len={} grad_len={}",
            input.len(),
            grad.len()
        ));
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let grad_device = DeviceBuffer::from_host(device, grad).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, input.len()).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "relu_module",
            "relu_backward_kernel",
            RELU_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let numel = i32::try_from(input.len()).map_err(|_| {
        format!(
            "relu_backward element count does not fit i32: {}",
            input.len()
        )
    })?;
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
            .arg(grad_device.cuda_slice())
            .arg(&numel)
            .launch(cfg)
            .map_err(|err| format!("CUDA kernel launch failed for relu_backward: {err}"))?;
    }

    out_device.copy_to_host(device).map_err(|err| err.message)
}

fn unary_kernel_f32(
    device: &CudaDevice,
    input: &[f32],
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
            "relu_module",
            kernel_name,
            RELU_KERNELS_CU,
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

const RELU_KERNELS_CU: &str = r#"
extern "C" __global__ void relu_kernel(float* out, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

extern "C" __global__ void relu_backward_kernel(
    float* out,
    const float* input,
    const float* grad,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = input[i] > 0.0f ? grad[i] : 0.0f;
    }
}
"#;
