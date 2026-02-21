use crate::ir::DeterminismLevel;
use crate::ir::NodeId;
use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::memory::DeviceBuffer;

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::CompileOptions;

pub fn run(nodes: &[NodeId]) -> Result<(), String> {
    if nodes.is_empty() {
        return Err("softmax kernel dispatch received no nodes".to_string());
    }
    Ok(())
}

pub fn softmax_f32(
    device: &CudaDevice,
    input: &[f32],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    if input.is_empty() {
        return Err("softmax expects non-empty input".to_string());
    }
    if input.len() > 1024 {
        return Err(format!(
            "softmax strict kernel currently supports up to 1024 elements, got {}",
            input.len()
        ));
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, input.len()).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "softmax_module",
            "softmax_kernel",
            SOFTMAX_KERNEL_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let numel = i32::try_from(input.len())
        .map_err(|_| format!("softmax element count does not fit i32: {}", input.len()))?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(input_device.cuda_slice())
            .arg(&numel)
            .launch(cfg)
            .map_err(|err| format!("CUDA kernel launch failed for softmax: {err}"))?;
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

const SOFTMAX_KERNEL_CU: &str = r#"
extern "C" __global__ void softmax_kernel(float* out, const float* input, int n) {
    __shared__ float scratch[1024];
    int tid = threadIdx.x;

    float x = -3.402823466e+38F;
    if (tid < n) {
        x = input[tid];
    }
    scratch[tid] = x;
    __syncthreads();

    for (int stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float left = scratch[tid];
            float right = scratch[tid + stride];
            scratch[tid] = left > right ? left : right;
        }
        __syncthreads();
    }

    float maxv = scratch[0];
    float ex = 0.0f;
    if (tid < n) {
        ex = expf(input[tid] - maxv);
    }
    scratch[tid] = ex;
    __syncthreads();

    for (int stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = scratch[tid] + scratch[tid + stride];
        }
        __syncthreads();
    }

    float sum = scratch[0];
    if (tid < n) {
        out[tid] = ex / sum;
    }
}
"#;
