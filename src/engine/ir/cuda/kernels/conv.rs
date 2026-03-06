#![allow(unsafe_code)]

use crate::ir::DeterminismLevel;
use crate::ir::NodeId;
use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::memory::DeviceBuffer;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::CompileOptions;

const TILE_DIM: usize = 16;
const MAX_SHARED_MEM_BYTES_PER_BLOCK: u32 = 48 * 1024;

pub fn run(nodes: &[NodeId]) -> Result<(), String> {
    if nodes.is_empty() {
        return Err("conv kernel dispatch received no nodes".to_string());
    }
    Ok(())
}

pub fn conv2d_valid_f32(
    device: &CudaDevice,
    input: &[f32],
    input_shape: [usize; 2],
    kernel: &[f32],
    kernel_shape: [usize; 2],
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    let in_h = input_shape[0];
    let in_w = input_shape[1];
    let k_h = kernel_shape[0];
    let k_w = kernel_shape[1];

    if in_h == 0 || in_w == 0 || k_h == 0 || k_w == 0 {
        return Err("conv2d shapes must be non-zero".to_string());
    }
    if input.len() != in_h * in_w {
        return Err(format!(
            "conv2d input length mismatch: got {}, expected {}",
            input.len(),
            in_h * in_w
        ));
    }
    if kernel.len() != k_h * k_w {
        return Err(format!(
            "conv2d kernel length mismatch: got {}, expected {}",
            kernel.len(),
            k_h * k_w
        ));
    }
    if k_h > in_h || k_w > in_w {
        return Err(format!(
            "conv2d kernel {:?} cannot be larger than input {:?}",
            kernel_shape, input_shape
        ));
    }

    let out_h = in_h - k_h + 1;
    let out_w = in_w - k_w + 1;
    let out_len = out_h * out_w;
    if out_len == 0 {
        return Ok(Vec::new());
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let kernel_device = DeviceBuffer::from_host(device, kernel).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, out_len).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "conv2d",
            "conv2d_valid_tiled_kernel",
            CONV_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let to_i32 = |name: &str, v: usize| -> Result<i32, String> {
        i32::try_from(v).map_err(|_| format!("{name} out of i32 range: {v}"))
    };
    let in_h_i = to_i32("in_h", in_h)?;
    let in_w_i = to_i32("in_w", in_w)?;
    let k_h_i = to_i32("k_h", k_h)?;
    let k_w_i = to_i32("k_w", k_w)?;
    let out_h_i = to_i32("out_h", out_h)?;
    let out_w_i = to_i32("out_w", out_w)?;

    let shared_h = TILE_DIM + k_h - 1;
    let shared_w = TILE_DIM + k_w - 1;
    let shared_mem_bytes = shared_h
        .checked_mul(shared_w)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "conv2d shared memory size overflow".to_string())?;
    let shared_mem_bytes = u32::try_from(shared_mem_bytes)
        .map_err(|_| "conv2d shared memory size exceeds u32".to_string())?;
    if shared_mem_bytes > MAX_SHARED_MEM_BYTES_PER_BLOCK {
        return Err(format!(
            "conv2d shared memory request {} exceeds conservative per-block limit {}",
            shared_mem_bytes, MAX_SHARED_MEM_BYTES_PER_BLOCK
        ));
    }

    let cfg = LaunchConfig {
        grid_dim: (
            (out_w as u32).div_ceil(TILE_DIM as u32),
            (out_h as u32).div_ceil(TILE_DIM as u32),
            1,
        ),
        block_dim: (TILE_DIM as u32, TILE_DIM as u32, 1),
        shared_mem_bytes,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(input_device.cuda_slice())
            .arg(kernel_device.cuda_slice())
            .arg(&in_h_i)
            .arg(&in_w_i)
            .arg(&k_h_i)
            .arg(&k_w_i)
            .arg(&out_h_i)
            .arg(&out_w_i)
            .launch(cfg)
            .map_err(|err| format!("CUDA conv2d kernel launch failed: {err}"))?;
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

const CONV_KERNELS_CU: &str = r#"
extern "C" __global__ void conv2d_valid_tiled_kernel(
    float* out,
    const float* input,
    const float* kernel,
    int in_h,
    int in_w,
    int k_h,
    int k_w,
    int out_h,
    int out_w
) {
    __shared__ float sh[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int block_col = blockIdx.x * blockDim.x;
    const int block_row = blockIdx.y * blockDim.y;

    const int shared_w = blockDim.x + k_w - 1;
    const int shared_h = blockDim.y + k_h - 1;

    for (int sy = ty; sy < shared_h; sy += blockDim.y) {
        for (int sx = tx; sx < shared_w; sx += blockDim.x) {
            int in_row = block_row + sy;
            int in_col = block_col + sx;
            float v = 0.0f;
            if (in_row < in_h && in_col < in_w) {
                v = input[in_row * in_w + in_col];
            }
            sh[sy * shared_w + sx] = v;
        }
    }
    __syncthreads();

    int out_row = block_row + ty;
    int out_col = block_col + tx;
    if (out_row >= out_h || out_col >= out_w) {
        return;
    }

    float acc = 0.0f;
    for (int ky = 0; ky < k_h; ++ky) {
        for (int kx = 0; kx < k_w; ++kx) {
            float a = sh[(ty + ky) * shared_w + (tx + kx)];
            float b = kernel[ky * k_w + kx];
            acc += a * b;
        }
    }

    out[out_row * out_w + out_col] = acc;
}
"#;
