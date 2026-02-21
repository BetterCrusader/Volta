use crate::ir::DeterminismLevel;
use crate::ir::NodeId;
use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::memory::DeviceBuffer;

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::CompileOptions;

pub fn run(nodes: &[NodeId]) -> Result<(), String> {
    if nodes.is_empty() {
        return Err("backward kernel dispatch received no nodes".to_string());
    }
    Ok(())
}

pub fn transpose_2d_f32(
    device: &CudaDevice,
    input: &[f32],
    rows: usize,
    cols: usize,
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    if input.len() != rows.saturating_mul(cols) {
        return Err(format!(
            "transpose input length mismatch: len={} expected={} (rows={} cols={})",
            input.len(),
            rows.saturating_mul(cols),
            rows,
            cols
        ));
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let stream = device.stream().map_err(|err| err.message)?;
    let input_device = DeviceBuffer::from_host(device, input).map_err(|err| err.message)?;
    let mut out_device = DeviceBuffer::zeros(device, input.len()).map_err(|err| err.message)?;

    let function = device
        .load_or_get_function(
            "backward_module",
            "transpose_2d_kernel",
            BACKWARD_KERNELS_CU,
            compile_options(device, determinism),
        )
        .map_err(|err| err.message)?;

    let rows_i32 = i32::try_from(rows).map_err(|_| format!("rows does not fit i32: {rows}"))?;
    let cols_i32 = i32::try_from(cols).map_err(|_| format!("cols does not fit i32: {cols}"))?;
    let cfg = LaunchConfig {
        grid_dim: ((cols as u32).div_ceil(16), (rows as u32).div_ceil(16), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&function)
            .arg(out_device.cuda_slice_mut())
            .arg(input_device.cuda_slice())
            .arg(&rows_i32)
            .arg(&cols_i32)
            .launch(cfg)
            .map_err(|err| format!("CUDA kernel launch failed for transpose: {err}"))?;
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

const BACKWARD_KERNELS_CU: &str = r#"
extern "C" __global__ void transpose_2d_kernel(
    float* out,
    const float* input,
    int rows,
    int cols
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        out[x * rows + y] = input[y * cols + x];
    }
}
"#;
