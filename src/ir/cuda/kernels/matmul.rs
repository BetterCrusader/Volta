use crate::ir::DeterminismLevel;
use crate::ir::NodeId;
use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::memory::DeviceBuffer;

use cudarc::cublas::{Gemm, GemmConfig, sys};
use cudarc::cublas::{result::CublasError, sys::cublasMath_t};

pub fn run(nodes: &[NodeId]) -> Result<(), String> {
    if nodes.is_empty() {
        return Err("matmul kernel dispatch received no nodes".to_string());
    }
    Ok(())
}

pub fn matmul_f32(
    device: &CudaDevice,
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    n: usize,
    k: usize,
    determinism: DeterminismLevel,
) -> Result<Vec<f32>, String> {
    if m == 0 || n == 0 {
        return Ok(vec![0.0; m.saturating_mul(n)]);
    }
    if k == 0 {
        return Ok(vec![0.0; m.saturating_mul(n)]);
    }
    if lhs.len() != m.saturating_mul(k) {
        return Err(format!(
            "matmul lhs length mismatch: len={} expected={} (m={} k={})",
            lhs.len(),
            m.saturating_mul(k),
            m,
            k
        ));
    }
    if rhs.len() != k.saturating_mul(n) {
        return Err(format!(
            "matmul rhs length mismatch: len={} expected={} (k={} n={})",
            rhs.len(),
            k.saturating_mul(n),
            k,
            n
        ));
    }

    let lhs_device = DeviceBuffer::from_host(device, lhs).map_err(|err| err.message)?;
    let rhs_device = DeviceBuffer::from_host(device, rhs).map_err(|err| err.message)?;
    let mut out_device =
        DeviceBuffer::zeros(device, m.saturating_mul(n)).map_err(|err| err.message)?;

    if !lhs_device.is_256_aligned() || !rhs_device.is_256_aligned() || !out_device.is_256_aligned()
    {
        return Err(
            "CUDA device pointer alignment check failed (expected 256-byte alignment)".to_string(),
        );
    }

    device
        .with_cublas(|blas| {
            let math_mode = match determinism {
                DeterminismLevel::Strict => cublasMath_t::CUBLAS_DEFAULT_MATH,
                DeterminismLevel::Balanced | DeterminismLevel::Fast => {
                    cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH
                }
            };

            set_cublas_math_mode(blas, math_mode).map_err(|err| {
                crate::ir::cuda::device::CudaDeviceError {
                    message: format!("cuBLAS math mode setup failed: {err}"),
                }
            })?;

            let cfg = GemmConfig {
                transa: sys::cublasOperation_t::CUBLAS_OP_N,
                transb: sys::cublasOperation_t::CUBLAS_OP_N,
                m: i32::try_from(n).map_err(|_| crate::ir::cuda::device::CudaDeviceError {
                    message: format!("n dimension does not fit i32: {n}"),
                })?,
                n: i32::try_from(m).map_err(|_| crate::ir::cuda::device::CudaDeviceError {
                    message: format!("m dimension does not fit i32: {m}"),
                })?,
                k: i32::try_from(k).map_err(|_| crate::ir::cuda::device::CudaDeviceError {
                    message: format!("k dimension does not fit i32: {k}"),
                })?,
                alpha: 1.0_f32,
                lda: i32::try_from(n).map_err(|_| crate::ir::cuda::device::CudaDeviceError {
                    message: format!("lda does not fit i32: {n}"),
                })?,
                ldb: i32::try_from(k).map_err(|_| crate::ir::cuda::device::CudaDeviceError {
                    message: format!("ldb does not fit i32: {k}"),
                })?,
                beta: 0.0_f32,
                ldc: i32::try_from(n).map_err(|_| crate::ir::cuda::device::CudaDeviceError {
                    message: format!("ldc does not fit i32: {n}"),
                })?,
            };

            unsafe {
                blas.gemm(
                    cfg,
                    rhs_device.cuda_slice(),
                    lhs_device.cuda_slice(),
                    out_device.cuda_slice_mut(),
                )
            }
            .map_err(|err| crate::ir::cuda::device::CudaDeviceError {
                message: format!("cuBLAS sgemm failed: {err}"),
            })
        })
        .map_err(|err| err.message)?;

    out_device.copy_to_host(device).map_err(|err| err.message)
}

fn set_cublas_math_mode(
    blas: &mut cudarc::cublas::CudaBlas,
    mode: cublasMath_t,
) -> Result<(), CublasError> {
    unsafe { cudarc::cublas::sys::cublasSetMathMode(*blas.handle(), mode).result() }
}
