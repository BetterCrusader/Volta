use crate::ir::cuda::LoweredCudaPlan;

#[derive(Debug, Clone)]
pub struct CudaExecutionError {
    pub message: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CudaExecutor;

impl CudaExecutor {
    pub fn execute(&self, _plan: &LoweredCudaPlan) -> Result<(), CudaExecutionError> {
        Err(CudaExecutionError {
            message: "CUDA executor scaffold is inference-only and not runnable yet".to_string(),
        })
    }
}
