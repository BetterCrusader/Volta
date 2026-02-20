use crate::ir::cuda::LoweredCudaPlan;
use crate::ir::cuda::kernels::execute_node;

#[derive(Debug, Clone)]
pub struct CudaExecutionError {
    pub message: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CudaExecutor;

impl CudaExecutor {
    pub fn execute(&self, plan: &LoweredCudaPlan) -> Result<(), CudaExecutionError> {
        for node in &plan.executable_nodes {
            execute_node(node).map_err(|message| CudaExecutionError { message })?;
        }
        Ok(())
    }
}
