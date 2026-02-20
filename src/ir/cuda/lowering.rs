use crate::ir::ExecutionPlan;
use crate::ir::cuda::kernels::{BackendExecutableNode, dispatch_group};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredCudaPlan {
    pub executable_nodes: Vec<BackendExecutableNode>,
}

#[derive(Debug, Clone)]
pub struct CudaLoweringError {
    pub message: String,
}

pub fn lower_plan(plan: &ExecutionPlan) -> Result<LoweredCudaPlan, CudaLoweringError> {
    let mut lowered = Vec::with_capacity(plan.kernel_groups.len());

    for group in &plan.kernel_groups {
        let executable = dispatch_group(group.kind.clone(), &group.nodes)
            .map_err(|message| CudaLoweringError { message })?;
        lowered.push(executable);
    }

    Ok(LoweredCudaPlan {
        executable_nodes: lowered,
    })
}
