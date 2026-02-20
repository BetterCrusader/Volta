use crate::ir::{ExecutionPlan, KernelKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredCudaPlan {
    pub kernel_kinds: Vec<KernelKind>,
}

#[derive(Debug, Clone)]
pub struct CudaLoweringError {
    pub message: String,
}

pub fn lower_plan(plan: &ExecutionPlan) -> Result<LoweredCudaPlan, CudaLoweringError> {
    let mut lowered = Vec::with_capacity(plan.kernel_groups.len());

    for group in &plan.kernel_groups {
        match group.kind {
            KernelKind::Data => lowered.push(group.kind.clone()),
            KernelKind::Elementwise
            | KernelKind::MatMul
            | KernelKind::Conv2D
            | KernelKind::Control => {
                return Err(CudaLoweringError {
                    message: format!("unsupported CUDA kernel class: {:?}", group.kind),
                });
            }
        }
    }

    Ok(LoweredCudaPlan {
        kernel_kinds: lowered,
    })
}
