use crate::ir::cuda::kernels::{BackendExecutableNode, CudaKernel, dispatch_group};
use crate::ir::{ExecutionPlan, PlacementClass, ValueId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredCudaPlan {
    pub executable_nodes: Vec<BackendExecutableNode>,
    pub memory_bindings: Vec<LoweredMemoryBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaMemoryClass {
    Input,
    Parameter,
    Temporary,
    Output,
    Gradient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoweredMemoryBinding {
    pub value: ValueId,
    pub class: CudaMemoryClass,
}

#[derive(Debug, Clone)]
pub struct CudaLoweringError {
    pub message: String,
}

pub fn lower_plan(plan: &ExecutionPlan) -> Result<LoweredCudaPlan, CudaLoweringError> {
    let mut lowered = Vec::with_capacity(plan.kernel_groups.len());
    let gradient_plan = plan
        .placement_hints
        .iter()
        .any(|hint| hint.class == PlacementClass::Gradient);

    for group in &plan.kernel_groups {
        let mut executable = dispatch_group(group.kind.clone(), &group.nodes)
            .map_err(|message| CudaLoweringError { message })?;

        if gradient_plan && executable.kernel == CudaKernel::Add {
            executable.kernel = CudaKernel::Reduction;
        }
        lowered.push(executable);
    }

    let memory_bindings = plan
        .placement_hints
        .iter()
        .map(|hint| LoweredMemoryBinding {
            value: hint.value,
            class: map_placement_class(hint.class),
        })
        .collect::<Vec<_>>();

    Ok(LoweredCudaPlan {
        executable_nodes: lowered,
        memory_bindings,
    })
}

fn map_placement_class(class: PlacementClass) -> CudaMemoryClass {
    match class {
        PlacementClass::Input => CudaMemoryClass::Input,
        PlacementClass::Parameter => CudaMemoryClass::Parameter,
        PlacementClass::Temporary => CudaMemoryClass::Temporary,
        PlacementClass::Output => CudaMemoryClass::Output,
        PlacementClass::Gradient => CudaMemoryClass::Gradient,
    }
}
