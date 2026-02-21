use crate::ir::cuda::kernels::{BackendExecutableNode, CudaKernel, dispatch_group};
use crate::ir::{CompilerFlags, DeterminismLevel, ExecutionPlan, PlacementClass, ValueId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredCudaPlan {
    pub executable_nodes: Vec<BackendExecutableNode>,
    pub memory_bindings: Vec<LoweredMemoryBinding>,
    pub workspace_buffers: Vec<CudaWorkspaceBuffer>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaWorkspaceBuffer {
    pub id: usize,
    pub bytes: usize,
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
    let strict_mode = CompilerFlags::from_env().determinism == DeterminismLevel::Strict;

    for group in &plan.kernel_groups {
        let mut executable = dispatch_group(group.kind.clone(), &group.nodes)
            .map_err(|message| CudaLoweringError { message })?;

        if gradient_plan && executable.kernel == CudaKernel::Add {
            executable.kernel = CudaKernel::Reduction;
        }

        if strict_mode && executable.kernel == CudaKernel::Reduction && executable.nodes.len() > 1 {
            for node in &executable.nodes {
                lowered.push(BackendExecutableNode {
                    kernel: CudaKernel::Reduction,
                    nodes: vec![*node],
                });
            }
            continue;
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

    let mut workspace_buffers = Vec::new();
    let mut next_workspace_id = 0usize;
    for node in &lowered {
        if !matches!(node.kernel, CudaKernel::Backward | CudaKernel::Reduction) {
            continue;
        }

        workspace_buffers.push(CudaWorkspaceBuffer {
            id: next_workspace_id,
            bytes: node.nodes.len().max(1) * 16,
        });
        next_workspace_id += 1;
    }

    Ok(LoweredCudaPlan {
        executable_nodes: lowered,
        memory_bindings,
        workspace_buffers,
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
