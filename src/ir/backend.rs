use crate::ir::ExecutionPlan;
use crate::ir::cuda::lower_plan;
use crate::ir::{BackendCapabilities, BackendKind, DeterminismLevel};

#[derive(Debug, Clone)]
pub struct CompiledProgram {
    pub schedule_len: usize,
    pub peak_bytes: usize,
    pub fingerprint: u64,
}

#[derive(Debug, Clone)]
pub struct BackendError {
    pub message: String,
}

pub trait Backend {
    fn capabilities(&self) -> BackendCapabilities;
    fn compile(&self, plan: &ExecutionPlan) -> Result<CompiledProgram, BackendError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend: BackendKind::Cpu,
            supports_inference: true,
            supports_training: true,
            supports_strict_determinism: true,
            default_determinism: DeterminismLevel::Strict,
        }
    }

    fn compile(&self, plan: &ExecutionPlan) -> Result<CompiledProgram, BackendError> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        plan.schedule.ordered_nodes.hash(&mut hasher);
        plan.allocation.peak_bytes.hash(&mut hasher);

        Ok(CompiledProgram {
            schedule_len: plan.schedule.ordered_nodes.len(),
            peak_bytes: plan.allocation.peak_bytes,
            fingerprint: hasher.finish(),
        })
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LlvmBackend;

impl Backend for LlvmBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend: BackendKind::Llvm,
            supports_inference: false,
            supports_training: false,
            supports_strict_determinism: false,
            default_determinism: DeterminismLevel::Balanced,
        }
    }

    fn compile(&self, _plan: &ExecutionPlan) -> Result<CompiledProgram, BackendError> {
        Err(BackendError {
            message: "LLVM backend is not implemented yet".to_string(),
        })
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CudaBackend;

impl Backend for CudaBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend: BackendKind::Cuda,
            supports_inference: true,
            supports_training: false,
            supports_strict_determinism: false,
            default_determinism: DeterminismLevel::Balanced,
        }
    }

    fn compile(&self, plan: &ExecutionPlan) -> Result<CompiledProgram, BackendError> {
        let lowered = lower_plan(plan).map_err(|err| BackendError {
            message: err.message,
        })?;

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        plan.schedule.ordered_nodes.hash(&mut hasher);
        plan.allocation.peak_bytes.hash(&mut hasher);
        lowered.kernel_kinds.hash(&mut hasher);

        Ok(CompiledProgram {
            schedule_len: plan.schedule.ordered_nodes.len(),
            peak_bytes: plan.allocation.peak_bytes,
            fingerprint: hasher.finish(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::ir::{Backend, CpuBackend, Graph, Op, build_execution_plan};

    #[test]
    fn cpu_backend_compiles_execution_plan() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Output(x))
            .expect("add op should succeed");

        let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");
        let backend = CpuBackend;
        let compiled = backend.compile(&plan).expect("compile should pass");
        assert_eq!(compiled.schedule_len, 2);
    }
}
