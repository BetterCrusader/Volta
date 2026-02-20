use crate::ir::DeterminismLevel;
use crate::ir::cuda::{CudaKernel, LoweredCudaPlan};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeterminismPolicy {
    pub mode: DeterminismLevel,
    pub allow_atomics: bool,
    pub fixed_reduction_topology: bool,
    pub allow_tf32: bool,
    pub allow_fast_math: bool,
}

#[derive(Debug, Clone)]
pub struct CudaDeterminismError {
    pub message: String,
}

pub fn policy_for(level: DeterminismLevel) -> CudaDeterminismPolicy {
    match level {
        DeterminismLevel::Strict => CudaDeterminismPolicy {
            mode: level,
            allow_atomics: false,
            fixed_reduction_topology: true,
            allow_tf32: false,
            allow_fast_math: false,
        },
        DeterminismLevel::Balanced | DeterminismLevel::Fast => CudaDeterminismPolicy {
            mode: level,
            allow_atomics: true,
            fixed_reduction_topology: false,
            allow_tf32: true,
            allow_fast_math: true,
        },
    }
}

pub fn enforce_policy(
    plan: &LoweredCudaPlan,
    policy: CudaDeterminismPolicy,
) -> Result<(), CudaDeterminismError> {
    if policy.mode != DeterminismLevel::Strict {
        return Ok(());
    }

    for node in &plan.executable_nodes {
        if node.kernel == CudaKernel::Softmax
            && (node.nodes.len() != 1 || !policy.fixed_reduction_topology)
        {
            return Err(CudaDeterminismError {
                message: format!(
                    "strict mode requires fixed reduction topology for softmax (group size must be 1, got {})",
                    node.nodes.len()
                ),
            });
        }

        if node.kernel == CudaKernel::Data {
            continue;
        }
        if !policy.allow_atomics && kernel_uses_atomics(node.kernel) {
            return Err(CudaDeterminismError {
                message: format!(
                    "strict mode forbids atomics for CUDA kernel {:?}",
                    node.kernel
                ),
            });
        }
        if !policy.allow_tf32 && kernel_uses_tf32(node.kernel) {
            return Err(CudaDeterminismError {
                message: format!("strict mode forbids TF32 for CUDA kernel {:?}", node.kernel),
            });
        }
        if !policy.allow_fast_math && kernel_uses_fast_math(node.kernel) {
            return Err(CudaDeterminismError {
                message: format!(
                    "strict mode forbids fast-math for CUDA kernel {:?}",
                    node.kernel
                ),
            });
        }
    }

    Ok(())
}

fn kernel_uses_atomics(_kernel: CudaKernel) -> bool {
    false
}

fn kernel_uses_tf32(_kernel: CudaKernel) -> bool {
    false
}

fn kernel_uses_fast_math(_kernel: CudaKernel) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use crate::ir::cuda::{
        BackendExecutableNode, CudaKernel, LoweredCudaPlan, enforce_policy, policy_for,
    };
    use crate::ir::{DeterminismLevel, NodeId};

    #[test]
    fn strict_policy_rejects_fused_softmax_groups() {
        let plan = LoweredCudaPlan {
            executable_nodes: vec![BackendExecutableNode {
                kernel: CudaKernel::Softmax,
                nodes: vec![NodeId(1), NodeId(2)],
            }],
            memory_bindings: Vec::new(),
        };

        let err = enforce_policy(&plan, policy_for(DeterminismLevel::Strict))
            .expect_err("strict policy should reject fused softmax groups");
        assert!(err.message.contains("fixed reduction topology for softmax"));
    }

    #[test]
    fn strict_policy_accepts_supported_single_node_kernels() {
        let plan = LoweredCudaPlan {
            executable_nodes: vec![
                BackendExecutableNode {
                    kernel: CudaKernel::MatMul,
                    nodes: vec![NodeId(1)],
                },
                BackendExecutableNode {
                    kernel: CudaKernel::Add,
                    nodes: vec![NodeId(2)],
                },
                BackendExecutableNode {
                    kernel: CudaKernel::Relu,
                    nodes: vec![NodeId(3)],
                },
                BackendExecutableNode {
                    kernel: CudaKernel::Softmax,
                    nodes: vec![NodeId(4)],
                },
            ],
            memory_bindings: Vec::new(),
        };

        enforce_policy(&plan, policy_for(DeterminismLevel::Strict))
            .expect("strict policy should accept current deterministic kernel subset");
    }
}
