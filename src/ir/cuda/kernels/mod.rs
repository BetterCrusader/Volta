pub mod add;
pub mod backward;
pub mod matmul;
pub mod reductions;
pub mod relu;
pub mod softmax;

use crate::ir::{KernelKind, NodeId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaKernel {
    Data,
    MatMul,
    Add,
    Relu,
    Softmax,
    Backward,
    Reduction,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendExecutableNode {
    pub kernel: CudaKernel,
    pub nodes: Vec<NodeId>,
}

pub fn dispatch_group(kind: KernelKind, nodes: &[NodeId]) -> Result<BackendExecutableNode, String> {
    let kernel = match kind {
        KernelKind::Data => CudaKernel::Data,
        KernelKind::MatMul => CudaKernel::MatMul,
        KernelKind::Add => CudaKernel::Add,
        KernelKind::Relu => CudaKernel::Relu,
        KernelKind::Softmax => CudaKernel::Softmax,
        KernelKind::Backward | KernelKind::Elementwise => CudaKernel::Backward,
        KernelKind::Conv2D | KernelKind::Control => {
            return Err(format!("unsupported CUDA kernel class: {:?}", kind));
        }
    };

    Ok(BackendExecutableNode {
        kernel,
        nodes: nodes.to_vec(),
    })
}

pub fn execute_node(node: &BackendExecutableNode) -> Result<(), String> {
    match node.kernel {
        CudaKernel::Data => Ok(()),
        CudaKernel::MatMul => matmul::run(&node.nodes),
        CudaKernel::Add => add::run(&node.nodes),
        CudaKernel::Relu => relu::run(&node.nodes),
        CudaKernel::Softmax => softmax::run(&node.nodes),
        CudaKernel::Backward => backward::run(&node.nodes),
        CudaKernel::Reduction => reductions::run(&node.nodes),
    }
}
