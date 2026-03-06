use crate::ir::{Graph, NodeId, Op, Schedule, verify_schedule};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KernelKind {
    Add,
    Relu,
    Softmax,
    Backward,
    Elementwise,
    MatMul,
    Conv2D,
    Data,
    Control,
}

#[derive(Debug, Clone)]
pub struct KernelGroup {
    pub kind: KernelKind,
    pub nodes: Vec<NodeId>,
}

#[derive(Debug, Clone)]
pub struct KernelGroupingError {
    pub message: String,
}

pub fn group_kernels(
    graph: &Graph,
    schedule: &Schedule,
) -> Result<Vec<KernelGroup>, KernelGroupingError> {
    verify_schedule(graph, schedule).map_err(|err| KernelGroupingError {
        message: err.message,
    })?;

    let mut groups = Vec::<KernelGroup>::new();
    for node_id in &schedule.ordered_nodes {
        let node = &graph.nodes[node_id.0];
        let kind = classify_op(&node.op);

        if let Some(last) = groups.last_mut()
            && last.kind == kind
        {
            last.nodes.push(*node_id);
            continue;
        }

        groups.push(KernelGroup {
            kind,
            nodes: vec![*node_id],
        });
    }

    Ok(groups)
}

fn classify_op(op: &Op) -> KernelKind {
    match op {
        Op::Add(_, _) => KernelKind::Add,
        Op::Relu(_) => KernelKind::Relu,
        Op::Softmax(_) => KernelKind::Softmax,
        Op::ReluBackward(_, _)
        | Op::SigmoidBackward(_, _)
        | Op::GeluBackward(_, _)
        | Op::GeluExactBackward(_, _)
        | Op::ReduceMaxBackward { .. }
        | Op::GemmBackward { .. } => KernelKind::Backward,
        Op::Sub(_, _)
        | Op::Mul(_, _)
        | Op::Div(_, _)
        | Op::Neg(_)
        | Op::Log(_)
        | Op::Exp(_)
        | Op::Sigmoid(_)
        | Op::GeluExact(_)
        | Op::Gelu(_)
        | Op::ElementwiseChain { .. }
        | Op::Reshape { .. }
        | Op::Concat { .. }
        | Op::Gather { .. }
        | Op::Slice { .. }
        | Op::ReduceSum { .. }
        | Op::ReduceMax { .. }
        | Op::ReduceMean { .. }
        | Op::MaxPool { .. }
        | Op::AvgPool { .. }
        | Op::BatchNorm { .. }
        | Op::Flatten { .. }
        | Op::GlobalAveragePool { .. }
        | Op::GlobalAveragePoolBackward { .. }
        | Op::GroupNorm { .. }
        | Op::GroupNormBackwardInput { .. }
        | Op::GroupNormBackwardWeight { .. }
        | Op::GroupNormBackwardBias { .. }
        | Op::InstanceNorm { .. }
        | Op::InstanceNormBackwardInput { .. }
        | Op::InstanceNormBackwardWeight { .. }
        | Op::InstanceNormBackwardBias { .. }
        | Op::Embedding { .. }
        | Op::EmbeddingBackward { .. }
        | Op::LstmCell { .. }
        | Op::LstmCellBackward { .. }
        | Op::GruCell { .. }
        | Op::GruCellBackward { .. }
        | Op::ConvTranspose2D { .. }
        | Op::Upsample2D { .. }
        | Op::Upsample2DBackward { .. }
        | Op::MultiHeadAttention { .. }
        | Op::SinusoidalPE { .. }
        | Op::RoPE { .. }
        | Op::RoPEBackward { .. }
        | Op::Dropout { .. }
        | Op::Identity(_)
        | Op::LayerNorm { .. } => KernelKind::Elementwise,
        Op::MatMul(_, _) | Op::Gemm { .. } => KernelKind::MatMul,
        Op::Conv2D(_, _)
        | Op::Conv2DBackwardInput(_, _, _)
        | Op::Conv2DBackwardWeight(_, _, _) => KernelKind::Conv2D,
        Op::ConstInt(_)
        | Op::ConstFloat(_)
        | Op::ConstTensor { .. }
        | Op::Input(_)
        | Op::Parameter(_)
        | Op::Output(_)
        | Op::Transpose(_) => KernelKind::Data,
        Op::Phi(_) | Op::Removed => KernelKind::Control,
        Op::SoftmaxCrossEntropyLossFromLogits { .. } => KernelKind::Backward, // Or Elementwise, it's computed as a single kernel
        Op::MaxPoolBackward { .. }
        | Op::AvgPoolBackward { .. }
        | Op::BatchNormBackwardInput { .. }
        | Op::BatchNormBackwardWeight { .. }
        | Op::BatchNormBackwardBias { .. }
        | Op::LayerNormBackwardInput { .. }
        | Op::LayerNormBackwardWeight { .. }
        | Op::LayerNormBackwardBias { .. } => KernelKind::Backward,
        _ => KernelKind::Control,
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, build_schedule, group_kernels};

    #[test]
    fn groups_contiguous_nodes_by_kernel_kind() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        let (_, c) = graph
            .add_op(block, Op::Add(a, b))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Mul(c, b))
            .expect("add op should succeed");

        let schedule = build_schedule(&graph).expect("schedule should pass");
        let groups = group_kernels(&graph, &schedule).expect("grouping should pass");
        assert!(groups.len() >= 2);
    }
}
