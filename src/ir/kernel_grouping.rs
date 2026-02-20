use crate::ir::{Graph, NodeId, Op, Schedule, verify_schedule};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KernelKind {
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
        Op::Add(_, _)
        | Op::Sub(_, _)
        | Op::Mul(_, _)
        | Op::Div(_, _)
        | Op::Neg(_)
        | Op::ElementwiseChain { .. }
        | Op::Relu(_)
        | Op::ReluBackward(_, _)
        | Op::Softmax(_) => KernelKind::Elementwise,
        Op::MatMul(_, _) => KernelKind::MatMul,
        Op::Conv2D(_, _) => KernelKind::Conv2D,
        Op::ConstInt(_)
        | Op::ConstFloat(_)
        | Op::ConstTensor { .. }
        | Op::Input(_)
        | Op::Parameter(_)
        | Op::Output(_)
        | Op::Transpose(_) => KernelKind::Data,
        Op::Phi(_) | Op::Removed => KernelKind::Control,
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
