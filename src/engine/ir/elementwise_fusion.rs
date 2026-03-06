use std::collections::HashMap;

use crate::ir::{
    ElementwiseUnaryOp, Graph, Op, Pass, ShapeFact, ValueId, infer_shapes, run_with_verifier_guard,
};

#[derive(Default)]
pub struct ElementwiseFusionPass;

impl ElementwiseFusionPass {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Pass for ElementwiseFusionPass {
    fn run(&mut self, graph: &mut Graph) {
        run_with_verifier_guard(graph, |graph| {
            let Ok(shapes) = infer_shapes(graph) else {
                return;
            };

            let producer = build_producer_map(graph);
            let mut use_count = build_use_count(graph);

            for index in 0..graph.nodes.len() {
                let Some((current_input, current_op)) = unary_from_op(&graph.nodes[index].op)
                else {
                    continue;
                };

                let Some(pred_index) = producer.get(&current_input).copied() else {
                    continue;
                };
                if pred_index >= index || use_count.get(&current_input).copied().unwrap_or(0) != 1 {
                    continue;
                }

                let pred_node = &graph.nodes[pred_index];
                let Some((root_input, mut fused_ops)) = chain_from_op(&pred_node.op) else {
                    continue;
                };

                if !shape_is_tensor(&shapes, root_input) || !shape_is_tensor(&shapes, current_input)
                {
                    continue;
                }

                fused_ops.push(current_op);
                graph.nodes[index].op = Op::ElementwiseChain {
                    input: root_input,
                    ops: fused_ops,
                };
                graph.nodes[pred_index].op = Op::Removed;

                if let Some(count) = use_count.get_mut(&root_input) {
                    *count += 1;
                }
                if let Some(count) = use_count.get_mut(&current_input) {
                    *count = count.saturating_sub(1);
                }
            }
        });
    }

    fn name(&self) -> &str {
        "ElementwiseFusionPass"
    }
}

fn build_producer_map(graph: &Graph) -> HashMap<ValueId, usize> {
    let mut producer = HashMap::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        producer.insert(node.output, index);
    }
    producer
}

fn build_use_count(graph: &Graph) -> HashMap<ValueId, usize> {
    let mut count = HashMap::new();
    for node in &graph.nodes {
        for input in node.op.input_values() {
            *count.entry(input).or_insert(0) += 1;
        }
    }
    count
}

fn unary_from_op(op: &Op) -> Option<(ValueId, ElementwiseUnaryOp)> {
    match op {
        Op::Neg(input) => Some((*input, ElementwiseUnaryOp::Neg)),
        Op::Relu(input) => Some((*input, ElementwiseUnaryOp::Relu)),
        Op::Sigmoid(input) => Some((*input, ElementwiseUnaryOp::Sigmoid)),
        Op::Gelu(input) => Some((*input, ElementwiseUnaryOp::Gelu)),
        Op::GeluExact(input) => Some((*input, ElementwiseUnaryOp::GeluExact)),
        Op::Exp(input) => Some((*input, ElementwiseUnaryOp::Exp)),
        Op::Log(input) => Some((*input, ElementwiseUnaryOp::Log)),
        _ => None,
    }
}

fn chain_from_op(op: &Op) -> Option<(ValueId, Vec<ElementwiseUnaryOp>)> {
    match op {
        Op::Neg(input) => Some((*input, vec![ElementwiseUnaryOp::Neg])),
        Op::Relu(input) => Some((*input, vec![ElementwiseUnaryOp::Relu])),
        Op::Sigmoid(input) => Some((*input, vec![ElementwiseUnaryOp::Sigmoid])),
        Op::Gelu(input) => Some((*input, vec![ElementwiseUnaryOp::Gelu])),
        Op::GeluExact(input) => Some((*input, vec![ElementwiseUnaryOp::GeluExact])),
        Op::Exp(input) => Some((*input, vec![ElementwiseUnaryOp::Exp])),
        Op::Log(input) => Some((*input, vec![ElementwiseUnaryOp::Log])),
        Op::ElementwiseChain { input, ops } => Some((*input, ops.clone())),
        _ => None,
    }
}

fn shape_is_tensor(shapes: &HashMap<ValueId, ShapeFact>, value: ValueId) -> bool {
    matches!(shapes.get(&value), Some(ShapeFact::Tensor(_)))
}
