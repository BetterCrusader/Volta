use std::collections::HashMap;

use crate::ir::{
    ElementwiseUnaryOp, Graph, Op, Pass, ShapeFact, ValueId, infer_shapes, run_with_verifier_guard,
};

#[derive(Default)]
pub struct ElementwiseFusionPass;

impl ElementwiseFusionPass {
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
        _ => None,
    }
}

fn chain_from_op(op: &Op) -> Option<(ValueId, Vec<ElementwiseUnaryOp>)> {
    match op {
        Op::Neg(input) => Some((*input, vec![ElementwiseUnaryOp::Neg])),
        Op::Relu(input) => Some((*input, vec![ElementwiseUnaryOp::Relu])),
        Op::ElementwiseChain { input, ops } => Some((*input, ops.clone())),
        _ => None,
    }
}

fn shape_is_tensor(shapes: &HashMap<ValueId, ShapeFact>, value: ValueId) -> bool {
    matches!(shapes.get(&value), Some(ShapeFact::Tensor(_)))
}

#[cfg(test)]
mod tests {
    use crate::ir::{ElementwiseFusionPass, Graph, Op, Pass, RuntimeValue, execute};

    #[test]
    fn fuses_relu_then_neg_chain() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![3],
                    data: vec![-1.0, 2.0, -3.0],
                },
            )
            .expect("add op should succeed");
        let (_, relu) = graph
            .add_op(block, Op::Relu(input))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Neg(relu))
            .expect("add op should succeed");

        let before = execute(&graph).expect("execute should pass");

        let mut pass = ElementwiseFusionPass::new();
        pass.run(&mut graph);

        assert!(matches!(graph.nodes[1].op, Op::Removed));
        assert!(matches!(graph.nodes[2].op, Op::ElementwiseChain { .. }));

        let after = execute(&graph).expect("execute should pass");
        assert_eq!(before, after);
        assert_eq!(
            after,
            Some(RuntimeValue::Tensor {
                shape: vec![3],
                data: vec![0.0, -2.0, 0.0],
            })
        );
    }
}
