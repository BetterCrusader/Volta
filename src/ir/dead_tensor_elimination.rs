use crate::ir::{Graph, Op, Pass, ShapeFact, ValueId, infer_shapes, run_with_verifier_guard};

#[derive(Default)]
pub struct DeadTensorEliminationPass;

impl DeadTensorEliminationPass {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Pass for DeadTensorEliminationPass {
    fn run(&mut self, graph: &mut Graph) {
        run_with_verifier_guard(graph, |graph| {
            let Ok(shapes) = infer_shapes(graph) else {
                return;
            };

            let roots = root_value_ids(graph);
            let value_to_node = value_to_node_index(graph);
            let mut live = vec![false; graph.nodes.len()];

            for root in roots {
                mark_live_from_value(root, graph, &value_to_node, &mut live);
            }

            for (index, node) in graph.nodes.iter_mut().enumerate() {
                if live[index] {
                    continue;
                }

                if matches!(shapes.get(&node.output), Some(ShapeFact::Tensor(_))) {
                    node.op = Op::Removed;
                }
            }
        });
    }
}

fn root_value_ids(graph: &Graph) -> Vec<ValueId> {
    let mut roots = Vec::new();
    for node in &graph.nodes {
        if let Op::Output(_) = node.op {
            roots.push(node.output);
        }
    }
    if roots.is_empty()
        && let Some(last) = graph.last_value_id()
    {
        roots.push(last);
    }
    roots
}

fn value_to_node_index(graph: &Graph) -> Vec<Option<usize>> {
    let mut table = vec![None; graph.value_count()];
    for (index, node) in graph.nodes.iter().enumerate() {
        if node.output.0 < table.len() {
            table[node.output.0] = Some(index);
        }
    }
    table
}

fn mark_live_from_value(
    value: ValueId,
    graph: &Graph,
    value_to_node: &[Option<usize>],
    live: &mut [bool],
) {
    let Some(Some(node_index)) = value_to_node.get(value.0) else {
        return;
    };

    if live[*node_index] {
        return;
    }
    live[*node_index] = true;

    let node = &graph.nodes[*node_index];
    for input in node.op.input_values() {
        mark_live_from_value(input, graph, value_to_node, live);
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{DeadTensorEliminationPass, Graph, Op, Pass, RuntimeValue, execute};

    #[test]
    fn removes_dead_tensor_nodes_only() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        graph
            .add_op(block, Op::ConstInt(123))
            .expect("add op should succeed");
        graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2],
                    data: vec![1.0, 2.0],
                },
            )
            .expect("add op should succeed");
        let (_, live) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2],
                    data: vec![3.0, 4.0],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Output(live))
            .expect("add op should succeed");

        let mut pass = DeadTensorEliminationPass::new();
        pass.run(&mut graph);

        assert!(!matches!(graph.nodes[0].op, Op::Removed));
        assert!(matches!(graph.nodes[1].op, Op::Removed));
        let result = execute(&graph).expect("execute should pass");
        assert_eq!(
            result,
            Some(RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![2], vec![3.0, 4.0]).unwrap()
            )))
        );
    }
}
