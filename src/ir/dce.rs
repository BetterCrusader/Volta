use crate::ir::{Graph, Op, Pass, ValueId, run_with_verifier_guard};

#[derive(Default)]
pub struct DcePass;

impl DcePass {
    pub fn new() -> Self {
        Self
    }
}

impl Pass for DcePass {
    fn run(&mut self, graph: &mut Graph) {
        run_with_verifier_guard(graph, |graph| {
            let roots = root_value_ids(graph);
            let value_to_node = value_to_node_index(graph);
            let mut live = vec![false; graph.nodes.len()];

            for root in roots {
                mark_live_from_value(root, graph, &value_to_node, &mut live);
            }

            for (index, node) in graph.nodes.iter_mut().enumerate() {
                if !live[index] {
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
    use crate::ir::{
        ConstantFoldingPass, DcePass, Graph, Op, Pass, RuntimeValue, execute, execute_value,
    };

    #[test]
    fn removes_unused_constant_node() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(3))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::ConstInt(999))
            .expect("add op should succeed");
        let (_, sum) = graph
            .add_op(block, Op::Add(a, b))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Output(sum))
            .expect("add op should succeed");

        let mut pass = DcePass::new();
        pass.run(&mut graph);

        assert!(matches!(graph.nodes[2].op, Op::Removed));
        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(result, Some(RuntimeValue::Int(5)));
    }

    #[test]
    fn works_after_constant_folding() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(10))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(5))
            .expect("add op should succeed");
        let (_, mid) = graph
            .add_op(block, Op::Add(a, b))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Mul(mid, b))
            .expect("add op should succeed");

        let mut fold = ConstantFoldingPass::new();
        fold.run(&mut graph);

        let before = execute(&graph).expect("execute should succeed");
        let mut dce = DcePass::new();
        dce.run(&mut graph);
        let after = execute(&graph).expect("execute should succeed");
        assert_eq!(before, after);
    }

    #[test]
    fn execute_value_still_works_for_live_non_terminal_value() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(3))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(4))
            .expect("add op should succeed");
        let (_, mid) = graph
            .add_op(block, Op::Mul(a, b))
            .expect("add op should succeed");
        let (_, out) = graph
            .add_op(block, Op::Add(mid, a))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Output(out))
            .expect("add op should succeed");

        let mut pass = DcePass::new();
        pass.run(&mut graph);

        let mid_value = execute_value(&graph, mid).expect("execute_value should succeed");
        assert_eq!(mid_value, RuntimeValue::Int(12));
    }
}
