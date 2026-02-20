#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::ir::{
        AllocationPlan, ConstantFoldingPass, ElementwiseFusionPass, Graph, Op,
        TensorConstantPropagationPass, build_execution_plan, graph_fingerprint, print_graph,
        run_verified_pass,
    };

    #[test]
    fn snapshot_ir_schedule_and_allocation() {
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
            .add_op(block, Op::Output(c))
            .expect("add op should succeed");

        let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should pass");

        let ir_snapshot = print_graph(&graph);
        assert_eq!(
            ir_snapshot,
            "%0 = const 1\n%1 = const 2\n%2 = add %0 %1\n%3 = output %2"
        );

        let schedule_snapshot = plan
            .schedule
            .ordered_nodes
            .iter()
            .map(|n| n.0.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(schedule_snapshot, "0,1,2,3");

        let allocation_snapshot = allocation_signature(&plan.allocation);
        assert!(!allocation_snapshot.is_empty());
    }

    #[test]
    fn fingerprint_is_stable_after_idempotent_passes() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2],
                    data: vec![1.0, -2.0],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Relu(x))
            .expect("add op should succeed");

        let mut fold = ConstantFoldingPass::new();
        run_verified_pass(&mut fold, &mut graph).expect("fold should pass");
        let mut tensor_fold = TensorConstantPropagationPass::new();
        run_verified_pass(&mut tensor_fold, &mut graph).expect("tensor fold should pass");
        let mut fusion = ElementwiseFusionPass::new();
        run_verified_pass(&mut fusion, &mut graph).expect("fusion should pass");

        let first = graph_fingerprint(&graph);
        run_verified_pass(&mut tensor_fold, &mut graph).expect("tensor fold should pass");
        run_verified_pass(&mut fusion, &mut graph).expect("fusion should pass");
        let second = graph_fingerprint(&graph);
        assert_eq!(first, second);
    }

    fn allocation_signature(plan: &AllocationPlan) -> String {
        let mut entries = plan.buffer_map.iter().collect::<Vec<_>>();
        entries.sort_by_key(|(value, _)| value.0);
        entries
            .into_iter()
            .map(|(value, buffer)| format!("v{}->b{}", value.0, buffer.0))
            .collect::<Vec<_>>()
            .join(";")
    }
}
