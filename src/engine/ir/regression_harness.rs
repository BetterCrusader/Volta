#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::ir::{
        AllocationPlan, ConstantFoldingPass, ElementwiseFusionPass, Graph, Op,
        TensorConstantPropagationPass, build_execution_plan, graph_fingerprint, plan_memory,
        print_graph, run_verified_pass, verify_graph,
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

        // After constant folding + DCE, ConstInt(1)+ConstInt(2) is folded to ConstInt(3).
        // The schedule covers only the surviving nodes (folded result + output).
        let schedule_snapshot = plan
            .schedule
            .ordered_nodes
            .iter()
            .map(|n| n.0.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert!(!schedule_snapshot.is_empty(), "schedule must not be empty");
        // Dead nodes (0,1) are eliminated; surviving nodes include the folded result and output.
        assert!(
            !schedule_snapshot.contains('0') && !schedule_snapshot.contains('1'),
            "dead constant nodes should be eliminated by DCE, got: {schedule_snapshot}"
        );

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

    #[test]
    fn pass_pipeline_preserves_ssa_and_liveness_sanity() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![4],
                    data: vec![1.0, -2.0, 3.0, -4.0],
                },
            )
            .expect("x");
        let (_, y) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![4],
                    data: vec![0.5, 0.5, 0.5, 0.5],
                },
            )
            .expect("y");
        let (_, add) = graph.add_op(block, Op::Add(x, y)).expect("add");
        let (_, relu) = graph.add_op(block, Op::Relu(add)).expect("relu");
        let (_, mul) = graph.add_op(block, Op::Mul(relu, y)).expect("mul");
        graph.add_op(block, Op::Output(mul)).expect("output");

        let mut fold = ConstantFoldingPass::new();
        let mut tensor_fold = TensorConstantPropagationPass::new();
        let mut fusion = ElementwiseFusionPass::new();
        run_verified_pass(&mut fold, &mut graph).expect("fold should pass");
        run_verified_pass(&mut tensor_fold, &mut graph).expect("tensor fold should pass");
        run_verified_pass(&mut fusion, &mut graph).expect("fusion should pass");

        verify_graph(&graph).expect("graph must remain SSA-valid after pass pipeline");

        let liveness = plan_memory(&graph).expect("liveness planning must succeed");
        assert!(!liveness.values.is_empty());
        let node_count = graph.nodes.len();
        for value in &liveness.values {
            assert!(value.start_node <= value.end_node);
            assert!(value.end_node < node_count);
        }

        let _plan = build_execution_plan(&graph, &std::collections::HashSet::new())
            .expect("execution plan must still build after pass pipeline");
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
