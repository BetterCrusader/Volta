use volta::ir::{Graph, Op, build_schedule, optimize_schedule, verify_schedule};

#[test]
fn schedule_optimization_is_deterministic_and_preserves_validity() {
    let mut graph = Graph::new();
    let block = graph.create_block();
    let (_, x) = graph
        .add_op(block, Op::Input("x".to_string()))
        .expect("add input should succeed");
    let (_, w) = graph
        .add_op(block, Op::Parameter("w".to_string()))
        .expect("add parameter should succeed");
    let (_, mm) = graph
        .add_op(block, Op::MatMul(x, w))
        .expect("add matmul should succeed");
    let (_, relu) = graph
        .add_op(block, Op::Relu(mm))
        .expect("add relu should succeed");
    graph
        .add_op(block, Op::Output(relu))
        .expect("add output should succeed");
    graph.bind_input_shape("x", vec![1, 2]);
    graph.bind_parameter_shape("w", vec![2, 2]);

    let base = build_schedule(&graph).expect("base schedule should build");
    let first = optimize_schedule(&graph, &base).expect("opt pass should succeed");
    let second = optimize_schedule(&graph, &base).expect("opt pass should be deterministic");

    assert_eq!(first.schedule.ordered_nodes, second.schedule.ordered_nodes);
    assert_eq!(first.schedule.ordered_nodes.len(), base.ordered_nodes.len());
    verify_schedule(&graph, &first.schedule).expect("optimized schedule must verify");
}
