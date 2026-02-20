use std::collections::HashSet;

use volta::ir::cuda::{CudaKernel, lower_plan};
use volta::ir::{Graph, Op, ValueId, build_execution_plan, build_reverse_graph, graph_fingerprint};

#[test]
fn cuda_backward_lowering_includes_backward_kernels_without_mutating_forward_ir() {
    let (forward, loss, parameter) = build_relu_training_graph();
    let forward_fingerprint_before = graph_fingerprint(&forward);

    let reverse = build_reverse_graph(&forward, loss, &[parameter]).expect("autograd should pass");
    let gradient_values = reverse.gradients.values().copied().collect::<HashSet<_>>();
    let backward_plan =
        build_execution_plan(&reverse.backward, &gradient_values).expect("plan should build");

    let lowered = lower_plan(&backward_plan).expect("cuda backward lowering should pass");
    let kernels = lowered
        .executable_nodes
        .iter()
        .map(|node| node.kernel)
        .collect::<Vec<_>>();

    assert!(
        kernels.contains(&CudaKernel::Backward),
        "expected backward kernel dispatch in lowered plan"
    );

    let forward_fingerprint_after = graph_fingerprint(&forward);
    assert_eq!(
        forward_fingerprint_before, forward_fingerprint_after,
        "forward graph must not be mutated by reverse build or cuda lowering"
    );
}

#[test]
fn cuda_backward_lowering_maps_gradient_accumulation_to_reduction_kernel() {
    let (forward, loss, parameter) = build_accumulation_training_graph();

    let reverse = build_reverse_graph(&forward, loss, &[parameter]).expect("autograd should pass");
    let gradient_values = reverse.gradients.values().copied().collect::<HashSet<_>>();
    let backward_plan =
        build_execution_plan(&reverse.backward, &gradient_values).expect("plan should build");

    let lowered = lower_plan(&backward_plan).expect("cuda backward lowering should pass");
    let kernels = lowered
        .executable_nodes
        .iter()
        .map(|node| node.kernel)
        .collect::<Vec<_>>();

    assert!(
        kernels.contains(&CudaKernel::Reduction),
        "expected reduction kernel dispatch for gradient accumulation"
    );
}

fn build_relu_training_graph() -> (Graph, ValueId, ValueId) {
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
    let (_, act) = graph
        .add_op(block, Op::Relu(mm))
        .expect("add relu should succeed");
    let (_, loss) = graph
        .add_op(block, Op::Output(act))
        .expect("add output should succeed");

    graph.bind_input_shape("x", vec![1, 1]);
    graph.bind_parameter_shape("w", vec![1, 1]);

    (graph, loss, w)
}

fn build_accumulation_training_graph() -> (Graph, ValueId, ValueId) {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, w) = graph
        .add_op(block, Op::Parameter("w".to_string()))
        .expect("add parameter should succeed");
    let (_, sum) = graph
        .add_op(block, Op::Add(w, w))
        .expect("add add should succeed");
    let (_, loss) = graph
        .add_op(block, Op::Output(sum))
        .expect("add output should succeed");

    graph.bind_parameter_shape("w", vec![1, 1]);

    (graph, loss, w)
}
