#[path = "common/cuda.rs"]
mod cuda_helpers;

use std::collections::HashSet;

use volta::ir::{
    CudaBackend, ExecutionContext, Graph, Op, RuntimeValue, build_execution_plan,
    execute_terminal_with_backend,
};

#[test]
fn cuda_inference_is_repeatable_in_strict_mode() {
    if !cuda_helpers::cuda_runtime_available() {
        eprintln!("[SKIP] cuda_inference_is_repeatable_in_strict_mode — no CUDA device available");
        return;
    }
    cuda_helpers::with_determinism("strict", || {
        let graph = build_repeatable_graph();
        let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");
        let context = repeatable_context();
        let backend = CudaBackend;

        let first = execute_terminal_with_backend(
            &graph,
            &plan,
            &plan.schedule.ordered_nodes,
            &backend,
            &context,
        )
        .expect("first strict cuda run should pass");
        let second = execute_terminal_with_backend(
            &graph,
            &plan,
            &plan.schedule.ordered_nodes,
            &backend,
            &context,
        )
        .expect("second strict cuda run should pass");

        assert_eq!(first, second, "strict mode must be exactly repeatable");
    });
}

#[test]
fn strict_mode_fails_fast_for_nondeterministic_softmax_grouping() {
    if !cuda_helpers::cuda_runtime_available() {
        eprintln!("[SKIP] strict_mode_fails_fast_for_nondeterministic_softmax_grouping — no CUDA device available");
        return;
    }
    cuda_helpers::with_determinism("strict", || {
        let graph = build_multi_softmax_graph();
        let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");
        let context = multi_softmax_context();
        let backend = CudaBackend;

        let err = execute_terminal_with_backend(
            &graph,
            &plan,
            &plan.schedule.ordered_nodes,
            &backend,
            &context,
        )
        .expect_err("strict mode should fail-fast for nondeterministic softmax grouping");

        assert!(
            err.message.contains("fixed reduction topology for softmax"),
            "unexpected error: {}",
            err.message
        );
    });
}

fn build_repeatable_graph() -> Graph {
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
    let (_, b) = graph
        .add_op(block, Op::Parameter("b".to_string()))
        .expect("add parameter should succeed");
    let (_, sum) = graph
        .add_op(block, Op::Add(mm, b))
        .expect("add add should succeed");
    graph
        .add_op(block, Op::Relu(sum))
        .expect("add relu should succeed");

    let (_, logits) = graph
        .add_op(block, Op::Input("logits".to_string()))
        .expect("add logits should succeed");
    let (_, probs) = graph
        .add_op(block, Op::Softmax(logits))
        .expect("add softmax should succeed");
    graph
        .add_op(block, Op::Output(probs))
        .expect("add output should succeed");

    graph.bind_input_shape("x", vec![1, 2]);
    graph.bind_input_shape("logits", vec![2]);
    graph.bind_parameter_shape("w", vec![2, 2]);
    graph.bind_parameter_shape("b", vec![1, 2]);
    graph
}

fn repeatable_context() -> ExecutionContext {
    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor {
            shape: vec![1, 2],
            data: vec![1.0, -2.0],
        },
    );
    context.inputs.insert(
        "logits".to_string(),
        RuntimeValue::Tensor {
            shape: vec![2],
            data: vec![0.5, 2.5],
        },
    );
    context.parameters.insert(
        "w".to_string(),
        RuntimeValue::Tensor {
            shape: vec![2, 2],
            data: vec![1.0, 0.25, -0.5, 0.75],
        },
    );
    context.parameters.insert(
        "b".to_string(),
        RuntimeValue::Tensor {
            shape: vec![1, 2],
            data: vec![0.1, -0.1],
        },
    );
    context
}

fn build_multi_softmax_graph() -> Graph {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, logits_a) = graph
        .add_op(block, Op::Input("logits_a".to_string()))
        .expect("add logits_a should succeed");
    graph
        .add_op(block, Op::Softmax(logits_a))
        .expect("add first softmax should succeed");

    let (_, logits_b) = graph
        .add_op(block, Op::Input("logits_b".to_string()))
        .expect("add logits_b should succeed");
    let (_, probs_b) = graph
        .add_op(block, Op::Softmax(logits_b))
        .expect("add second softmax should succeed");
    graph
        .add_op(block, Op::Output(probs_b))
        .expect("add output should succeed");

    graph.bind_input_shape("logits_a", vec![2]);
    graph.bind_input_shape("logits_b", vec![2]);
    graph
}

fn multi_softmax_context() -> ExecutionContext {
    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "logits_a".to_string(),
        RuntimeValue::Tensor {
            shape: vec![2],
            data: vec![0.2, 0.8],
        },
    );
    context.inputs.insert(
        "logits_b".to_string(),
        RuntimeValue::Tensor {
            shape: vec![2],
            data: vec![1.0, 3.0],
        },
    );
    context
}

