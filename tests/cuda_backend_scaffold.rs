use std::collections::HashSet;

use volta::ir::{Backend, CudaBackend, Graph, Op, build_execution_plan};

#[test]
fn cuda_backend_compiles_data_only_execution_plan() {
    let mut graph = Graph::new();
    let block = graph.create_block();
    let (_, x) = graph
        .add_op(block, Op::ConstInt(7))
        .expect("add const should succeed");
    graph
        .add_op(block, Op::Output(x))
        .expect("add output should succeed");

    let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");
    let compiled = CudaBackend
        .compile(&plan)
        .expect("data-only plan should compile in scaffold");

    assert_eq!(compiled.schedule_len, 2);
    assert_eq!(compiled.peak_bytes, plan.allocation.peak_bytes);
}

#[test]
fn cuda_backend_reports_unsupported_kernel_class_errors() {
    let mut graph = Graph::new();
    let block = graph.create_block();
    let (_, a) = graph
        .add_op(block, Op::ConstInt(1))
        .expect("add first const should succeed");
    let (_, b) = graph
        .add_op(block, Op::ConstInt(2))
        .expect("add second const should succeed");
    let (_, product) = graph
        .add_op(block, Op::Mul(a, b))
        .expect("add mul should succeed");
    graph
        .add_op(block, Op::Output(product))
        .expect("add output should succeed");

    let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");
    let err = CudaBackend
        .compile(&plan)
        .expect_err("elementwise kernels should be unsupported in scaffold");

    assert!(
        err.message
            .contains("unsupported CUDA kernel class: Elementwise"),
        "unexpected error: {}",
        err.message
    );
}
