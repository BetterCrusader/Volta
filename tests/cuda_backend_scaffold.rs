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
    let (_, input) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![3, 3],
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            },
        )
        .expect("add conv input should succeed");
    let (_, kernel) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![2, 2],
                data: vec![1.0, 0.0, 0.0, -1.0],
            },
        )
        .expect("add conv kernel should succeed");
    let (_, conv) = graph
        .add_op(block, Op::Conv2D(input, kernel))
        .expect("add conv should succeed");
    graph
        .add_op(block, Op::Output(conv))
        .expect("add output should succeed");

    let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");
    let err = CudaBackend
        .compile(&plan)
        .expect_err("conv kernels should be unsupported in scaffold");

    assert!(
        err.message
            .contains("unsupported CUDA kernel class: Conv2D"),
        "unexpected error: {}",
        err.message
    );
}
