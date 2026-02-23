use std::collections::HashSet;

use volta::ir::{
    CpuBackend, CudaBackend, ExecutionContext, Graph, Op, RuntimeValue, build_execution_plan,
    execute_terminal_with_backend,
};

fn cuda_available() -> bool {
    match std::panic::catch_unwind(|| volta::ir::cuda::device::CudaDevice::new(0)) {
        Ok(Ok(_)) => true,
        Ok(Err(_)) => false,
        Err(_) => false,
    }
}

#[test]
fn cuda_sigmoid_and_gelu_match_cpu() {
    if !cuda_available() {
        return;
    }

    let mut graph = Graph::new();
    let block = graph.create_block();
    let (_, input) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![4],
                data: vec![-2.0, -0.5, 0.5, 2.0],
            },
        )
        .expect("add input");
    let (_, sig) = graph
        .add_op(block, Op::Sigmoid(input))
        .expect("add sigmoid");
    let (_, gelu) = graph.add_op(block, Op::Gelu(sig)).expect("add gelu");
    graph.add_op(block, Op::Output(gelu)).expect("add output");

    let plan = build_execution_plan(&graph, &HashSet::new()).expect("build plan");
    let ctx = ExecutionContext::default();

    let cpu = execute_terminal_with_backend(
        &graph,
        &plan,
        &plan.schedule.ordered_nodes,
        &CpuBackend,
        &ctx,
    )
    .expect("cpu run")
    .expect("cpu output");
    let cuda = execute_terminal_with_backend(
        &graph,
        &plan,
        &plan.schedule.ordered_nodes,
        &CudaBackend,
        &ctx,
    )
    .expect("cuda run")
    .expect("cuda output");

    let (RuntimeValue::Tensor(cpu_t), RuntimeValue::Tensor(cuda_t)) = (cpu, cuda) else {
        panic!("expected tensor outputs");
    };
    assert_eq!(cpu_t.shape, cuda_t.shape);
    for (a, b) in cpu_t.data.iter().zip(cuda_t.data.iter()) {
        assert!((a - b).abs() <= 1e-6, "mismatch: cpu={a}, cuda={b}");
    }
}

#[test]
fn cuda_gemm_matches_cpu() {
    if !cuda_available() {
        return;
    }

    let mut graph = Graph::new();
    let block = graph.create_block();
    let (_, a) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![2, 2],
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
        )
        .expect("add lhs");
    let (_, b) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![2, 2],
                data: vec![5.0, 6.0, 7.0, 8.0],
            },
        )
        .expect("add rhs");
    let (_, bias) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![2],
                data: vec![0.5, -1.5],
            },
        )
        .expect("add bias");
    let (_, out) = graph
        .add_op(
            block,
            Op::Gemm {
                lhs: a,
                rhs: b,
                bias: Some(bias),
                alpha: 0.5,
                beta: 1.0,
            },
        )
        .expect("add gemm");
    graph.add_op(block, Op::Output(out)).expect("add output");

    let plan = build_execution_plan(&graph, &HashSet::new()).expect("build plan");
    let ctx = ExecutionContext::default();

    let cpu = execute_terminal_with_backend(
        &graph,
        &plan,
        &plan.schedule.ordered_nodes,
        &CpuBackend,
        &ctx,
    )
    .expect("cpu run")
    .expect("cpu output");
    let cuda = execute_terminal_with_backend(
        &graph,
        &plan,
        &plan.schedule.ordered_nodes,
        &CudaBackend,
        &ctx,
    )
    .expect("cuda run")
    .expect("cuda output");

    let (RuntimeValue::Tensor(cpu_t), RuntimeValue::Tensor(cuda_t)) = (cpu, cuda) else {
        panic!("expected tensor outputs");
    };
    assert_eq!(cpu_t.shape, cuda_t.shape);
    for (a, b) in cpu_t.data.iter().zip(cuda_t.data.iter()) {
        assert!((a - b).abs() <= 1e-6, "mismatch: cpu={a}, cuda={b}");
    }
}
