use std::collections::HashSet;

use volta::ir::cuda::{CudaKernel, lower_plan};
use volta::ir::{
    CpuBackend, CudaBackend, ExecutionContext, Graph, Op, RuntimeValue, build_execution_plan,
    execute_terminal_with_backend,
};

#[test]
fn cuda_lowering_dispatches_supported_inference_kernels() {
    let graph = build_supported_inference_graph();
    let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");

    let lowered =
        lower_plan(&plan).expect("matmul/add/relu/softmax subset should lower for CUDA dispatch");
    let kernels = lowered
        .executable_nodes
        .iter()
        .map(|node| node.kernel)
        .collect::<Vec<_>>();

    assert!(kernels.contains(&CudaKernel::MatMul));
    assert!(kernels.contains(&CudaKernel::Add));
    assert!(kernels.contains(&CudaKernel::Relu));
    assert!(kernels.contains(&CudaKernel::Softmax));
}

#[test]
fn cuda_and_cpu_runtime_paths_match_for_supported_dispatch_graph() {
    if !cuda_runtime_available() {
        return;
    }
    let graph = build_supported_inference_graph();
    let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");
    let context = supported_inference_context();

    let cpu = CpuBackend;
    let cuda = CudaBackend;

    let cpu_out =
        execute_terminal_with_backend(&graph, &plan, &plan.schedule.ordered_nodes, &cpu, &context)
            .expect("cpu execution should pass");
    let cuda_out =
        execute_terminal_with_backend(&graph, &plan, &plan.schedule.ordered_nodes, &cuda, &context)
            .expect("cuda execution should pass for supported dispatch subset");

    assert_eq!(cpu_out, cuda_out, "cpu and cuda paths must stay in parity");
}

fn build_supported_inference_graph() -> Graph {
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
        .expect("add bias should succeed");
    let (_, sum) = graph
        .add_op(block, Op::Add(mm, b))
        .expect("add add should succeed");
    let (_, relu) = graph
        .add_op(block, Op::Relu(sum))
        .expect("add relu should succeed");
    let (_, logits) = graph
        .add_op(block, Op::Input("logits".to_string()))
        .expect("add logits input should succeed");
    graph
        .add_op(block, Op::Softmax(logits))
        .expect("add softmax should succeed");
    graph
        .add_op(block, Op::Output(relu))
        .expect("add output should succeed");

    graph.bind_input_shape("x", vec![1, 2]);
    graph.bind_input_shape("logits", vec![2]);
    graph.bind_parameter_shape("w", vec![2, 2]);
    graph.bind_parameter_shape("b", vec![1, 2]);
    graph
}

fn supported_inference_context() -> ExecutionContext {
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
            data: vec![0.1, 0.9],
        },
    );
    context.parameters.insert(
        "w".to_string(),
        RuntimeValue::Tensor {
            shape: vec![2, 2],
            data: vec![0.5, -1.0, 1.5, 2.0],
        },
    );
    context.parameters.insert(
        "b".to_string(),
        RuntimeValue::Tensor {
            shape: vec![1, 2],
            data: vec![0.25, -0.75],
        },
    );
    context
}

fn cuda_runtime_available() -> bool {
    let result = std::panic::catch_unwind(|| volta::ir::cuda::device::CudaDevice::new(0));
    match result {
        Ok(Ok(_)) => true,
        Ok(Err(_)) => false,
        Err(_) => false,
    }
}
