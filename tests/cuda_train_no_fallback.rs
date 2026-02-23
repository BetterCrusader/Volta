#[path = "common/cuda.rs"]
mod cuda_helpers;

use std::collections::HashMap;

use volta::ir::{
    CudaBackend, Graph, Op, OptimizerConfig, Tensor, TrainConfig, TrainSample,
    train_graph_with_backend,
};

#[test]
fn cuda_train_fails_explicitly_for_unsupported_graph_without_cpu_fallback() {
    if !cuda_helpers::cuda_runtime_available() {
        eprintln!(
            "[SKIP] cuda_train_fails_explicitly_for_unsupported_graph_without_cpu_fallback — no CUDA device available"
        );
        return;
    }

    cuda_helpers::with_determinism("strict", || {
        let (graph, loss) = build_conv_loss_graph();
        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, -1.0]).expect("valid kernel tensor"),
        );

        let dataset = vec![sample_input()];
        let cfg = TrainConfig {
            epochs: 1,
            optimizer: OptimizerConfig::Sgd { lr: 0.01 },
        };

        let cuda = CudaBackend;
        let err = train_graph_with_backend(&graph, loss, params, &dataset, &cfg, &cuda)
            .expect_err("unsupported CUDA train graph must fail explicitly");

        assert!(
            err.message
                .contains("unsupported CUDA kernel class: Conv2D")
                || err.message.contains("Failed to build reverse graph"),
            "unexpected error: {}",
            err.message
        );
        assert!(
            !err.message.to_ascii_lowercase().contains("fallback"),
            "error must not indicate silent fallback: {}",
            err.message
        );
    });
}

fn build_conv_loss_graph() -> (Graph, volta::ir::ValueId) {
    let mut graph = Graph::new();
    let block = graph.create_block();
    let (_, x) = graph
        .add_op(block, Op::Input("x".to_string()))
        .expect("add input");
    let (_, w) = graph
        .add_op(block, Op::Parameter("w".to_string()))
        .expect("add parameter");
    let (_, conv) = graph.add_op(block, Op::Conv2D(x, w)).expect("add conv");
    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: conv,
                axis: None,
                keepdims: false,
            },
        )
        .expect("add reduce sum");
    graph.add_op(block, Op::Output(loss)).expect("add output");
    (graph, loss)
}

fn sample_input() -> TrainSample {
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::new(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .expect("valid input tensor"),
    );
    TrainSample { inputs }
}
