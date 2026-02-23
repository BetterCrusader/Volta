use std::sync::Arc;
use std::thread;

use volta::ir::{
    ExecutionContext, Graph, Op, RuntimeValue, Tensor, build_schedule, execute_with_context,
    graph_fingerprint, schedule_hash,
};

/// Construct a non-trivial graph (Simulated multi-layer perceptron with Softmax).
fn build_complex_graph() -> Graph {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
    let (_, w1) = graph
        .add_op(block, Op::Parameter("w1".to_string()))
        .unwrap();
    let (_, b1) = graph
        .add_op(block, Op::Parameter("b1".to_string()))
        .unwrap();

    let (_, xw1) = graph.add_op(block, Op::MatMul(x, w1)).unwrap();
    let (_, h1) = graph.add_op(block, Op::Add(xw1, b1)).unwrap();
    let (_, relu1) = graph.add_op(block, Op::Relu(h1)).unwrap();

    let (_, w2) = graph
        .add_op(block, Op::Parameter("w2".to_string()))
        .unwrap();
    let (_, b2) = graph
        .add_op(block, Op::Parameter("b2".to_string()))
        .unwrap();

    let (_, h2) = graph.add_op(block, Op::MatMul(relu1, w2)).unwrap();
    let (_, logits2d) = graph.add_op(block, Op::Add(h2, b2)).unwrap();
    let (_, logits) = graph
        .add_op(
            block,
            Op::Reshape {
                input: logits2d,
                shape: vec![3],
            },
        )
        .unwrap();
    let (_, probs) = graph.add_op(block, Op::Softmax(logits)).unwrap();

    graph.add_op(block, Op::Output(probs)).unwrap();

    graph.bind_input_shape("x", vec![1, 4]);
    graph.bind_parameter_shape("w1", vec![4, 8]);
    graph.bind_parameter_shape("b1", vec![8]);
    graph.bind_parameter_shape("w2", vec![8, 3]);
    graph.bind_parameter_shape("b2", vec![3]);

    graph
}

/// Helper to execute the graph consistently and return the output as a flat Vec.
fn run_graph(graph: &Graph) -> Vec<f32> {
    let mut ctx = ExecutionContext::default();
    ctx.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![1, 4], vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
        )),
    );
    ctx.parameters.insert(
        "w1".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![4, 8], (0..32).map(|x| (x as f32) * 0.01).collect()).unwrap(),
        )),
    );
    ctx.parameters.insert(
        "b1".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![8], vec![0.05; 8]).unwrap())),
    );
    ctx.parameters.insert(
        "w2".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![8, 3], (0..24).map(|x| (x as f32) * -0.01).collect()).unwrap(),
        )),
    );
    ctx.parameters.insert(
        "b2".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![3], vec![0.0; 3]).unwrap())),
    );

    let out = execute_with_context(graph, &ctx).unwrap();

    if let Some(RuntimeValue::Tensor(t)) = out {
        t.data.clone()
    } else {
        panic!("Expected tensor")
    }
}

#[test]
fn test_determinism_across_multiple_threads() {
    let base_graph = build_complex_graph();

    // Get the gold standard output
    let gold_output = run_graph(&base_graph);

    // Spawn 10 threads, each compiling and running the graph 10 times.
    let mut handles = vec![];
    for _ in 0..10 {
        let graph_clone = base_graph.clone();
        let gold_clone = gold_output.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..10 {
                let out = run_graph(&graph_clone);
                // Assert bit-exact binary equality (no epsilon tolerance).
                assert_eq!(
                    out, gold_clone,
                    "Non-deterministic output detected in thread!"
                );
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_schedule_is_identical_over_100_runs() {
    let graph = build_complex_graph();
    let baseline_schedule = build_schedule(&graph).expect("schedule should build");
    let baseline_order = baseline_schedule
        .ordered_nodes
        .iter()
        .map(|id| id.0)
        .collect::<Vec<_>>();
    let baseline_hash = schedule_hash(&baseline_schedule);
    let baseline_fp = graph_fingerprint(&graph);

    for run in 0..100 {
        let g = graph.clone();
        let schedule = build_schedule(&g).expect("schedule should build");
        let order = schedule
            .ordered_nodes
            .iter()
            .map(|id| id.0)
            .collect::<Vec<_>>();
        let hash = schedule_hash(&schedule);
        let fp = graph_fingerprint(&g);

        assert_eq!(
            order, baseline_order,
            "schedule node order drifted at run {run}"
        );
        assert_eq!(hash, baseline_hash, "schedule hash drifted at run {run}");
        assert_eq!(fp, baseline_fp, "graph fingerprint drifted at run {run}");
    }
}

#[test]
fn test_schedule_is_identical_across_threads() {
    let graph = build_complex_graph();
    let baseline = build_schedule(&graph).expect("schedule should build");
    let baseline_order = baseline
        .ordered_nodes
        .iter()
        .map(|id| id.0)
        .collect::<Vec<_>>();

    let mut handles = vec![];
    for _ in 0..8 {
        let graph_clone = graph.clone();
        let expected = baseline_order.clone();
        handles.push(thread::spawn(move || {
            for run in 0..25 {
                let schedule = build_schedule(&graph_clone).expect("schedule should build");
                let order = schedule
                    .ordered_nodes
                    .iter()
                    .map(|id| id.0)
                    .collect::<Vec<_>>();
                assert_eq!(order, expected, "thread schedule drift at run {run}");
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread should complete without panic");
    }
}
