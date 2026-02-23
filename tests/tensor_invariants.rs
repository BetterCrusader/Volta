use std::sync::Arc;

use volta::ir::{
    ExecutionContext, Graph, Op, RuntimeValue, ShapeFact, Tensor, execute_with_context,
    infer_shapes,
};

#[test]
fn test_shape_inference_broadcasting() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    let (_, b) = graph.add_op(block, Op::Input("b".to_string())).unwrap();
    let (_, add) = graph.add_op(block, Op::Add(a, b)).unwrap();

    graph.bind_input_shape("a", vec![2, 3]);
    graph.bind_input_shape("b", vec![1, 3]);

    let shapes = infer_shapes(&graph).unwrap();
    assert_eq!(shapes.get(&add), Some(&ShapeFact::Tensor(vec![2, 3])));
}

#[test]
fn test_shape_inference_broadcasting_scalar() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    let (_, b) = graph.add_op(block, Op::Input("b".to_string())).unwrap();
    let (_, mul) = graph.add_op(block, Op::Mul(a, b)).unwrap();

    graph.bind_input_shape("a", vec![4, 5]);
    graph.bind_input_shape("b", vec![1]);

    let shapes = infer_shapes(&graph).unwrap();
    assert_eq!(shapes.get(&mul), Some(&ShapeFact::Tensor(vec![4, 5])));
}

#[test]
fn test_shape_inference_incompatible_shapes_error() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    let (_, b) = graph.add_op(block, Op::Input("b".to_string())).unwrap();
    graph.add_op(block, Op::Add(a, b)).unwrap();

    graph.bind_input_shape("a", vec![2, 3]);
    graph.bind_input_shape("b", vec![3, 2]);

    let err = infer_shapes(&graph).unwrap_err();
    assert!(err.message.contains("Shape mismatch"));
}

#[test]
fn test_interpreter_broadcasting_execution() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    let (_, b) = graph.add_op(block, Op::Input("b".to_string())).unwrap();
    let (_, add) = graph.add_op(block, Op::Add(a, b)).unwrap();
    graph.add_op(block, Op::Output(add)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.inputs.insert(
        "a".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )),
    );
    // broadcast across rows
    ctx.inputs.insert(
        "b".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![1, 3], vec![10.0, 20.0, 30.0]).unwrap(),
        )),
    );

    let result = execute_with_context(&graph, &ctx).unwrap();
    match result {
        Some(RuntimeValue::Tensor(t)) => {
            assert_eq!(t.shape, vec![2, 3]);
            assert_eq!(t.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        }
        _ => panic!("Expected tensor output"),
    }
}

#[test]
fn test_interpreter_broadcasting_scalar_execution() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    let (_, b) = graph.add_op(block, Op::Input("b".to_string())).unwrap();
    let (_, mul) = graph.add_op(block, Op::Mul(a, b)).unwrap();
    graph.add_op(block, Op::Output(mul)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.inputs.insert(
        "a".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        )),
    );
    ctx.inputs.insert(
        "b".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![1], vec![5.0]).unwrap())),
    );

    let result = execute_with_context(&graph, &ctx).unwrap();
    match result {
        Some(RuntimeValue::Tensor(t)) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.data, vec![5.0, 10.0, 15.0, 20.0]);
        }
        _ => panic!("Expected tensor output"),
    }
}

#[test]
fn test_interpreter_incompatible_shapes_execution_error() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    let (_, b) = graph.add_op(block, Op::Input("b".to_string())).unwrap();
    let (_, add) = graph.add_op(block, Op::Add(a, b)).unwrap();
    graph.add_op(block, Op::Output(add)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.inputs.insert(
        "a".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![2, 3], vec![1.0; 6]).unwrap())),
    );
    ctx.inputs.insert(
        "b".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![3, 2], vec![1.0; 6]).unwrap())),
    );

    let err = execute_with_context(&graph, &ctx).unwrap_err();
    assert!(err.message.contains("Cannot broadcast"));
}

#[test]
fn test_interpreter_nan_propagation() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    let (_, b) = graph.add_op(block, Op::Input("b".to_string())).unwrap();
    // 0.0 / 0.0 -> division by zero err or NaN?
    // Wait, in volta interpreter, div_values specifically returns Err("Division by zero") for b == 0.0
    // If the user inputs a NaN explicitly, will it propagate?
    let (_, add) = graph.add_op(block, Op::Add(a, b)).unwrap();
    graph.add_op(block, Op::Output(add)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.inputs.insert(
        "a".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![2], vec![f32::NAN, 1.0]).unwrap())),
    );
    ctx.inputs.insert(
        "b".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![2], vec![2.0, 3.0]).unwrap())),
    );

    let result = execute_with_context(&graph, &ctx).unwrap();
    match result {
        Some(RuntimeValue::Tensor(t)) => {
            assert!(t.data[0].is_nan());
            assert_eq!(t.data[1], 4.0);
        }
        _ => panic!("Expected tensor output"),
    }
}

#[test]
fn test_interpreter_gather_out_of_bounds() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    let (_, gather) = graph
        .add_op(
            block,
            Op::Gather {
                input: a,
                indices: vec![0, 5],
                axis: 0,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(gather)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.inputs.insert(
        "a".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![2, 3], vec![1.0; 6]).unwrap())),
    );

    let err = execute_with_context(&graph, &ctx).unwrap_err();
    assert!(err.message.contains("out of bounds"));
}

#[test]
fn test_shape_inference_reducesum() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    let (_, reduce_all) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: a,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();
    let (_, reduce_ax0) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: a,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();

    graph.bind_input_shape("a", vec![2, 3, 5]);

    let shapes = infer_shapes(&graph).unwrap();

    // Reduces to scalar ([1]) if no axis is provided
    assert_eq!(shapes.get(&reduce_all), Some(&ShapeFact::Tensor(vec![1])));
    // Drops the reduced axis
    assert_eq!(
        shapes.get(&reduce_ax0),
        Some(&ShapeFact::Tensor(vec![3, 5]))
    );
}

#[test]
fn test_shape_inference_reducesum_out_of_bounds_axis() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    graph
        .add_op(
            block,
            Op::ReduceSum {
                input: a,
                axis: Some(3),
                keepdims: false,
            },
        )
        .unwrap(); // shape is rank 3 so ax3 is out of bounds

    graph.bind_input_shape("a", vec![2, 3, 5]);

    let err = infer_shapes(&graph).unwrap_err();
    assert!(err.message.contains("out of bounds"));
}

#[test]
fn test_shape_inference_reducemax_and_reducemean() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    let (_, reduce_max) = graph
        .add_op(
            block,
            Op::ReduceMax {
                input: a,
                axis: Some(1),
                keepdims: false,
            },
        )
        .unwrap();
    let (_, reduce_mean) = graph
        .add_op(
            block,
            Op::ReduceMean {
                input: a,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();

    graph.bind_input_shape("a", vec![2, 3, 5]);

    let shapes = infer_shapes(&graph).unwrap();
    assert_eq!(
        shapes.get(&reduce_max),
        Some(&ShapeFact::Tensor(vec![2, 5]))
    );
    assert_eq!(
        shapes.get(&reduce_mean),
        Some(&ShapeFact::Tensor(vec![3, 5]))
    );
}

#[test]
fn test_interpreter_reducemax_and_reducemean_execution() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, a) = graph.add_op(block, Op::Input("a".to_string())).unwrap();
    let (_, reduce_max) = graph
        .add_op(
            block,
            Op::ReduceMax {
                input: a,
                axis: Some(1),
                keepdims: false,
            },
        )
        .unwrap();
    let (_, reduce_mean) = graph
        .add_op(
            block,
            Op::ReduceMean {
                input: a,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    let (_, out_sum) = graph
        .add_op(block, Op::Add(reduce_max, reduce_mean))
        .unwrap();
    graph.add_op(block, Op::Output(out_sum)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.inputs.insert(
        "a".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![2, 2], vec![1.0, 4.0, 3.0, 2.0]).unwrap(),
        )),
    );

    let result = execute_with_context(&graph, &ctx).unwrap();
    match result {
        Some(RuntimeValue::Tensor(t)) => {
            // reduce_max axis=1 => [4, 3]
            // reduce_mean axis=0 => [2, 3]
            // sum => [6, 6]
            assert_eq!(t.shape, vec![2]);
            assert_eq!(t.data, vec![6.0, 6.0]);
        }
        _ => panic!("Expected tensor output"),
    }
}
