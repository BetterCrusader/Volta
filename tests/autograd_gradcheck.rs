use std::sync::Arc;
use volta::ir::shape_inference::infer_shapes;
use volta::ir::{
    ExecutionContext, Graph, Op, RuntimeValue, Tensor, ValueId, build_reverse_graph,
    execute_value_with_context,
};

/// Tolerance parameters for finite-differences gradient checks.
struct GradCheckTolerance {
    eps: f32,
    atol: f32,
    rtol: f32,
}

/// Performs a basic finite-differences gradient check.
/// Only tests a single parameter against a scalar loss output.
fn assert_gradcheck(
    graph: &mut Graph,
    loss_node: ValueId,
    param_node: ValueId,
    param_name: &str,
    mut context: ExecutionContext,
    tol: GradCheckTolerance,
) {
    infer_shapes(graph).expect("Shape inference failed before gradcheck");
    // 1. Analytical Gradient
    let backward = build_reverse_graph(graph, loss_node, &[param_node]).expect("autograd failed");
    let grad_node = *backward
        .gradients
        .get(&param_node)
        .expect("gradient for parameter not found");

    // We must supply __loss_grad manually initialized to 1.0 (assuming scalar loss)
    context.inputs.insert(
        "__loss_grad".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![1], vec![1.0]).unwrap())),
    );

    let analytical_grad_val = execute_value_with_context(&backward.backward, grad_node, &context)
        .expect("analytical path execution failed");
    let RuntimeValue::Tensor(analytical_tensor) = analytical_grad_val else {
        panic!("Analytical gradient must be a tensor");
    };

    // 2. Numerical Gradient via Finite Differences
    let mut param_tensor = {
        let val = context
            .parameters
            .get(param_name)
            .expect("Parameter not found in context");
        let RuntimeValue::Tensor(t) = val else {
            panic!("Parameter must be a tensor");
        };
        Tensor::new(t.shape.clone(), t.data.clone()).unwrap()
    };

    let len = param_tensor.data.len();
    assert_eq!(
        len,
        analytical_tensor.data.len(),
        "Analytical gradient shape must match parameter shape"
    );

    let mut num_grad = vec![0.0; len];

    for (i, slot) in num_grad.iter_mut().enumerate() {
        let orig_val = param_tensor.data[i];

        // f(x + epsilon)
        param_tensor.data[i] = orig_val + tol.eps;
        context.parameters.insert(
            param_name.to_string(),
            RuntimeValue::Tensor(Arc::new(param_tensor.clone())),
        );
        let loss_plus_val = execute_value_with_context(graph, loss_node, &context)
            .expect("Forward pass (plus) failed");
        let RuntimeValue::Tensor(loss_plus_t) = loss_plus_val else {
            panic!("Loss must be a tensor");
        };
        assert_eq!(loss_plus_t.data.len(), 1, "Loss must be scalar");
        let loss_plus = loss_plus_t.data[0];

        // f(x - epsilon)
        param_tensor.data[i] = orig_val - tol.eps;
        context.parameters.insert(
            param_name.to_string(),
            RuntimeValue::Tensor(Arc::new(param_tensor.clone())),
        );
        let loss_minus_val = execute_value_with_context(graph, loss_node, &context)
            .expect("Forward pass (minus) failed");
        let RuntimeValue::Tensor(loss_minus_t) = loss_minus_val else {
            panic!("Loss must be a tensor");
        };
        let loss_minus = loss_minus_t.data[0];

        // Restore
        param_tensor.data[i] = orig_val;

        // Compute finite difference
        *slot = (loss_plus - loss_minus) / (2.0 * tol.eps);
    }

    // 3. Comparison
    for (i, (&a, &n)) in analytical_tensor
        .data
        .iter()
        .zip(num_grad.iter())
        .enumerate()
    {
        let diff = (a - n).abs();
        let max_diff = tol.atol + tol.rtol * n.abs();

        assert!(
            diff <= max_diff,
            "Gradcheck failed at index {}:\nAnalytical: {}\nNumerical:  {}\nDifference: {} > {}",
            i,
            a,
            n,
            diff,
            max_diff
        );
    }
}

#[test]
fn gradcheck_add_sub_mul() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap();
    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();

    // L = (p * x) + p - x
    let (_, mul) = graph.add_op(block, Op::Mul(p, x)).unwrap();
    let (_, add) = graph.add_op(block, Op::Add(mul, p)).unwrap();
    let (_, loss) = graph.add_op(block, Op::Sub(add, x)).unwrap();

    // To scalar
    let (_, scalar_loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: loss,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(scalar_loss)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![3], vec![1.5, -0.5, 2.0]).unwrap(),
        )),
    );
    ctx.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![3], vec![0.5, 1.0, -1.0]).unwrap(),
        )),
    );

    graph.bind_parameter_shape("p", vec![3]);
    graph.bind_input_shape("x", vec![3]);
    assert_gradcheck(
        &mut graph,
        scalar_loss,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-4_f32,
            rtol: 1e-4_f32,
        },
    );
}

#[test]
fn gradcheck_add_broadcast() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap(); // [3,1]
    let (_, q) = graph.add_op(block, Op::Parameter("q".to_string())).unwrap(); // [1,4]
    let (_, add) = graph.add_op(block, Op::Add(p, q)).unwrap();
    let (_, scalar_loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: add,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(scalar_loss)).unwrap();

    graph.bind_parameter_shape("p", vec![3, 1]);
    graph.bind_parameter_shape("q", vec![1, 4]);

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![3, 1], vec![0.1, -0.2, 0.3]).unwrap(),
        )),
    );
    ctx.parameters.insert(
        "q".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![1, 4], vec![1.0, -2.0, 3.0, -4.0]).unwrap(),
        )),
    );

    assert_gradcheck(
        &mut graph,
        scalar_loss,
        p,
        "p",
        ctx.clone(),
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-4_f32,
            rtol: 1e-4_f32,
        },
    );
    assert_gradcheck(
        &mut graph,
        scalar_loss,
        q,
        "q",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-4_f32,
            rtol: 1e-4_f32,
        },
    );
}

#[test]
fn gradcheck_mul_broadcast() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap(); // [2,1]
    let (_, q) = graph.add_op(block, Op::Parameter("q".to_string())).unwrap(); // [1,3]
    let (_, mul) = graph.add_op(block, Op::Mul(p, q)).unwrap();
    let (_, scalar_loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: mul,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(scalar_loss)).unwrap();

    graph.bind_parameter_shape("p", vec![2, 1]);
    graph.bind_parameter_shape("q", vec![1, 3]);

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![2, 1], vec![0.5, -1.5]).unwrap())),
    );
    ctx.parameters.insert(
        "q".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![1, 3], vec![2.0, -1.0, 0.25]).unwrap(),
        )),
    );

    assert_gradcheck(
        &mut graph,
        scalar_loss,
        p,
        "p",
        ctx.clone(),
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-4_f32,
            rtol: 1e-4_f32,
        },
    );
    assert_gradcheck(
        &mut graph,
        scalar_loss,
        q,
        "q",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-4_f32,
            rtol: 1e-4_f32,
        },
    );
}

#[test]
fn gradcheck_reduce_mean() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap();
    let (_, mean) = graph
        .add_op(
            block,
            Op::ReduceMean {
                input: p,
                axis: Some(1),
                keepdims: false,
            },
        )
        .unwrap();
    let (_, scalar_loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: mean,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(scalar_loss)).unwrap();

    graph.bind_parameter_shape("p", vec![2, 3]);

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![2, 3], vec![1.0, -2.0, 3.0, 0.5, 1.5, -1.0]).unwrap(),
        )),
    );

    assert_gradcheck(
        &mut graph,
        scalar_loss,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-4_f32,
            rtol: 1e-4_f32,
        },
    );
}

#[test]
fn gradcheck_relu() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap();
    let (_, relu) = graph.add_op(block, Op::Relu(p)).unwrap();
    let (_, scalar_loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: relu,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(scalar_loss)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        // Avoid values exactly at 0.0 to prevent non-differentiable sharp points failing the finite difference check
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![4], vec![1.5, -0.5, 0.1, -0.1]).unwrap(),
        )),
    );

    graph.bind_parameter_shape("p", vec![4]);
    assert_gradcheck(
        &mut graph,
        scalar_loss,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-4_f32,
            rtol: 1e-4_f32,
        },
    );
}

#[test]
fn gradcheck_matmul() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap(); // 2x3
    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap(); // 3x2

    let (_, mm) = graph.add_op(block, Op::MatMul(p, x)).unwrap();

    let (_, t1) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: mm,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: t1,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(loss)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.5, 0.0]).unwrap(),
        )),
    );
    ctx.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![3, 2], vec![0.5, 1.5, -0.5, 1.0, 0.0, -1.0]).unwrap(),
        )),
    );

    graph.bind_parameter_shape("p", vec![2, 3]);
    graph.bind_input_shape("x", vec![3, 2]);
    assert_gradcheck(
        &mut graph,
        loss,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-4_f32,
            rtol: 1e-4_f32,
        },
    );
}

#[test]
fn gradcheck_div() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap();
    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();

    let (_, div) = graph.add_op(block, Op::Div(p, x)).unwrap();
    let (_, scalar_loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: div,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(scalar_loss)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![3], vec![1.5, -0.5, 2.0]).unwrap(),
        )),
    );
    // Denominator far away from 0 to avoid precision issues in gradient
    ctx.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![3], vec![2.0, 3.0, -1.0]).unwrap(),
        )),
    );

    graph.bind_parameter_shape("p", vec![3]);
    graph.bind_input_shape("x", vec![3]);
    assert_gradcheck(
        &mut graph,
        scalar_loss,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-4_f32,
            rtol: 1e-4_f32,
        },
    );
}

#[test]
fn gradcheck_softmax() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap();
    let (_, sm) = graph.add_op(block, Op::Softmax(p)).unwrap();
    // Reduce completely to scalar
    let (_, scalar_loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: sm,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(scalar_loss)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![3], vec![1.0, 2.0, 0.5]).unwrap())),
    );

    graph.bind_parameter_shape("p", vec![3]);
    // Softmax gradient finite difference check requires relaxed tolerance due to exponentials
    assert_gradcheck(
        &mut graph,
        scalar_loss,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-3_f32,
            rtol: 1e-3_f32,
        },
    );
}

#[test]
fn gradcheck_reshape() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap();
    let (_, r) = graph
        .add_op(
            block,
            Op::Reshape {
                input: p,
                shape: vec![2, 3],
            },
        )
        .unwrap();
    let (_, l) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: r,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    let (_, l2) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: l,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(l2)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![6], vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0]).unwrap(),
        )),
    );

    graph.bind_parameter_shape("p", vec![6]);
    assert_gradcheck(
        &mut graph,
        l2,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-3_f32,
            rtol: 1e-3_f32,
        },
    );
}

#[test]
fn gradcheck_transpose() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap();
    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();

    let (_, t) = graph.add_op(block, Op::Transpose(p)).unwrap();
    let (_, m) = graph.add_op(block, Op::MatMul(x, t)).unwrap();
    let (_, l) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: m,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(l)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )),
    );
    ctx.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![1, 2], vec![2.0, 3.0]).unwrap())),
    );

    graph.bind_parameter_shape("p", vec![3, 2]);
    graph.bind_input_shape("x", vec![1, 2]);
    assert_gradcheck(
        &mut graph,
        l,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-3_f32,
            rtol: 1e-3_f32,
        },
    );
}

#[test]
fn gradcheck_slice() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap();
    let (_, s) = graph
        .add_op(
            block,
            Op::Slice {
                input: p,
                starts: vec![1],
                ends: vec![3],
                axes: vec![0],
            },
        )
        .unwrap();
    let (_, l) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: s,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(l)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        )),
    );

    graph.bind_parameter_shape("p", vec![4]);
    assert_gradcheck(
        &mut graph,
        l,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-4_f32,
            rtol: 1e-4_f32,
        },
    );
}

#[test]
fn gradcheck_gather() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap();
    let (_, g) = graph
        .add_op(
            block,
            Op::Gather {
                input: p,
                indices: vec![0, 2, 0],
                axis: 0,
            },
        )
        .unwrap();
    let (_, l) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: g,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(l)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap())),
    );

    graph.bind_parameter_shape("p", vec![3]);
    assert_gradcheck(
        &mut graph,
        l,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-4_f32,
            rtol: 1e-4_f32,
        },
    );
}

#[test]
fn gradcheck_sigmoid() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap();
    let (_, sig) = graph.add_op(block, Op::Sigmoid(p)).unwrap();
    let (_, scalar_loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: sig,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(scalar_loss)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![4], vec![-2.0, -0.5, 0.5, 2.0]).unwrap(),
        )),
    );

    graph.bind_parameter_shape("p", vec![4]);
    assert_gradcheck(
        &mut graph,
        scalar_loss,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-3_f32,
            rtol: 1e-3_f32,
        },
    );
}

#[test]
fn gradcheck_gelu() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap();
    let (_, g) = graph.add_op(block, Op::Gelu(p)).unwrap();
    let (_, scalar_loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: g,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(scalar_loss)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![4], vec![-1.0, -0.5, 0.5, 1.0]).unwrap(),
        )),
    );

    graph.bind_parameter_shape("p", vec![4]);
    assert_gradcheck(
        &mut graph,
        scalar_loss,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-3_f32,
            rtol: 1e-3_f32,
        },
    );
}

#[test]
fn gradcheck_gemm() {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, p) = graph.add_op(block, Op::Parameter("p".to_string())).unwrap(); // 2x3
    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap(); // 3x2

    let (_, gemm) = graph
        .add_op(
            block,
            Op::Gemm {
                lhs: p,
                rhs: x,
                bias: None,
                alpha: 1.0,
                beta: 0.0,
            },
        )
        .unwrap();

    let (_, t1) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: gemm,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: t1,
                axis: Some(0),
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(loss)).unwrap();

    let mut ctx = ExecutionContext::default();
    ctx.parameters.insert(
        "p".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.5, 0.0]).unwrap(),
        )),
    );
    ctx.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![3, 2], vec![0.5, 1.5, -0.5, 1.0, 0.0, -1.0]).unwrap(),
        )),
    );

    graph.bind_parameter_shape("p", vec![2, 3]);
    graph.bind_input_shape("x", vec![3, 2]);
    assert_gradcheck(
        &mut graph,
        loss,
        p,
        "p",
        ctx,
        GradCheckTolerance {
            eps: 1e-3_f32,
            atol: 1e-3_f32,
            rtol: 1e-3_f32,
        },
    );
}
