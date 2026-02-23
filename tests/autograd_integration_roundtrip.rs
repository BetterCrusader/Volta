use std::sync::Arc;

use volta::ir::{
    ExecutionContext, Graph, Op, RuntimeValue, Tensor, build_reverse_graph,
    execute_value_with_context, execute_with_context,
};

fn build_regression_graph() -> (
    Graph,
    volta::ir::ValueId,
    volta::ir::ValueId,
    volta::ir::ValueId,
) {
    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
    let (_, w) = graph.add_op(block, Op::Parameter("w".to_string())).unwrap();
    let (_, b) = graph.add_op(block, Op::Parameter("b".to_string())).unwrap();

    let (_, pred_mm) = graph.add_op(block, Op::MatMul(x, w)).unwrap();
    let (_, pred) = graph.add_op(block, Op::Add(pred_mm, b)).unwrap();

    let target = Tensor::new(vec![2, 2], vec![0.5, -1.0, 1.5, -0.5]).unwrap();
    let (_, target_v) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: target.shape,
                data: target.data,
            },
        )
        .unwrap();

    let (_, diff) = graph.add_op(block, Op::Sub(pred, target_v)).unwrap();
    let (_, sq) = graph.add_op(block, Op::Mul(diff, diff)).unwrap();
    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceMean {
                input: sq,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();
    graph.add_op(block, Op::Output(loss)).unwrap();

    graph.bind_input_shape("x", vec![2, 2]);
    graph.bind_parameter_shape("w", vec![2, 2]);
    graph.bind_parameter_shape("b", vec![2, 2]);

    (graph, loss, w, b)
}

fn base_context() -> ExecutionContext {
    let mut ctx = ExecutionContext::default();
    ctx.inputs.insert(
        "__loss_grad".to_string(),
        RuntimeValue::Tensor(Arc::new(Tensor::new(vec![1], vec![1.0]).unwrap())),
    );
    ctx.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![2, 2], vec![1.0, 2.0, -1.0, 0.5]).unwrap(),
        )),
    );
    ctx.parameters.insert(
        "w".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![2, 2], vec![0.2, -0.1, 0.3, 0.4]).unwrap(),
        )),
    );
    ctx.parameters.insert(
        "b".to_string(),
        RuntimeValue::Tensor(Arc::new(
            Tensor::new(vec![2, 2], vec![0.0, 0.1, -0.2, 0.05]).unwrap(),
        )),
    );
    ctx
}

fn scalar_loss(value: RuntimeValue) -> f32 {
    match value {
        RuntimeValue::Tensor(t) => t.data[0],
        other => panic!("Expected tensor scalar loss, got {other:?}"),
    }
}

#[test]
fn autograd_roundtrip_gradients_are_finite_and_shape_correct() {
    let (graph, loss, w, b) = build_regression_graph();
    let backward =
        build_reverse_graph(&graph, loss, &[w, b]).expect("autograd build should succeed");
    let ctx = base_context();

    let grad_w_id = backward
        .gradients
        .get(&w)
        .copied()
        .expect("grad for w must exist");
    let grad_b_id = backward
        .gradients
        .get(&b)
        .copied()
        .expect("grad for b must exist");

    let grad_w = execute_value_with_context(&backward.backward, grad_w_id, &ctx)
        .expect("grad w execution should succeed");
    let grad_b = execute_value_with_context(&backward.backward, grad_b_id, &ctx)
        .expect("grad b execution should succeed");

    let RuntimeValue::Tensor(grad_w) = grad_w else {
        panic!("grad w must be tensor");
    };
    let RuntimeValue::Tensor(grad_b) = grad_b else {
        panic!("grad b must be tensor");
    };

    assert_eq!(grad_w.shape, vec![2, 2]);
    assert_eq!(grad_b.shape, vec![2, 2]);
    assert!(grad_w.data.iter().all(|v| v.is_finite()));
    assert!(grad_b.data.iter().all(|v| v.is_finite()));
}

#[test]
fn autograd_roundtrip_single_sgd_step_reduces_loss() {
    let (graph, loss, w, b) = build_regression_graph();
    let backward =
        build_reverse_graph(&graph, loss, &[w, b]).expect("autograd build should succeed");
    let mut ctx = base_context();

    let loss_before = scalar_loss(
        execute_with_context(&graph, &ctx)
            .expect("forward execution should succeed")
            .expect("graph should return value"),
    );

    let grad_w_id = backward
        .gradients
        .get(&w)
        .copied()
        .expect("grad for w must exist");
    let grad_b_id = backward
        .gradients
        .get(&b)
        .copied()
        .expect("grad for b must exist");

    let RuntimeValue::Tensor(grad_w) =
        execute_value_with_context(&backward.backward, grad_w_id, &ctx)
            .expect("grad w execution should succeed")
    else {
        panic!("grad w must be tensor");
    };
    let RuntimeValue::Tensor(grad_b) =
        execute_value_with_context(&backward.backward, grad_b_id, &ctx)
            .expect("grad b execution should succeed")
    else {
        panic!("grad b must be tensor");
    };

    let RuntimeValue::Tensor(w_old) = ctx.parameters.get("w").expect("w param").clone() else {
        panic!("w must be tensor");
    };
    let RuntimeValue::Tensor(b_old) = ctx.parameters.get("b").expect("b param").clone() else {
        panic!("b must be tensor");
    };

    let lr = 0.05_f32;
    let w_new = Tensor::new(
        w_old.shape.clone(),
        w_old
            .data
            .iter()
            .zip(grad_w.data.iter())
            .map(|(w0, g)| w0 - lr * g)
            .collect(),
    )
    .unwrap();
    let b_new = Tensor::new(
        b_old.shape.clone(),
        b_old
            .data
            .iter()
            .zip(grad_b.data.iter())
            .map(|(b0, g)| b0 - lr * g)
            .collect(),
    )
    .unwrap();

    ctx.parameters
        .insert("w".to_string(), RuntimeValue::Tensor(Arc::new(w_new)));
    ctx.parameters
        .insert("b".to_string(), RuntimeValue::Tensor(Arc::new(b_new)));

    let loss_after = scalar_loss(
        execute_with_context(&graph, &ctx)
            .expect("forward execution after step should succeed")
            .expect("graph should return value"),
    );

    assert!(
        loss_after < loss_before,
        "expected SGD step to reduce loss: before={loss_before}, after={loss_after}"
    );
}
