use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;

use serde_json::Value;
use volta::ir::{
    ExecutionContext, Graph, Op, OptimizerConfig, OptimizerState, RuntimeValue, Tensor,
    TrainConfig, TrainSample, TransformerConfig, add_transformer_encoder_block, apply_gradients,
    build_reverse_graph, execute_value_with_context, train_graph,
};

fn require_pytorch() -> bool {
    std::env::var("VOLTA_REQUIRE_PYTORCH_PARITY")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn python_candidates() -> Vec<(String, Vec<String>)> {
    vec![
        ("python".to_string(), Vec::new()),
        ("py".to_string(), vec!["-3".to_string()]),
    ]
}

fn run_pytorch_case(case: &str) -> Option<Value> {
    let script = "examples/pytorch_parity.py";
    let require = require_pytorch();

    for (program, args) in python_candidates() {
        let probe = Command::new(&program).args(&args).arg("--version").output();
        if probe.is_err() {
            continue;
        }

        let torch_probe = Command::new(&program)
            .args(&args)
            .arg("-c")
            .arg("import torch")
            .output();
        match torch_probe {
            Ok(output) if output.status.success() => {
                let result = Command::new(&program)
                    .args(&args)
                    .arg(script)
                    .arg(case)
                    .output()
                    .expect("failed to run pytorch parity script");
                if !result.status.success() {
                    panic!(
                        "pytorch parity script failed for case '{case}': {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                }
                return Some(
                    serde_json::from_slice(&result.stdout)
                        .expect("pytorch parity script must emit valid json"),
                );
            }
            Ok(_) => continue,
            Err(_) => continue,
        }
    }

    if require {
        panic!("PyTorch parity required, but python/torch was not available");
    }

    eprintln!("skipping pytorch parity case '{case}' because python/torch is unavailable");
    None
}

fn tensor(shape: Vec<usize>, data: Vec<f32>) -> RuntimeValue {
    RuntimeValue::Tensor(Arc::new(Tensor::new(shape, data).expect("valid tensor")))
}

fn tensor_data(value: RuntimeValue) -> Vec<f32> {
    match value {
        RuntimeValue::Tensor(tensor) => tensor.make_contiguous().expect("contiguous").data.to_vec(),
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn json_f32_array(value: &Value, key: &str) -> Vec<f32> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("missing array key '{key}'"))
        .iter()
        .map(|item| item.as_f64().expect("f64 json number") as f32)
        .collect()
}

fn scalar_f32(value: &Value, key: &str) -> f32 {
    value[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing scalar key '{key}'")) as f32
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch {} != {}",
        actual.len(),
        expected.len()
    );
    for (index, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (lhs - rhs).abs();
        assert!(
            delta <= tol,
            "{label}[{index}] mismatch: actual={lhs}, expected={rhs}, delta={delta}, tol={tol}"
        );
    }
}

#[test]
fn pytorch_parity_mlp_forward_and_gradients() {
    let Some(expected) = run_pytorch_case("mlp") else {
        return;
    };

    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
    let (_, w1) = graph
        .add_op(block, Op::Parameter("w1".to_string()))
        .unwrap();
    let (_, b1) = graph
        .add_op(block, Op::Parameter("b1".to_string()))
        .unwrap();
    let (_, w2) = graph
        .add_op(block, Op::Parameter("w2".to_string()))
        .unwrap();
    let (_, b2) = graph
        .add_op(block, Op::Parameter("b2".to_string()))
        .unwrap();

    let (_, hidden) = graph
        .add_op(
            block,
            Op::Gemm {
                lhs: x,
                rhs: w1,
                bias: Some(b1),
                alpha: 1.0,
                beta: 1.0,
            },
        )
        .unwrap();
    let (_, hidden_relu) = graph.add_op(block, Op::Relu(hidden)).unwrap();
    let (_, out) = graph
        .add_op(
            block,
            Op::Gemm {
                lhs: hidden_relu,
                rhs: w2,
                bias: Some(b2),
                alpha: 1.0,
                beta: 1.0,
            },
        )
        .unwrap();
    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: out,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();

    graph.bind_input_shape("x", vec![2, 3]);
    graph.bind_parameter_shape("w1", vec![3, 4]);
    graph.bind_parameter_shape("b1", vec![4]);
    graph.bind_parameter_shape("w2", vec![4, 2]);
    graph.bind_parameter_shape("b2", vec![2]);

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        tensor(vec![2, 3], vec![0.2, -0.1, 0.3, 0.7, 0.5, -0.4]),
    );
    context.parameters.insert(
        "w1".to_string(),
        tensor(
            vec![3, 4],
            vec![
                0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.2, -0.3,
            ],
        ),
    );
    context.parameters.insert(
        "b1".to_string(),
        tensor(vec![4], vec![0.05, -0.1, 0.15, 0.2]),
    );
    context.parameters.insert(
        "w2".to_string(),
        tensor(vec![4, 2], vec![0.2, -0.4, 0.1, 0.3, -0.5, 0.7, 0.6, -0.2]),
    );
    context
        .parameters
        .insert("b2".to_string(), tensor(vec![2], vec![0.25, -0.35]));
    context
        .inputs
        .insert("__loss_grad".to_string(), tensor(vec![1], vec![1.0]));

    let actual_out = tensor_data(execute_value_with_context(&graph, out, &context).unwrap());
    assert_close(
        "mlp.output",
        &actual_out,
        &json_f32_array(&expected, "output"),
        1e-5,
    );

    let grad_graph = build_reverse_graph(&graph, loss, &[x, w1, b1, w2, b2]).unwrap();
    for (name, value_id) in [("x", x), ("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)] {
        let gradient = *grad_graph.gradients.get(&value_id).unwrap();
        let actual = tensor_data(
            execute_value_with_context(&grad_graph.backward, gradient, &context).unwrap(),
        );
        assert_close(
            &format!("mlp.grad.{name}"),
            &actual,
            &json_f32_array(&expected["gradients"], name),
            1e-5,
        );
    }
}

#[test]
fn pytorch_parity_conv2d_forward_and_gradients() {
    let Some(expected) = run_pytorch_case("conv2d") else {
        return;
    };

    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
    let (_, w) = graph.add_op(block, Op::Parameter("w".to_string())).unwrap();
    let (_, out) = graph.add_op(block, Op::Conv2D(x, w)).unwrap();
    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: out,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();

    graph.bind_input_shape("x", vec![3, 3]);
    graph.bind_parameter_shape("w", vec![2, 2]);

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        tensor(
            vec![3, 3],
            vec![0.2, -0.1, 0.3, 0.7, 0.5, -0.4, 0.6, -0.2, 0.9],
        ),
    );
    context.parameters.insert(
        "w".to_string(),
        tensor(vec![2, 2], vec![0.5, -0.25, 0.75, 0.1]),
    );
    context
        .inputs
        .insert("__loss_grad".to_string(), tensor(vec![1], vec![1.0]));

    let actual_out = tensor_data(execute_value_with_context(&graph, out, &context).unwrap());
    assert_close(
        "conv2d.output",
        &actual_out,
        &json_f32_array(&expected, "output"),
        1e-5,
    );

    let grad_graph = build_reverse_graph(&graph, loss, &[x, w]).unwrap();
    for (name, value_id) in [("x", x), ("w", w)] {
        let gradient = *grad_graph.gradients.get(&value_id).unwrap();
        let actual = tensor_data(
            execute_value_with_context(&grad_graph.backward, gradient, &context).unwrap(),
        );
        assert_close(
            &format!("conv2d.grad.{name}"),
            &actual,
            &json_f32_array(&expected["gradients"], name),
            1e-5,
        );
    }
}

#[test]
fn pytorch_parity_layernorm_forward_and_gradients() {
    let Some(expected) = run_pytorch_case("layernorm") else {
        return;
    };

    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
    let (_, w) = graph
        .add_op(block, Op::Parameter("ln_w".to_string()))
        .unwrap();
    let (_, b) = graph
        .add_op(block, Op::Parameter("ln_b".to_string()))
        .unwrap();
    let (_, out) = graph
        .add_op(
            block,
            Op::LayerNorm {
                input: x,
                weight: w,
                bias: b,
                epsilon: 1e-5,
            },
        )
        .unwrap();
    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: out,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();

    graph.bind_input_shape("x", vec![2, 5]);
    graph.bind_parameter_shape("ln_w", vec![5]);
    graph.bind_parameter_shape("ln_b", vec![5]);

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        tensor(
            vec![2, 5],
            vec![0.2, -0.1, 0.3, 0.4, -0.2, 0.7, 0.5, -0.4, 0.1, 0.9],
        ),
    );
    context.parameters.insert(
        "ln_w".to_string(),
        tensor(vec![5], vec![1.1, 0.9, -0.7, 0.5, 0.3]),
    );
    context.parameters.insert(
        "ln_b".to_string(),
        tensor(vec![5], vec![0.05, -0.15, 0.2, -0.1, 0.25]),
    );
    context
        .inputs
        .insert("__loss_grad".to_string(), tensor(vec![1], vec![1.0]));

    let actual_out = tensor_data(execute_value_with_context(&graph, out, &context).unwrap());
    assert_close(
        "layernorm.output",
        &actual_out,
        &json_f32_array(&expected, "output"),
        1e-5,
    );

    let grad_graph = build_reverse_graph(&graph, loss, &[x, w, b]).unwrap();
    for (name, value_id) in [("x", x), ("w", w), ("b", b)] {
        let gradient = *grad_graph.gradients.get(&value_id).unwrap();
        let actual = tensor_data(
            execute_value_with_context(&grad_graph.backward, gradient, &context).unwrap(),
        );
        assert_close(
            &format!("layernorm.grad.{name}"),
            &actual,
            &json_f32_array(&expected["gradients"], name),
            1e-5,
        );
    }
}

#[test]
fn pytorch_parity_batchnorm_forward_and_gradients() {
    let Some(expected) = run_pytorch_case("batchnorm") else {
        return;
    };

    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
    let (_, w) = graph
        .add_op(block, Op::Parameter("bn_w".to_string()))
        .unwrap();
    let (_, b) = graph
        .add_op(block, Op::Parameter("bn_b".to_string()))
        .unwrap();
    let (_, mean) = graph
        .add_op(block, Op::Input("bn_mean".to_string()))
        .unwrap();
    let (_, var) = graph
        .add_op(block, Op::Input("bn_var".to_string()))
        .unwrap();
    let (_, out) = graph
        .add_op(
            block,
            Op::BatchNorm {
                input: x,
                weight: w,
                bias: b,
                mean,
                var,
            },
        )
        .unwrap();
    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: out,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();

    graph.bind_input_shape("x", vec![1, 2, 2, 2]);
    graph.bind_input_shape("bn_mean", vec![2]);
    graph.bind_input_shape("bn_var", vec![2]);
    graph.bind_parameter_shape("bn_w", vec![2]);
    graph.bind_parameter_shape("bn_b", vec![2]);

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        tensor(
            vec![1, 2, 2, 2],
            vec![0.2, -0.1, 0.3, 0.4, 0.7, 0.5, -0.4, 0.1],
        ),
    );
    context
        .parameters
        .insert("bn_w".to_string(), tensor(vec![2], vec![1.2, -0.8]));
    context
        .parameters
        .insert("bn_b".to_string(), tensor(vec![2], vec![0.1, -0.2]));
    context
        .inputs
        .insert("bn_mean".to_string(), tensor(vec![2], vec![0.15, 0.05]));
    context
        .inputs
        .insert("bn_var".to_string(), tensor(vec![2], vec![0.25, 0.5]));
    context
        .inputs
        .insert("__loss_grad".to_string(), tensor(vec![1], vec![1.0]));

    let actual_out = tensor_data(execute_value_with_context(&graph, out, &context).unwrap());
    assert_close(
        "batchnorm.output",
        &actual_out,
        &json_f32_array(&expected, "output"),
        1e-5,
    );

    let grad_graph = build_reverse_graph(&graph, loss, &[x, w, b]).unwrap();
    for (name, value_id) in [("x", x), ("w", w), ("b", b)] {
        let gradient = *grad_graph.gradients.get(&value_id).unwrap();
        let actual = tensor_data(
            execute_value_with_context(&grad_graph.backward, gradient, &context).unwrap(),
        );
        assert_close(
            &format!("batchnorm.grad.{name}"),
            &actual,
            &json_f32_array(&expected["gradients"], name),
            1e-5,
        );
    }
}

#[test]
fn pytorch_parity_mha_forward_and_gradients() {
    let Some(expected) = run_pytorch_case("mha") else {
        return;
    };

    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, q) = graph.add_op(block, Op::Input("q".to_string())).unwrap();
    let (_, k) = graph.add_op(block, Op::Input("k".to_string())).unwrap();
    let (_, v) = graph.add_op(block, Op::Input("v".to_string())).unwrap();
    let (_, w_q) = graph
        .add_op(block, Op::Parameter("w_q".to_string()))
        .unwrap();
    let (_, w_k) = graph
        .add_op(block, Op::Parameter("w_k".to_string()))
        .unwrap();
    let (_, w_v) = graph
        .add_op(block, Op::Parameter("w_v".to_string()))
        .unwrap();
    let (_, w_o) = graph
        .add_op(block, Op::Parameter("w_o".to_string()))
        .unwrap();
    let (_, b_q) = graph
        .add_op(block, Op::Parameter("b_q".to_string()))
        .unwrap();
    let (_, b_k) = graph
        .add_op(block, Op::Parameter("b_k".to_string()))
        .unwrap();
    let (_, b_v) = graph
        .add_op(block, Op::Parameter("b_v".to_string()))
        .unwrap();
    let (_, b_o) = graph
        .add_op(block, Op::Parameter("b_o".to_string()))
        .unwrap();

    let mut out_value = None;
    for output_idx in 0..6 {
        let (_, value) = graph
            .add_op(
                block,
                Op::MultiHeadAttention {
                    q_input: q,
                    k_input: k,
                    v_input: v,
                    w_q,
                    w_k,
                    w_v,
                    w_o,
                    bias_q: b_q,
                    bias_k: b_k,
                    bias_v: b_v,
                    bias_o: b_o,
                    num_heads: 2,
                    causal: false,
                    output_idx,
                },
            )
            .unwrap();
        if output_idx == 0 {
            out_value = Some(value);
        }
    }
    let out = out_value.unwrap();
    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: out,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();

    graph.bind_input_shape("q", vec![1, 2, 4]);
    graph.bind_input_shape("k", vec![1, 2, 4]);
    graph.bind_input_shape("v", vec![1, 2, 4]);
    graph.bind_parameter_shape("w_q", vec![4, 4]);
    graph.bind_parameter_shape("w_k", vec![4, 4]);
    graph.bind_parameter_shape("w_v", vec![4, 4]);
    graph.bind_parameter_shape("w_o", vec![4, 4]);
    graph.bind_parameter_shape("b_q", vec![4]);
    graph.bind_parameter_shape("b_k", vec![4]);
    graph.bind_parameter_shape("b_v", vec![4]);
    graph.bind_parameter_shape("b_o", vec![4]);

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "q".to_string(),
        tensor(
            vec![1, 2, 4],
            vec![0.2, -0.1, 0.3, 0.4, 0.7, 0.5, -0.4, 0.1],
        ),
    );
    context.inputs.insert(
        "k".to_string(),
        tensor(
            vec![1, 2, 4],
            vec![0.6, -0.2, 0.5, -0.3, 0.1, 0.8, -0.7, 0.2],
        ),
    );
    context.inputs.insert(
        "v".to_string(),
        tensor(
            vec![1, 2, 4],
            vec![0.3, 0.4, -0.5, 0.2, 0.9, -0.6, 0.1, 0.7],
        ),
    );
    context.parameters.insert(
        "w_q".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.2, -0.1, 0.3, 0.4, -0.5, 0.6, 0.1, -0.2, 0.7, 0.2, -0.3, 0.5, 0.4, -0.6, 0.8, 0.1,
            ],
        ),
    );
    context.parameters.insert(
        "w_k".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.1, 0.2, -0.4, 0.3, 0.5, -0.7, 0.6, 0.2, -0.3, 0.8, 0.4, -0.1, 0.2, 0.1, 0.5, -0.6,
            ],
        ),
    );
    context.parameters.insert(
        "w_v".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.3, -0.2, 0.1, 0.7, 0.6, 0.4, -0.5, 0.2, 0.2, -0.8, 0.9, 0.1, -0.4, 0.3, 0.2, 0.5,
            ],
        ),
    );
    context.parameters.insert(
        "w_o".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.4, -0.3, 0.2, 0.1, 0.5, 0.6, -0.7, 0.2, -0.1, 0.8, 0.3, -0.4, 0.7, -0.2, 0.5, 0.6,
            ],
        ),
    );
    context.parameters.insert(
        "b_q".to_string(),
        tensor(vec![4], vec![0.05, -0.1, 0.15, -0.2]),
    );
    context.parameters.insert(
        "b_k".to_string(),
        tensor(vec![4], vec![-0.05, 0.2, -0.15, 0.1]),
    );
    context.parameters.insert(
        "b_v".to_string(),
        tensor(vec![4], vec![0.1, 0.05, -0.2, 0.25]),
    );
    context.parameters.insert(
        "b_o".to_string(),
        tensor(vec![4], vec![-0.1, 0.15, 0.05, -0.05]),
    );
    context
        .inputs
        .insert("__loss_grad".to_string(), tensor(vec![1], vec![1.0]));

    let actual_out = tensor_data(execute_value_with_context(&graph, out, &context).unwrap());
    assert_close(
        "mha.output",
        &actual_out,
        &json_f32_array(&expected, "output"),
        1e-4,
    );

    let grad_graph = build_reverse_graph(&graph, loss, &[q, v, b_q, b_o, w_q, w_o]).unwrap();
    for (name, value_id) in [
        ("q", q),
        ("v", v),
        ("b_q", b_q),
        ("b_o", b_o),
        ("w_q", w_q),
        ("w_o", w_o),
    ] {
        let gradient = *grad_graph.gradients.get(&value_id).unwrap();
        let actual = tensor_data(
            execute_value_with_context(&grad_graph.backward, gradient, &context).unwrap(),
        );
        assert_close(
            &format!("mha.grad.{name}"),
            &actual,
            &json_f32_array(&expected["gradients"], name),
            1e-4,
        );
    }
}

#[test]
fn pytorch_parity_mha_self_attention_forward_and_gradients() {
    let Some(expected) = run_pytorch_case("mha_self") else {
        return;
    };

    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
    let (_, w_q) = graph
        .add_op(block, Op::Parameter("w_q".to_string()))
        .unwrap();
    let (_, w_k) = graph
        .add_op(block, Op::Parameter("w_k".to_string()))
        .unwrap();
    let (_, w_v) = graph
        .add_op(block, Op::Parameter("w_v".to_string()))
        .unwrap();
    let (_, w_o) = graph
        .add_op(block, Op::Parameter("w_o".to_string()))
        .unwrap();
    let (_, b_q) = graph
        .add_op(block, Op::Parameter("b_q".to_string()))
        .unwrap();
    let (_, b_k) = graph
        .add_op(block, Op::Parameter("b_k".to_string()))
        .unwrap();
    let (_, b_v) = graph
        .add_op(block, Op::Parameter("b_v".to_string()))
        .unwrap();
    let (_, b_o) = graph
        .add_op(block, Op::Parameter("b_o".to_string()))
        .unwrap();

    let mut out_value = None;
    for output_idx in 0..6 {
        let (_, value) = graph
            .add_op(
                block,
                Op::MultiHeadAttention {
                    q_input: x,
                    k_input: x,
                    v_input: x,
                    w_q,
                    w_k,
                    w_v,
                    w_o,
                    bias_q: b_q,
                    bias_k: b_k,
                    bias_v: b_v,
                    bias_o: b_o,
                    num_heads: 2,
                    causal: false,
                    output_idx,
                },
            )
            .unwrap();
        if output_idx == 0 {
            out_value = Some(value);
        }
    }
    let out = out_value.unwrap();
    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: out,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();

    graph.bind_input_shape("x", vec![1, 2, 4]);
    graph.bind_parameter_shape("w_q", vec![4, 4]);
    graph.bind_parameter_shape("w_k", vec![4, 4]);
    graph.bind_parameter_shape("w_v", vec![4, 4]);
    graph.bind_parameter_shape("w_o", vec![4, 4]);
    graph.bind_parameter_shape("b_q", vec![4]);
    graph.bind_parameter_shape("b_k", vec![4]);
    graph.bind_parameter_shape("b_v", vec![4]);
    graph.bind_parameter_shape("b_o", vec![4]);

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        tensor(
            vec![1, 2, 4],
            vec![0.2, -0.1, 0.3, 0.4, 0.7, 0.5, -0.4, 0.1],
        ),
    );
    context.parameters.insert(
        "w_q".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.2, -0.1, 0.3, 0.4, -0.5, 0.6, 0.1, -0.2, 0.7, 0.2, -0.3, 0.5, 0.4, -0.6, 0.8, 0.1,
            ],
        ),
    );
    context.parameters.insert(
        "w_k".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.1, 0.2, -0.4, 0.3, 0.5, -0.7, 0.6, 0.2, -0.3, 0.8, 0.4, -0.1, 0.2, 0.1, 0.5, -0.6,
            ],
        ),
    );
    context.parameters.insert(
        "w_v".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.3, -0.2, 0.1, 0.7, 0.6, 0.4, -0.5, 0.2, 0.2, -0.8, 0.9, 0.1, -0.4, 0.3, 0.2, 0.5,
            ],
        ),
    );
    context.parameters.insert(
        "w_o".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.4, -0.3, 0.2, 0.1, 0.5, 0.6, -0.7, 0.2, -0.1, 0.8, 0.3, -0.4, 0.7, -0.2, 0.5, 0.6,
            ],
        ),
    );
    context.parameters.insert(
        "b_q".to_string(),
        tensor(vec![4], vec![0.05, -0.1, 0.15, -0.2]),
    );
    context.parameters.insert(
        "b_k".to_string(),
        tensor(vec![4], vec![-0.05, 0.2, -0.15, 0.1]),
    );
    context.parameters.insert(
        "b_v".to_string(),
        tensor(vec![4], vec![0.1, 0.05, -0.2, 0.25]),
    );
    context.parameters.insert(
        "b_o".to_string(),
        tensor(vec![4], vec![-0.1, 0.15, 0.05, -0.05]),
    );
    context
        .inputs
        .insert("__loss_grad".to_string(), tensor(vec![1], vec![1.0]));

    let actual_out = tensor_data(execute_value_with_context(&graph, out, &context).unwrap());
    assert_close(
        "mha_self.output",
        &actual_out,
        &json_f32_array(&expected, "output"),
        1e-4,
    );

    let grad_graph = build_reverse_graph(&graph, loss, &[x, b_q, b_o, w_q, w_o]).unwrap();
    for (name, value_id) in [
        ("x", x),
        ("b_q", b_q),
        ("b_o", b_o),
        ("w_q", w_q),
        ("w_o", w_o),
    ] {
        let gradient = *grad_graph.gradients.get(&value_id).unwrap();
        let actual = tensor_data(
            execute_value_with_context(&grad_graph.backward, gradient, &context).unwrap(),
        );
        assert_close(
            &format!("mha_self.grad.{name}"),
            &actual,
            &json_f32_array(&expected["gradients"], name),
            1e-4,
        );
    }
}

#[test]
fn pytorch_parity_transformer_block_forward_and_gradients() {
    let Some(expected) = run_pytorch_case("transformer") else {
        return;
    };

    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
    let (_, w_q) = graph
        .add_op(block, Op::Parameter("w_q".to_string()))
        .unwrap();
    let (_, w_k) = graph
        .add_op(block, Op::Parameter("w_k".to_string()))
        .unwrap();
    let (_, w_v) = graph
        .add_op(block, Op::Parameter("w_v".to_string()))
        .unwrap();
    let (_, w_o) = graph
        .add_op(block, Op::Parameter("w_o".to_string()))
        .unwrap();
    let (_, b_q) = graph
        .add_op(block, Op::Parameter("b_q".to_string()))
        .unwrap();
    let (_, b_k) = graph
        .add_op(block, Op::Parameter("b_k".to_string()))
        .unwrap();
    let (_, b_v) = graph
        .add_op(block, Op::Parameter("b_v".to_string()))
        .unwrap();
    let (_, b_o) = graph
        .add_op(block, Op::Parameter("b_o".to_string()))
        .unwrap();
    let (_, ln1_w) = graph
        .add_op(block, Op::Parameter("ln1_w".to_string()))
        .unwrap();
    let (_, ln1_b) = graph
        .add_op(block, Op::Parameter("ln1_b".to_string()))
        .unwrap();
    let (_, ffn_w1) = graph
        .add_op(block, Op::Parameter("ffn_w1".to_string()))
        .unwrap();
    let (_, ffn_b1) = graph
        .add_op(block, Op::Parameter("ffn_b1".to_string()))
        .unwrap();
    let (_, ffn_w2) = graph
        .add_op(block, Op::Parameter("ffn_w2".to_string()))
        .unwrap();
    let (_, ffn_b2) = graph
        .add_op(block, Op::Parameter("ffn_b2".to_string()))
        .unwrap();
    let (_, ln2_w) = graph
        .add_op(block, Op::Parameter("ln2_w".to_string()))
        .unwrap();
    let (_, ln2_b) = graph
        .add_op(block, Op::Parameter("ln2_b".to_string()))
        .unwrap();

    let config = TransformerConfig {
        d_model: 4,
        num_heads: 2,
        ffn_dim: 6,
        dropout: 0.0,
        causal: false,
        epsilon: 1e-5,
    };

    let out = add_transformer_encoder_block(
        &mut graph, block, x, w_q, w_k, w_v, w_o, b_q, b_k, b_v, b_o, ln1_w, ln1_b, ffn_w1, ffn_b1,
        ffn_w2, ffn_b2, ln2_w, ln2_b, &config,
    )
    .unwrap();

    let (_, loss) = graph
        .add_op(
            block,
            Op::ReduceSum {
                input: out,
                axis: None,
                keepdims: false,
            },
        )
        .unwrap();

    graph.bind_input_shape("x", vec![1, 2, 4]);
    graph.bind_parameter_shape("w_q", vec![4, 4]);
    graph.bind_parameter_shape("w_k", vec![4, 4]);
    graph.bind_parameter_shape("w_v", vec![4, 4]);
    graph.bind_parameter_shape("w_o", vec![4, 4]);
    graph.bind_parameter_shape("b_q", vec![4]);
    graph.bind_parameter_shape("b_k", vec![4]);
    graph.bind_parameter_shape("b_v", vec![4]);
    graph.bind_parameter_shape("b_o", vec![4]);
    graph.bind_parameter_shape("ln1_w", vec![4]);
    graph.bind_parameter_shape("ln1_b", vec![4]);
    graph.bind_parameter_shape("ffn_w1", vec![4, 6]);
    graph.bind_parameter_shape("ffn_b1", vec![6]);
    graph.bind_parameter_shape("ffn_w2", vec![6, 4]);
    graph.bind_parameter_shape("ffn_b2", vec![4]);
    graph.bind_parameter_shape("ln2_w", vec![4]);
    graph.bind_parameter_shape("ln2_b", vec![4]);

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        tensor(
            vec![1, 2, 4],
            vec![0.2, -0.1, 0.3, 0.4, 0.7, 0.5, -0.4, 0.1],
        ),
    );
    context.parameters.insert(
        "w_q".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.2, -0.1, 0.3, 0.4, -0.5, 0.6, 0.1, -0.2, 0.7, 0.2, -0.3, 0.5, 0.4, -0.6, 0.8, 0.1,
            ],
        ),
    );
    context.parameters.insert(
        "w_k".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.1, 0.2, -0.4, 0.3, 0.5, -0.7, 0.6, 0.2, -0.3, 0.8, 0.4, -0.1, 0.2, 0.1, 0.5, -0.6,
            ],
        ),
    );
    context.parameters.insert(
        "w_v".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.3, -0.2, 0.1, 0.7, 0.6, 0.4, -0.5, 0.2, 0.2, -0.8, 0.9, 0.1, -0.4, 0.3, 0.2, 0.5,
            ],
        ),
    );
    context.parameters.insert(
        "w_o".to_string(),
        tensor(
            vec![4, 4],
            vec![
                0.4, -0.3, 0.2, 0.1, 0.5, 0.6, -0.7, 0.2, -0.1, 0.8, 0.3, -0.4, 0.7, -0.2, 0.5, 0.6,
            ],
        ),
    );
    context.parameters.insert(
        "b_q".to_string(),
        tensor(vec![4], vec![0.05, -0.1, 0.15, -0.2]),
    );
    context.parameters.insert(
        "b_k".to_string(),
        tensor(vec![4], vec![-0.05, 0.2, -0.15, 0.1]),
    );
    context.parameters.insert(
        "b_v".to_string(),
        tensor(vec![4], vec![0.1, 0.05, -0.2, 0.25]),
    );
    context.parameters.insert(
        "b_o".to_string(),
        tensor(vec![4], vec![-0.1, 0.15, 0.05, -0.05]),
    );
    context.parameters.insert(
        "ln1_w".to_string(),
        tensor(vec![4], vec![1.0, 0.9, 1.1, -0.8]),
    );
    context.parameters.insert(
        "ln1_b".to_string(),
        tensor(vec![4], vec![0.05, -0.1, 0.15, 0.2]),
    );
    context.parameters.insert(
        "ffn_w1".to_string(),
        tensor(
            vec![4, 6],
            vec![
                0.2, -0.3, 0.1, 0.5, 0.4, -0.2, 0.6, 0.7, -0.5, 0.2, -0.1, 0.3, -0.4, 0.8, 0.9,
                -0.6, 0.2, 0.1, 0.3, -0.7, 0.4, 0.5, -0.8, 0.6,
            ],
        ),
    );
    context.parameters.insert(
        "ffn_b1".to_string(),
        tensor(vec![6], vec![0.1, -0.2, 0.05, 0.15, -0.1, 0.2]),
    );
    context.parameters.insert(
        "ffn_w2".to_string(),
        tensor(
            vec![6, 4],
            vec![
                0.3, -0.4, 0.2, 0.1, 0.5, 0.6, -0.7, 0.2, -0.1, 0.8, 0.3, -0.4, 0.7, -0.2, 0.5,
                0.6, 0.4, 0.1, -0.3, 0.2, -0.5, 0.9, 0.6, -0.7,
            ],
        ),
    );
    context.parameters.insert(
        "ffn_b2".to_string(),
        tensor(vec![4], vec![0.2, -0.15, 0.05, 0.1]),
    );
    context.parameters.insert(
        "ln2_w".to_string(),
        tensor(vec![4], vec![0.95, -1.05, 0.85, 1.1]),
    );
    context.parameters.insert(
        "ln2_b".to_string(),
        tensor(vec![4], vec![-0.05, 0.1, -0.15, 0.2]),
    );
    context
        .inputs
        .insert("__loss_grad".to_string(), tensor(vec![1], vec![1.0]));

    let actual_out = tensor_data(execute_value_with_context(&graph, out, &context).unwrap());
    assert_close(
        "transformer.output",
        &actual_out,
        &json_f32_array(&expected, "output"),
        2e-4,
    );

    let grad_graph = build_reverse_graph(&graph, loss, &[x, w_q, ln1_w, ffn_w1, ln2_w]).unwrap();
    for (name, value_id) in [
        ("x", x),
        ("w_q", w_q),
        ("ln1_w", ln1_w),
        ("ffn_w1", ffn_w1),
        ("ln2_w", ln2_w),
    ] {
        let gradient = *grad_graph.gradients.get(&value_id).unwrap();
        let actual = tensor_data(
            execute_value_with_context(&grad_graph.backward, gradient, &context).unwrap(),
        );
        assert_close(
            &format!("transformer.grad.{name}"),
            &actual,
            &json_f32_array(&expected["gradients"], name),
            2e-4,
        );
    }
}

#[test]
fn pytorch_parity_transformer_multi_step_sgd_train_graph() {
    let Some(expected) = run_pytorch_case("transformer_train_loop_sgd") else {
        return;
    };

    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
    let (_, target) = graph
        .add_op(block, Op::Input("target".to_string()))
        .unwrap();
    let (_, w_q) = graph
        .add_op(block, Op::Parameter("w_q".to_string()))
        .unwrap();
    let (_, w_k) = graph
        .add_op(block, Op::Parameter("w_k".to_string()))
        .unwrap();
    let (_, w_v) = graph
        .add_op(block, Op::Parameter("w_v".to_string()))
        .unwrap();
    let (_, w_o) = graph
        .add_op(block, Op::Parameter("w_o".to_string()))
        .unwrap();
    let (_, b_q) = graph
        .add_op(block, Op::Parameter("b_q".to_string()))
        .unwrap();
    let (_, b_k) = graph
        .add_op(block, Op::Parameter("b_k".to_string()))
        .unwrap();
    let (_, b_v) = graph
        .add_op(block, Op::Parameter("b_v".to_string()))
        .unwrap();
    let (_, b_o) = graph
        .add_op(block, Op::Parameter("b_o".to_string()))
        .unwrap();
    let (_, ln1_w) = graph
        .add_op(block, Op::Parameter("ln1_w".to_string()))
        .unwrap();
    let (_, ln1_b) = graph
        .add_op(block, Op::Parameter("ln1_b".to_string()))
        .unwrap();
    let (_, ffn_w1) = graph
        .add_op(block, Op::Parameter("ffn_w1".to_string()))
        .unwrap();
    let (_, ffn_b1) = graph
        .add_op(block, Op::Parameter("ffn_b1".to_string()))
        .unwrap();
    let (_, ffn_w2) = graph
        .add_op(block, Op::Parameter("ffn_w2".to_string()))
        .unwrap();
    let (_, ffn_b2) = graph
        .add_op(block, Op::Parameter("ffn_b2".to_string()))
        .unwrap();
    let (_, ln2_w) = graph
        .add_op(block, Op::Parameter("ln2_w".to_string()))
        .unwrap();
    let (_, ln2_b) = graph
        .add_op(block, Op::Parameter("ln2_b".to_string()))
        .unwrap();

    let config = TransformerConfig {
        d_model: 4,
        num_heads: 2,
        ffn_dim: 6,
        dropout: 0.0,
        causal: false,
        epsilon: 1e-5,
    };

    let out = add_transformer_encoder_block(
        &mut graph, block, x, w_q, w_k, w_v, w_o, b_q, b_k, b_v, b_o, ln1_w, ln1_b, ffn_w1, ffn_b1,
        ffn_w2, ffn_b2, ln2_w, ln2_b, &config,
    )
    .unwrap();
    let (_, diff) = graph.add_op(block, Op::Sub(out, target)).unwrap();
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

    graph.bind_input_shape("x", vec![1, 2, 4]);
    graph.bind_input_shape("target", vec![2, 4]);
    graph.bind_parameter_shape("w_q", vec![4, 4]);
    graph.bind_parameter_shape("w_k", vec![4, 4]);
    graph.bind_parameter_shape("w_v", vec![4, 4]);
    graph.bind_parameter_shape("w_o", vec![4, 4]);
    graph.bind_parameter_shape("b_q", vec![4]);
    graph.bind_parameter_shape("b_k", vec![4]);
    graph.bind_parameter_shape("b_v", vec![4]);
    graph.bind_parameter_shape("b_o", vec![4]);
    graph.bind_parameter_shape("ln1_w", vec![4]);
    graph.bind_parameter_shape("ln1_b", vec![4]);
    graph.bind_parameter_shape("ffn_w1", vec![4, 6]);
    graph.bind_parameter_shape("ffn_b1", vec![6]);
    graph.bind_parameter_shape("ffn_w2", vec![6, 4]);
    graph.bind_parameter_shape("ffn_b2", vec![4]);
    graph.bind_parameter_shape("ln2_w", vec![4]);
    graph.bind_parameter_shape("ln2_b", vec![4]);

    let initial_parameters = HashMap::from([
        (
            "w_q".to_string(),
            Tensor::new(
                vec![4, 4],
                vec![
                    0.2, -0.1, 0.3, 0.4, -0.5, 0.6, 0.1, -0.2, 0.7, 0.2, -0.3, 0.5, 0.4, -0.6, 0.8,
                    0.1,
                ],
            )
            .unwrap(),
        ),
        (
            "w_k".to_string(),
            Tensor::new(
                vec![4, 4],
                vec![
                    0.1, 0.2, -0.4, 0.3, 0.5, -0.7, 0.6, 0.2, -0.3, 0.8, 0.4, -0.1, 0.2, 0.1, 0.5,
                    -0.6,
                ],
            )
            .unwrap(),
        ),
        (
            "w_v".to_string(),
            Tensor::new(
                vec![4, 4],
                vec![
                    0.3, -0.2, 0.1, 0.7, 0.6, 0.4, -0.5, 0.2, 0.2, -0.8, 0.9, 0.1, -0.4, 0.3, 0.2,
                    0.5,
                ],
            )
            .unwrap(),
        ),
        (
            "w_o".to_string(),
            Tensor::new(
                vec![4, 4],
                vec![
                    0.4, -0.3, 0.2, 0.1, 0.5, 0.6, -0.7, 0.2, -0.1, 0.8, 0.3, -0.4, 0.7, -0.2, 0.5,
                    0.6,
                ],
            )
            .unwrap(),
        ),
        (
            "b_q".to_string(),
            Tensor::new(vec![4], vec![0.05, -0.1, 0.15, -0.2]).unwrap(),
        ),
        (
            "b_k".to_string(),
            Tensor::new(vec![4], vec![-0.05, 0.2, -0.15, 0.1]).unwrap(),
        ),
        (
            "b_v".to_string(),
            Tensor::new(vec![4], vec![0.1, 0.05, -0.2, 0.25]).unwrap(),
        ),
        (
            "b_o".to_string(),
            Tensor::new(vec![4], vec![-0.1, 0.15, 0.05, -0.05]).unwrap(),
        ),
        (
            "ln1_w".to_string(),
            Tensor::new(vec![4], vec![1.0, 0.9, 1.1, -0.8]).unwrap(),
        ),
        (
            "ln1_b".to_string(),
            Tensor::new(vec![4], vec![0.05, -0.1, 0.15, 0.2]).unwrap(),
        ),
        (
            "ffn_w1".to_string(),
            Tensor::new(
                vec![4, 6],
                vec![
                    0.2, -0.3, 0.1, 0.5, 0.4, -0.2, 0.6, 0.7, -0.5, 0.2, -0.1, 0.3, -0.4, 0.8, 0.9,
                    -0.6, 0.2, 0.1, 0.3, -0.7, 0.4, 0.5, -0.8, 0.6,
                ],
            )
            .unwrap(),
        ),
        (
            "ffn_b1".to_string(),
            Tensor::new(vec![6], vec![0.1, -0.2, 0.05, 0.15, -0.1, 0.2]).unwrap(),
        ),
        (
            "ffn_w2".to_string(),
            Tensor::new(
                vec![6, 4],
                vec![
                    0.3, -0.4, 0.2, 0.1, 0.5, 0.6, -0.7, 0.2, -0.1, 0.8, 0.3, -0.4, 0.7, -0.2, 0.5,
                    0.6, 0.4, 0.1, -0.3, 0.2, -0.5, 0.9, 0.6, -0.7,
                ],
            )
            .unwrap(),
        ),
        (
            "ffn_b2".to_string(),
            Tensor::new(vec![4], vec![0.2, -0.15, 0.05, 0.1]).unwrap(),
        ),
        (
            "ln2_w".to_string(),
            Tensor::new(vec![4], vec![0.95, -1.05, 0.85, 1.1]).unwrap(),
        ),
        (
            "ln2_b".to_string(),
            Tensor::new(vec![4], vec![-0.05, 0.1, -0.15, 0.2]).unwrap(),
        ),
    ]);

    let dataset = vec![
        TrainSample {
            inputs: HashMap::from([
                (
                    "x".to_string(),
                    Tensor::new(
                        vec![1, 2, 4],
                        vec![0.2, -0.1, 0.3, 0.4, 0.7, 0.5, -0.4, 0.1],
                    )
                    .unwrap(),
                ),
                (
                    "target".to_string(),
                    Tensor::new(vec![2, 4], vec![0.05, -0.1, 0.2, 0.3, 0.4, -0.2, 0.1, 0.5])
                        .unwrap(),
                ),
            ]),
        },
        TrainSample {
            inputs: HashMap::from([
                (
                    "x".to_string(),
                    Tensor::new(
                        vec![1, 2, 4],
                        vec![0.4, 0.2, -0.5, 0.6, -0.3, 0.8, 0.1, -0.7],
                    )
                    .unwrap(),
                ),
                (
                    "target".to_string(),
                    Tensor::new(
                        vec![2, 4],
                        vec![0.15, 0.05, -0.2, 0.25, -0.1, 0.3, 0.6, -0.4],
                    )
                    .unwrap(),
                ),
            ]),
        },
    ];

    let result = train_graph(
        &graph,
        loss,
        initial_parameters,
        &dataset,
        &[],
        &TrainConfig::new(2, OptimizerConfig::Sgd { lr: 0.01 }),
    )
    .expect("transformer training should succeed");

    assert!(
        (result.final_loss - scalar_f32(&expected, "final_loss")).abs() <= 1e-5,
        "transformer final_loss mismatch: actual={}, expected={}",
        result.final_loss,
        scalar_f32(&expected, "final_loss")
    );

    for name in ["w_q", "b_q", "w_o", "ln1_w", "ffn_w1", "ln2_w"] {
        let actual = result
            .final_parameters
            .get(name)
            .unwrap_or_else(|| panic!("missing final parameter '{name}'"))
            .make_contiguous()
            .expect("contiguous final parameter")
            .data
            .to_vec();
        let expected_values = json_f32_array(&expected["final_parameters"], name);
        assert_close(
            &format!("transformer_train_loop.param.{name}"),
            &actual,
            &expected_values,
            1e-5,
        );
    }
}

#[test]
fn pytorch_parity_mlp_single_sgd_step() {
    let Some(expected) = run_pytorch_case("mlp_train_sgd") else {
        return;
    };

    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
    let (_, target) = graph
        .add_op(block, Op::Input("target".to_string()))
        .unwrap();
    let (_, w1) = graph
        .add_op(block, Op::Parameter("w1".to_string()))
        .unwrap();
    let (_, b1) = graph
        .add_op(block, Op::Parameter("b1".to_string()))
        .unwrap();
    let (_, w2) = graph
        .add_op(block, Op::Parameter("w2".to_string()))
        .unwrap();
    let (_, b2) = graph
        .add_op(block, Op::Parameter("b2".to_string()))
        .unwrap();

    let (_, hidden) = graph
        .add_op(
            block,
            Op::Gemm {
                lhs: x,
                rhs: w1,
                bias: Some(b1),
                alpha: 1.0,
                beta: 1.0,
            },
        )
        .unwrap();
    let (_, hidden_relu) = graph.add_op(block, Op::Relu(hidden)).unwrap();
    let (_, out) = graph
        .add_op(
            block,
            Op::Gemm {
                lhs: hidden_relu,
                rhs: w2,
                bias: Some(b2),
                alpha: 1.0,
                beta: 1.0,
            },
        )
        .unwrap();
    let (_, diff) = graph.add_op(block, Op::Sub(out, target)).unwrap();
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

    graph.bind_input_shape("x", vec![2, 3]);
    graph.bind_input_shape("target", vec![2, 2]);
    graph.bind_parameter_shape("w1", vec![3, 4]);
    graph.bind_parameter_shape("b1", vec![4]);
    graph.bind_parameter_shape("w2", vec![4, 2]);
    graph.bind_parameter_shape("b2", vec![2]);

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        tensor(vec![2, 3], vec![0.2, -0.1, 0.3, 0.7, 0.5, -0.4]),
    );
    context.inputs.insert(
        "target".to_string(),
        tensor(vec![2, 2], vec![0.1, -0.2, 0.3, 0.4]),
    );
    context.parameters.insert(
        "w1".to_string(),
        tensor(
            vec![3, 4],
            vec![
                0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.2, -0.3,
            ],
        ),
    );
    context.parameters.insert(
        "b1".to_string(),
        tensor(vec![4], vec![0.05, -0.1, 0.15, 0.2]),
    );
    context.parameters.insert(
        "w2".to_string(),
        tensor(vec![4, 2], vec![0.2, -0.4, 0.1, 0.3, -0.5, 0.7, 0.6, -0.2]),
    );
    context
        .parameters
        .insert("b2".to_string(), tensor(vec![2], vec![0.25, -0.35]));
    context
        .inputs
        .insert("__loss_grad".to_string(), tensor(vec![1], vec![1.0]));

    let loss_before = tensor_data(execute_value_with_context(&graph, loss, &context).unwrap())[0];
    assert!(
        (loss_before - scalar_f32(&expected, "loss_before")).abs() <= 1e-6,
        "loss_before mismatch: actual={loss_before}, expected={}",
        scalar_f32(&expected, "loss_before")
    );

    let grad_graph = build_reverse_graph(&graph, loss, &[w1, b1, w2, b2]).unwrap();
    let mut gradients = HashMap::new();
    for (name, value_id, shape) in [
        ("w1", w1, vec![3, 4]),
        ("b1", b1, vec![4]),
        ("w2", w2, vec![4, 2]),
        ("b2", b2, vec![2]),
    ] {
        let gradient_value = *grad_graph.gradients.get(&value_id).unwrap();
        let actual = tensor_data(
            execute_value_with_context(&grad_graph.backward, gradient_value, &context).unwrap(),
        );
        assert_close(
            &format!("mlp_train.grad.{name}"),
            &actual,
            &json_f32_array(&expected["gradients"], name),
            1e-5,
        );
        gradients.insert(value_id, Tensor::new(shape, actual).unwrap());
    }

    let mut parameters = HashMap::from([
        (
            w1,
            Tensor::new(
                vec![3, 4],
                vec![
                    0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.2, -0.3,
                ],
            )
            .unwrap(),
        ),
        (
            b1,
            Tensor::new(vec![4], vec![0.05, -0.1, 0.15, 0.2]).unwrap(),
        ),
        (
            w2,
            Tensor::new(vec![4, 2], vec![0.2, -0.4, 0.1, 0.3, -0.5, 0.7, 0.6, -0.2]).unwrap(),
        ),
        (b2, Tensor::new(vec![2], vec![0.25, -0.35]).unwrap()),
    ]);
    let mut optimizer_state = OptimizerState::default();
    apply_gradients(
        &mut parameters,
        &gradients,
        &OptimizerConfig::Sgd { lr: 0.05 },
        &mut optimizer_state,
    )
    .unwrap();

    for (name, value_id) in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)] {
        let actual = parameters.get(&value_id).unwrap().data.to_vec();
        assert_close(
            &format!("mlp_train.param.{name}"),
            &actual,
            &json_f32_array(&expected["updated_parameters"], name),
            1e-5,
        );
    }

    let mut post_context = context.clone();
    post_context.parameters = HashMap::from([
        (
            "w1".to_string(),
            RuntimeValue::Tensor(Arc::new(parameters[&w1].clone())),
        ),
        (
            "b1".to_string(),
            RuntimeValue::Tensor(Arc::new(parameters[&b1].clone())),
        ),
        (
            "w2".to_string(),
            RuntimeValue::Tensor(Arc::new(parameters[&w2].clone())),
        ),
        (
            "b2".to_string(),
            RuntimeValue::Tensor(Arc::new(parameters[&b2].clone())),
        ),
    ]);
    let loss_after =
        tensor_data(execute_value_with_context(&graph, loss, &post_context).unwrap())[0];
    assert!(
        (loss_after - scalar_f32(&expected, "loss_after")).abs() <= 1e-6,
        "loss_after mismatch: actual={loss_after}, expected={}",
        scalar_f32(&expected, "loss_after")
    );
}

#[test]
fn pytorch_parity_mlp_multi_step_sgd_train_graph() {
    let Some(expected) = run_pytorch_case("mlp_train_loop_sgd") else {
        return;
    };

    let mut graph = Graph::new();
    let block = graph.create_block();

    let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
    let (_, target) = graph
        .add_op(block, Op::Input("target".to_string()))
        .unwrap();
    let (_, w1) = graph
        .add_op(block, Op::Parameter("w1".to_string()))
        .unwrap();
    let (_, b1) = graph
        .add_op(block, Op::Parameter("b1".to_string()))
        .unwrap();
    let (_, w2) = graph
        .add_op(block, Op::Parameter("w2".to_string()))
        .unwrap();
    let (_, b2) = graph
        .add_op(block, Op::Parameter("b2".to_string()))
        .unwrap();

    let (_, hidden) = graph
        .add_op(
            block,
            Op::Gemm {
                lhs: x,
                rhs: w1,
                bias: Some(b1),
                alpha: 1.0,
                beta: 1.0,
            },
        )
        .unwrap();
    let (_, hidden_relu) = graph.add_op(block, Op::Relu(hidden)).unwrap();
    let (_, out) = graph
        .add_op(
            block,
            Op::Gemm {
                lhs: hidden_relu,
                rhs: w2,
                bias: Some(b2),
                alpha: 1.0,
                beta: 1.0,
            },
        )
        .unwrap();
    let (_, diff) = graph.add_op(block, Op::Sub(out, target)).unwrap();
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

    graph.bind_input_shape("x", vec![2, 3]);
    graph.bind_input_shape("target", vec![2, 2]);
    graph.bind_parameter_shape("w1", vec![3, 4]);
    graph.bind_parameter_shape("b1", vec![4]);
    graph.bind_parameter_shape("w2", vec![4, 2]);
    graph.bind_parameter_shape("b2", vec![2]);

    let initial_parameters = HashMap::from([
        (
            "w1".to_string(),
            Tensor::new(
                vec![3, 4],
                vec![
                    0.1, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.2, -0.3,
                ],
            )
            .unwrap(),
        ),
        (
            "b1".to_string(),
            Tensor::new(vec![4], vec![0.05, -0.1, 0.15, 0.2]).unwrap(),
        ),
        (
            "w2".to_string(),
            Tensor::new(vec![4, 2], vec![0.2, -0.4, 0.1, 0.3, -0.5, 0.7, 0.6, -0.2]).unwrap(),
        ),
        (
            "b2".to_string(),
            Tensor::new(vec![2], vec![0.25, -0.35]).unwrap(),
        ),
    ]);

    let dataset = vec![
        TrainSample {
            inputs: HashMap::from([
                (
                    "x".to_string(),
                    Tensor::new(vec![2, 3], vec![0.2, -0.1, 0.3, 0.7, 0.5, -0.4]).unwrap(),
                ),
                (
                    "target".to_string(),
                    Tensor::new(vec![2, 2], vec![0.1, -0.2, 0.3, 0.4]).unwrap(),
                ),
            ]),
        },
        TrainSample {
            inputs: HashMap::from([
                (
                    "x".to_string(),
                    Tensor::new(vec![2, 3], vec![0.4, 0.2, -0.5, -0.3, 0.6, 0.8]).unwrap(),
                ),
                (
                    "target".to_string(),
                    Tensor::new(vec![2, 2], vec![0.2, 0.05, -0.1, 0.6]).unwrap(),
                ),
            ]),
        },
    ];

    let result = train_graph(
        &graph,
        loss,
        initial_parameters,
        &dataset,
        &[],
        &TrainConfig::new(3, OptimizerConfig::Sgd { lr: 0.05 }),
    )
    .expect("multi-step training should succeed");

    assert!(
        (result.final_loss - scalar_f32(&expected, "final_loss")).abs() <= 1e-5,
        "final_loss mismatch: actual={}, expected={}",
        result.final_loss,
        scalar_f32(&expected, "final_loss")
    );

    for name in ["w1", "b1", "w2", "b2"] {
        let actual = result
            .final_parameters
            .get(name)
            .unwrap_or_else(|| panic!("missing final parameter '{name}'"))
            .data
            .to_vec();
        assert_close(
            &format!("mlp_train_loop.param.{name}"),
            &actual,
            &json_f32_array(&expected["final_parameters"], name),
            1e-5,
        );
    }
}
