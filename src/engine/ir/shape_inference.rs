use std::collections::HashMap;

use crate::ir::{Graph, Op, ValueId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeFact {
    NonTensor,
    Unknown,
    Tensor(Vec<usize>),
}

#[derive(Debug, Clone)]
pub struct ShapeError {
    pub message: String,
}

pub fn infer_shapes(graph: &Graph) -> Result<HashMap<ValueId, ShapeFact>, ShapeError> {
    let mut shapes = HashMap::new();

    for node in &graph.nodes {
        let inferred =
            infer_shape_for_op(graph, &node.op, &shapes).map_err(|message| ShapeError {
                message: format!("{} (at node {})", message, node.id.0),
            })?;
        shapes.insert(node.output, inferred);
    }

    Ok(shapes)
}

fn infer_shape_for_op(
    graph: &Graph,
    op: &Op,
    shapes: &HashMap<ValueId, ShapeFact>,
) -> Result<ShapeFact, String> {
    match op {
        Op::Plugin { operator, inputs } => {
            let mut input_shapes = Vec::with_capacity(inputs.len());
            for input in inputs {
                input_shapes.push(shape_of(*input, shapes));
            }
            operator.check_constraints(&input_shapes)?;
            operator.infer_shape(&input_shapes)
        }
        Op::ConstInt(_) | Op::ConstFloat(_) => Ok(ShapeFact::NonTensor),
        Op::ConstTensor { shape, .. } => Ok(ShapeFact::Tensor(shape.clone())),
        Op::Parameter(name) => Ok(graph
            .parameter_shape(name)
            .map(|shape| ShapeFact::Tensor(shape.to_vec()))
            .unwrap_or(ShapeFact::Unknown)),
        Op::Input(name) => Ok(graph
            .input_shape(name)
            .map(|shape| ShapeFact::Tensor(shape.to_vec()))
            .unwrap_or(ShapeFact::Unknown)),
        Op::Output(value) => Ok(shape_of(*value, shapes)),
        Op::Add(left, right)
        | Op::Sub(left, right)
        | Op::Mul(left, right)
        | Op::Div(left, right) => infer_elementwise(*left, *right, shapes),
        Op::Neg(value) => Ok(shape_of(*value, shapes)),
        Op::ElementwiseChain { input, .. } => infer_tensor_unary(*input, shapes),
        Op::Reshape { input, shape } => infer_reshape(*input, shape, shapes),
        Op::Concat { inputs, axis } => infer_concat(inputs, *axis, shapes),
        Op::Gather {
            input,
            indices,
            axis,
        } => infer_gather(*input, indices, *axis, shapes),
        Op::Slice {
            input,
            starts,
            ends,
            axes,
        } => infer_slice(*input, starts, ends, axes, shapes),
        Op::Transpose(value) => match shape_of(*value, shapes) {
            ShapeFact::Tensor(shape) => {
                if shape.len() != 2 {
                    Err(format!(
                        "transpose expects rank-2 tensor, got shape {:?}",
                        shape
                    ))
                } else {
                    Ok(ShapeFact::Tensor(vec![shape[1], shape[0]]))
                }
            }
            ShapeFact::Unknown => Ok(ShapeFact::Unknown),
            ShapeFact::NonTensor => Err("transpose expects tensor input".to_string()),
        },
        Op::MatMul(left, right) => infer_matmul(*left, *right, shapes),
        Op::Relu(value)
        | Op::Softmax(value)
        | Op::Log(value)
        | Op::Exp(value)
        | Op::Sigmoid(value)
        | Op::GeluExact(value)
        | Op::Gelu(value) => infer_tensor_unary(*value, shapes),
        Op::SigmoidBackward(input, grad)
        | Op::GeluBackward(input, grad)
        | Op::GeluExactBackward(input, grad) => {
            infer_same_tensor(*input, *grad, shapes, "backward")
        }
        Op::Gemm { lhs, rhs, .. } => infer_matmul(*lhs, *rhs, shapes),
        Op::GemmBackward { lhs, rhs, .. } => infer_matmul(*lhs, *rhs, shapes),
        Op::ReduceSum {
            input,
            axis,
            keepdims,
        } => infer_reduce(*input, *axis, *keepdims, shapes, "reduce_sum"),
        Op::ReduceMax {
            input,
            axis,
            keepdims,
        } => infer_reduce(*input, *axis, *keepdims, shapes, "reduce_max"),
        Op::ReduceMean {
            input,
            axis,
            keepdims,
        } => infer_reduce(*input, *axis, *keepdims, shapes, "reduce_mean"),
        Op::ReluBackward(input, grad) => infer_same_tensor(*input, *grad, shapes, "relu_backward"),
        Op::Conv2D(input, weight) => infer_conv2d(*input, *weight, shapes),
        Op::Conv2DBackwardInput(input, _weight, upstream) => {
            infer_same_tensor(*input, *upstream, shapes, "conv2d_backward_input")
        }
        Op::Conv2DBackwardWeight(_input, weight, upstream) => {
            infer_same_tensor(*weight, *upstream, shapes, "conv2d_backward_weight")
        }
        Op::MaxPool {
            input,
            kernel_shape,
            strides,
            pads,
        }
        | Op::AvgPool {
            input,
            kernel_shape,
            strides,
            pads,
        } => infer_pool2d_nchw(*input, kernel_shape, strides, pads, shapes),
        Op::BatchNorm { input, .. } | Op::Dropout { input, .. } | Op::Identity(input) => {
            infer_tensor_unary(*input, shapes)
        }
        Op::MaxPoolBackward { input, .. }
        | Op::AvgPoolBackward { input, .. }
        | Op::ReduceMaxBackward { input, .. }
        | Op::BatchNormBackwardInput { input, .. } => infer_tensor_unary(*input, shapes),
        Op::BatchNormBackwardWeight { input, .. }
        | Op::BatchNormBackwardBias { upstream: input } => match shape_of(*input, shapes) {
            ShapeFact::Tensor(shape) => {
                if shape.len() != 4 {
                    Err(format!(
                        "BatchNorm backward expects rank-4 tensor, got {:?}",
                        shape
                    ))
                } else {
                    Ok(ShapeFact::Tensor(vec![shape[1]]))
                }
            }
            ShapeFact::Unknown => Ok(ShapeFact::Unknown),
            ShapeFact::NonTensor => Err("BatchNorm backward expects tensor input".to_string()),
        },
        Op::LayerNorm { input, .. } => infer_tensor_unary(*input, shapes),
        Op::LayerNormBackwardInput { input, .. } => infer_tensor_unary(*input, shapes),
        Op::LayerNormBackwardWeight { input, .. }
        | Op::LayerNormBackwardBias { upstream: input } => match shape_of(*input, shapes) {
            ShapeFact::Tensor(shape) => {
                if shape.is_empty() {
                    Err("LayerNorm backward expects at least rank-1 tensor".to_string())
                } else {
                    Ok(ShapeFact::Tensor(vec![*shape.last().unwrap()]))
                }
            }
            ShapeFact::Unknown => Ok(ShapeFact::Unknown),
            ShapeFact::NonTensor => Err("LayerNorm backward expects tensor input".to_string()),
        },
        Op::Flatten { input, axis } => match shape_of(*input, shapes) {
            ShapeFact::Tensor(shape) => {
                if *axis > shape.len() {
                    Err(format!(
                        "flatten axis {} out of bounds for shape {:?}",
                        axis, shape
                    ))
                } else {
                    let left = shape[..*axis].iter().product::<usize>();
                    let right = shape[*axis..].iter().product::<usize>();
                    Ok(ShapeFact::Tensor(vec![left.max(1), right.max(1)]))
                }
            }
            ShapeFact::Unknown => Ok(ShapeFact::Unknown),
            ShapeFact::NonTensor => Err("flatten expects tensor input".to_string()),
        },
        Op::LstmCell {
            h_prev,
            weight_ih,
            output_idx,
            ..
        } => {
            match (shape_of(*h_prev, shapes), shape_of(*weight_ih, shapes)) {
                (ShapeFact::Tensor(h), ShapeFact::Tensor(w)) if h.len() == 2 && w.len() == 2 => {
                    let batch = h[0];
                    let hidden = h[1];
                    let out_shape = match output_idx {
                        0 | 1 => vec![batch, hidden], // h_next / c_next
                        2 => vec![batch, 4 * hidden], // gates_raw
                        3 => vec![batch, hidden],     // tanh_c_next
                        _ => return Err("LstmCell invalid output_idx".to_string()),
                    };
                    Ok(ShapeFact::Tensor(out_shape))
                }
                (ShapeFact::Unknown, _) | (_, ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
                _ => Err("LstmCell expects rank-2 h_prev and weight_ih".to_string()),
            }
        }
        Op::LstmCellBackward {
            x,
            h_prev,
            weight_ih,
            weight_hh,
            grad_target,
            ..
        } => {
            match grad_target {
                0 => match shape_of(*x, shapes) {
                    ShapeFact::Tensor(s) => Ok(ShapeFact::Tensor(s)),
                    o => Ok(o),
                },
                1 | 2 => match shape_of(*h_prev, shapes) {
                    ShapeFact::Tensor(s) => Ok(ShapeFact::Tensor(s)),
                    o => Ok(o),
                },
                3 => match shape_of(*weight_ih, shapes) {
                    ShapeFact::Tensor(s) => Ok(ShapeFact::Tensor(s)),
                    o => Ok(o),
                },
                4 => match shape_of(*weight_hh, shapes) {
                    ShapeFact::Tensor(s) => Ok(ShapeFact::Tensor(s)),
                    o => Ok(o),
                },
                // bias grad shape: [4*hidden], derive from weight_hh [4*hidden, hidden]
                5 => match shape_of(*weight_hh, shapes) {
                    ShapeFact::Tensor(s) if !s.is_empty() => Ok(ShapeFact::Tensor(vec![s[0]])),
                    other => Ok(other),
                },
                _ => Err("LstmCellBackward invalid grad_target".to_string()),
            }
        }
        Op::GruCell {
            h_prev, weight_ih, ..
        } => match (shape_of(*h_prev, shapes), shape_of(*weight_ih, shapes)) {
            (ShapeFact::Tensor(h), ShapeFact::Tensor(_)) if h.len() == 2 => {
                let (batch, hidden) = (h[0], h[1]);
                Ok(ShapeFact::Tensor(vec![batch, hidden]))
            }
            (ShapeFact::Unknown, _) | (_, ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
            _ => Err("GruCell expects rank-2 h_prev".to_string()),
        },
        Op::GruCellBackward {
            x,
            h_prev,
            weight_ih,
            weight_hh,
            grad_target,
            ..
        } => {
            match grad_target {
                0 => match shape_of(*x, shapes) {
                    ShapeFact::Tensor(s) => Ok(ShapeFact::Tensor(s)),
                    o => Ok(o),
                },
                1 => match shape_of(*h_prev, shapes) {
                    ShapeFact::Tensor(s) => Ok(ShapeFact::Tensor(s)),
                    o => Ok(o),
                },
                2 => match shape_of(*weight_ih, shapes) {
                    ShapeFact::Tensor(s) => Ok(ShapeFact::Tensor(s)),
                    o => Ok(o),
                },
                3 => match shape_of(*weight_hh, shapes) {
                    ShapeFact::Tensor(s) => Ok(ShapeFact::Tensor(s)),
                    o => Ok(o),
                },
                // Bias shapes: [3*hidden] derived from weight_ih [3*hidden, input_size]
                4 | 5 => match shape_of(*weight_ih, shapes) {
                    ShapeFact::Tensor(s) if !s.is_empty() => Ok(ShapeFact::Tensor(vec![s[0]])),
                    other => Ok(other),
                },
                _ => Err("GruCellBackward invalid grad_target".to_string()),
            }
        }
        Op::ConvTranspose2D {
            input,
            weight,
            stride,
            padding,
        } => match (shape_of(*input, shapes), shape_of(*weight, shapes)) {
            (ShapeFact::Tensor(inp), ShapeFact::Tensor(w)) if inp.len() == 4 && w.len() == 4 => {
                let (n, c_out, kh, kw) = (inp[0], w[1], w[2], w[3]);
                let out_h = (inp[2] - 1) * stride[0] + kh - 2 * padding[0];
                let out_w = (inp[3] - 1) * stride[1] + kw - 2 * padding[1];
                Ok(ShapeFact::Tensor(vec![n, c_out, out_h, out_w]))
            }
            (ShapeFact::Unknown, _) | (_, ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
            _ => Err("ConvTranspose2D expects rank-4 input and weight".to_string()),
        },
        Op::Upsample2D {
            input,
            scale_h,
            scale_w,
            ..
        } => match shape_of(*input, shapes) {
            ShapeFact::Tensor(s) if s.len() == 4 => {
                let out_h = (s[2] as f32 * scale_h).round() as usize;
                let out_w = (s[3] as f32 * scale_w).round() as usize;
                Ok(ShapeFact::Tensor(vec![s[0], s[1], out_h, out_w]))
            }
            ShapeFact::Unknown => Ok(ShapeFact::Unknown),
            _ => Err("Upsample2D expects rank-4 NCHW input".to_string()),
        },
        Op::Upsample2DBackward {
            upstream,
            orig_h,
            orig_w,
            ..
        } => match shape_of(*upstream, shapes) {
            ShapeFact::Tensor(s) if s.len() == 4 => {
                Ok(ShapeFact::Tensor(vec![s[0], s[1], *orig_h, *orig_w]))
            }
            other => Ok(other),
        },
        Op::MultiHeadAttention {
            q_input,
            k_input,
            num_heads,
            output_idx,
            ..
        } => match (shape_of(*q_input, shapes), shape_of(*k_input, shapes)) {
            (ShapeFact::Tensor(q), ShapeFact::Tensor(k)) if q.len() == 3 && k.len() == 3 => {
                let (batch, seq_q, d_model) = (q[0], q[1], q[2]);
                let seq_k = k[1];
                let s = match output_idx {
                    0 | 5 => vec![batch, seq_q, d_model],
                    1 => vec![batch * num_heads, seq_q, seq_k],
                    2 | 3 | 4 => vec![batch, seq_q.max(seq_k), d_model],
                    _ => return Err("MultiHeadAttention invalid output_idx".to_string()),
                };
                Ok(ShapeFact::Tensor(s))
            }
            (ShapeFact::Unknown, _) | (_, ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
            _ => Err("MultiHeadAttention expects rank-3 q_input and k_input".to_string()),
        },
        Op::SinusoidalPE { input } | Op::RoPE { input, .. } => match shape_of(*input, shapes) {
            ShapeFact::Tensor(s) => Ok(ShapeFact::Tensor(s)),
            other => Ok(other),
        },
        Op::RoPEBackward { upstream, .. } => match shape_of(*upstream, shapes) {
            ShapeFact::Tensor(s) => Ok(ShapeFact::Tensor(s)),
            other => Ok(other),
        },
        Op::Embedding { weight, indices } => {
            match (shape_of(*weight, shapes), shape_of(*indices, shapes)) {
                (ShapeFact::Tensor(w), ShapeFact::Tensor(idx)) if w.len() == 2 => {
                    let embed_dim = w[1];
                    let mut out_shape = idx.clone();
                    out_shape.push(embed_dim);
                    Ok(ShapeFact::Tensor(out_shape))
                }
                (ShapeFact::Unknown, _) | (_, ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
                _ => Err("Embedding weight must be rank-2".to_string()),
            }
        }
        Op::EmbeddingBackward { weight, .. } => match shape_of(*weight, shapes) {
            ShapeFact::Tensor(s) => Ok(ShapeFact::Tensor(s)),
            other => Ok(other),
        },
        Op::GlobalAveragePool { input } => match shape_of(*input, shapes) {
            // Input: [N, C, H, W] or [N, C, ...] → output: [N, C, 1, 1] or [N, C]
            ShapeFact::Tensor(shape) if shape.len() >= 2 => {
                let mut out = shape[..2].to_vec();
                for _ in 2..shape.len() {
                    out.push(1);
                }
                Ok(ShapeFact::Tensor(out))
            }
            ShapeFact::Tensor(shape) => Err(format!(
                "GlobalAveragePool expects at least rank-2 tensor, got {:?}",
                shape
            )),
            ShapeFact::Unknown => Ok(ShapeFact::Unknown),
            ShapeFact::NonTensor => Err("GlobalAveragePool expects tensor input".to_string()),
        },
        Op::GlobalAveragePoolBackward { input, .. } => match shape_of(*input, shapes) {
            ShapeFact::Tensor(shape) => Ok(ShapeFact::Tensor(shape)),
            other => Ok(other),
        },
        Op::GroupNorm { input, .. } | Op::InstanceNorm { input, .. } => {
            match shape_of(*input, shapes) {
                ShapeFact::Tensor(shape) => Ok(ShapeFact::Tensor(shape)),
                other => Ok(other),
            }
        }
        Op::GroupNormBackwardInput { input, .. } | Op::InstanceNormBackwardInput { input, .. } => {
            match shape_of(*input, shapes) {
                ShapeFact::Tensor(shape) => Ok(ShapeFact::Tensor(shape)),
                other => Ok(other),
            }
        }
        Op::GroupNormBackwardWeight { input, .. }
        | Op::InstanceNormBackwardWeight { input, .. } => match shape_of(*input, shapes) {
            // dW has shape [C]
            ShapeFact::Tensor(shape) if shape.len() >= 2 => Ok(ShapeFact::Tensor(vec![shape[1]])),
            ShapeFact::Unknown => Ok(ShapeFact::Unknown),
            _ => Err("GroupNorm/InstanceNorm backward weight expects rank >= 2 input".to_string()),
        },
        Op::GroupNormBackwardBias { upstream } | Op::InstanceNormBackwardBias { upstream } => {
            match shape_of(*upstream, shapes) {
                // dB has shape [C]
                ShapeFact::Tensor(shape) if shape.len() >= 2 => {
                    Ok(ShapeFact::Tensor(vec![shape[1]]))
                }
                ShapeFact::Unknown => Ok(ShapeFact::Unknown),
                _ => Err(
                    "GroupNorm/InstanceNorm backward bias expects rank >= 2 upstream".to_string(),
                ),
            }
        }
        Op::Phi(values) => infer_phi(values, shapes),
        Op::CustomCall { inputs, .. } => {
            // Unknown output shape; propagate first input's shape if available
            if let Some(first) = inputs.first() {
                Ok(shape_of(*first, shapes))
            } else {
                Ok(ShapeFact::Unknown)
            }
        }
        Op::QuantizeLinear { input, .. } | Op::DequantizeLinear { input, .. } => {
            // Same shape as input (simulated quantization stores values as f32)
            Ok(shape_of(*input, shapes))
        }
        Op::DepthwiseSeparableConv {
            input,
            pw_weight,
            stride,
            padding,
            ..
        } => {
            // Output shape: [N, C_out, H_out, W_out]
            let input_shape = shape_of(*input, shapes);
            let pw_shape = shape_of(*pw_weight, shapes);
            match (input_shape, pw_shape) {
                (ShapeFact::Tensor(inp), ShapeFact::Tensor(pw))
                    if inp.len() == 4 && !pw.is_empty() =>
                {
                    let n = inp[0];
                    let h = inp[2];
                    let w = inp[3];
                    // We don't know the depthwise kernel size here, use 3x3 as estimate
                    let out_h = (h + 2 * padding[0]).saturating_sub(2) / stride[0] + 1;
                    let out_w = (w + 2 * padding[1]).saturating_sub(2) / stride[1] + 1;
                    let c_out = pw[0];
                    Ok(ShapeFact::Tensor(vec![n, c_out, out_h, out_w]))
                }
                _ => Ok(ShapeFact::Unknown),
            }
        }
        Op::Removed => Ok(ShapeFact::Unknown),
        Op::MultiHeadAttentionBackward {
            q_input,
            k_input,
            v_input,
            w_q,
            bias_q,
            output_idx,
            ..
        } => match output_idx {
            0 => match shape_of(*q_input, shapes) {
                ShapeFact::Tensor(s) if s.len() == 3 => Ok(ShapeFact::Tensor(s)),
                ShapeFact::Unknown => Ok(ShapeFact::Unknown),
                _ => Err("MultiHeadAttentionBackward dq: expected rank-3 q_input".to_string()),
            },
            1 => match shape_of(*k_input, shapes) {
                ShapeFact::Tensor(s) if s.len() == 3 => Ok(ShapeFact::Tensor(s)),
                ShapeFact::Unknown => Ok(ShapeFact::Unknown),
                _ => Err("MultiHeadAttentionBackward dk: expected rank-3 k_input".to_string()),
            },
            2 => match shape_of(*v_input, shapes) {
                ShapeFact::Tensor(s) if s.len() == 3 => Ok(ShapeFact::Tensor(s)),
                ShapeFact::Unknown => Ok(ShapeFact::Unknown),
                _ => Err("MultiHeadAttentionBackward dv: expected rank-3 v_input".to_string()),
            },
            3..=6 => match shape_of(*w_q, shapes) {
                ShapeFact::Tensor(s) if s.len() == 2 => Ok(ShapeFact::Tensor(s)),
                ShapeFact::Unknown => Ok(ShapeFact::Unknown),
                _ => Err(
                    "MultiHeadAttentionBackward dW: expected rank-2 projection weight".to_string(),
                ),
            },
            7..=10 => match shape_of(*bias_q, shapes) {
                ShapeFact::Tensor(s) if s.len() == 1 => Ok(ShapeFact::Tensor(s)),
                ShapeFact::Unknown => Ok(ShapeFact::Unknown),
                _ => Err(
                    "MultiHeadAttentionBackward db: expected rank-1 projection bias".to_string(),
                ),
            },
            _ => Err(format!(
                "MultiHeadAttentionBackward invalid output_idx={output_idx}"
            )),
        },
        Op::SoftmaxCrossEntropyLossFromLogits { logits, targets } => {
            let log_shape = shape_of(*logits, shapes);
            let target_shape = shape_of(*targets, shapes);
            match (log_shape, target_shape) {
                (ShapeFact::Tensor(l), ShapeFact::Tensor(t)) => {
                    if l != t {
                        return Err(format!(
                            "Shape mismatch in SoftmaxCrossEntropy: {:?} and {:?}",
                            l, t
                        ));
                    }
                    Ok(ShapeFact::Tensor(vec![1]))
                }
                (ShapeFact::Unknown, _) | (_, ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
                _ => Err("SoftmaxCrossEntropy expects tensor inputs".to_string()),
            }
        }
    }
}

fn infer_elementwise(
    left: ValueId,
    right: ValueId,
    shapes: &HashMap<ValueId, ShapeFact>,
) -> Result<ShapeFact, String> {
    let left_shape = shape_of(left, shapes);
    let right_shape = shape_of(right, shapes);
    match (left_shape, right_shape) {
        (ShapeFact::NonTensor, ShapeFact::NonTensor) => Ok(ShapeFact::NonTensor),
        (ShapeFact::Tensor(a), ShapeFact::Tensor(b)) => {
            if let Some(out_shape) = broadcast_shapes(&a, &b) {
                Ok(ShapeFact::Tensor(out_shape))
            } else {
                Err(format!("Shape mismatch in elementwise op: {a:?} vs {b:?}"))
            }
        }
        (ShapeFact::Unknown, _) | (_, ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
        (ShapeFact::NonTensor, ShapeFact::Tensor(_))
        | (ShapeFact::Tensor(_), ShapeFact::NonTensor) => {
            Err("Type mismatch in elementwise op: tensor vs non-tensor".to_string())
        }
    }
}

fn infer_tensor_unary(
    value: ValueId,
    shapes: &HashMap<ValueId, ShapeFact>,
) -> Result<ShapeFact, String> {
    match shape_of(value, shapes) {
        ShapeFact::Tensor(shape) => Ok(ShapeFact::Tensor(shape)),
        ShapeFact::Unknown => Ok(ShapeFact::Unknown),
        ShapeFact::NonTensor => Err("Tensor unary op expects tensor input".to_string()),
    }
}

fn infer_same_tensor(
    left: ValueId,
    right: ValueId,
    shapes: &HashMap<ValueId, ShapeFact>,
    label: &str,
) -> Result<ShapeFact, String> {
    let l = shape_of(left, shapes);
    let r = shape_of(right, shapes);
    match (l, r) {
        (ShapeFact::Tensor(a), ShapeFact::Tensor(b)) => {
            if a == b {
                Ok(ShapeFact::Tensor(a))
            } else {
                Err(format!("Shape mismatch in {label}: {:?} vs {:?}", a, b))
            }
        }
        (ShapeFact::Unknown, _) | (_, ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
        (ShapeFact::NonTensor, _) | (_, ShapeFact::NonTensor) => {
            Err(format!("{label} expects tensor inputs"))
        }
    }
}

fn infer_matmul(
    left: ValueId,
    right: ValueId,
    shapes: &HashMap<ValueId, ShapeFact>,
) -> Result<ShapeFact, String> {
    let left_shape = shape_of(left, shapes);
    let right_shape = shape_of(right, shapes);
    match (left_shape, right_shape) {
        (ShapeFact::Tensor(a), ShapeFact::Tensor(b)) => {
            if a.len() != 2 || b.len() != 2 {
                return Err(format!(
                    "MatMul expects rank-2 tensors, got {:?} and {:?}",
                    a, b
                ));
            }
            if a[1] != b[0] {
                return Err(format!("Shape mismatch in MatMul: {:?} x {:?}", a, b));
            }
            Ok(ShapeFact::Tensor(vec![a[0], b[1]]))
        }
        (ShapeFact::Unknown, ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
        (ShapeFact::Unknown, ShapeFact::Tensor(_)) | (ShapeFact::Tensor(_), ShapeFact::Unknown) => {
            Ok(ShapeFact::Unknown)
        }
        (ShapeFact::NonTensor, _) | (_, ShapeFact::NonTensor) => {
            Err("MatMul expects tensor inputs".to_string())
        }
    }
}

fn infer_conv2d(
    input: ValueId,
    weight: ValueId,
    shapes: &HashMap<ValueId, ShapeFact>,
) -> Result<ShapeFact, String> {
    let input_shape = shape_of(input, shapes);
    let weight_shape = shape_of(weight, shapes);
    match (input_shape, weight_shape) {
        (ShapeFact::Tensor(input), ShapeFact::Tensor(weight)) => {
            if input.len() != 2 || weight.len() != 2 {
                return Err(format!(
                    "Conv2D expects rank-2 tensors, got {:?} and {:?}",
                    input, weight
                ));
            }
            if weight[0] == 0 || weight[1] == 0 || weight[0] > input[0] || weight[1] > input[1] {
                return Err(format!(
                    "Shape mismatch in Conv2D: input {:?}, kernel {:?}",
                    input, weight
                ));
            }
            Ok(ShapeFact::Tensor(vec![
                input[0] - weight[0] + 1,
                input[1] - weight[1] + 1,
            ]))
        }
        (ShapeFact::Unknown, ShapeFact::Unknown)
        | (ShapeFact::Unknown, ShapeFact::Tensor(_))
        | (ShapeFact::Tensor(_), ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
        (ShapeFact::NonTensor, _) | (_, ShapeFact::NonTensor) => {
            Err("Conv2D expects tensor inputs".to_string())
        }
    }
}

fn infer_pool2d_nchw(
    input: ValueId,
    kernel_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    shapes: &HashMap<ValueId, ShapeFact>,
) -> Result<ShapeFact, String> {
    match shape_of(input, shapes) {
        ShapeFact::Tensor(shape) => {
            if shape.len() != 4 {
                return Err(format!("Pool expects rank-4 NCHW input, got {:?}", shape));
            }
            if kernel_shape.len() != 2 {
                return Err("Pool kernel_shape must contain 2 values".to_string());
            }
            let kh = kernel_shape[0];
            let kw = kernel_shape[1];
            if kh == 0 || kw == 0 {
                return Err("Pool kernel values must be > 0".to_string());
            }

            let sh = strides.first().copied().unwrap_or(1);
            let sw = strides.get(1).copied().unwrap_or(1);
            if sh == 0 || sw == 0 {
                return Err("Pool strides must be > 0".to_string());
            }

            let (pt, pl, pb, pr) = if pads.is_empty() {
                (0_usize, 0_usize, 0_usize, 0_usize)
            } else if pads.len() == 4 {
                (pads[0], pads[1], pads[2], pads[3])
            } else {
                return Err("Pool pads must be empty or [top,left,bottom,right]".to_string());
            };

            let h_padded = shape[2] + pt + pb;
            let w_padded = shape[3] + pl + pr;
            if h_padded < kh || w_padded < kw {
                return Err(format!(
                    "Pool kernel {:?} too large for padded input {:?}",
                    kernel_shape,
                    [shape[0], shape[1], h_padded, w_padded]
                ));
            }

            let out_h = (h_padded - kh) / sh + 1;
            let out_w = (w_padded - kw) / sw + 1;
            Ok(ShapeFact::Tensor(vec![shape[0], shape[1], out_h, out_w]))
        }
        ShapeFact::Unknown => Ok(ShapeFact::Unknown),
        ShapeFact::NonTensor => Err("Pool expects tensor input".to_string()),
    }
}

fn infer_reshape(
    input: ValueId,
    target_shape: &[usize],
    shapes: &HashMap<ValueId, ShapeFact>,
) -> Result<ShapeFact, String> {
    if target_shape.is_empty() {
        return Err("reshape target must be non-empty".to_string());
    }
    if target_shape.contains(&0) {
        return Err("reshape target cannot contain zero dimension".to_string());
    }
    let source = shape_of(input, shapes);
    match source {
        ShapeFact::Tensor(current_shape) => {
            let src_count = element_count(&current_shape)
                .ok_or_else(|| "reshape source shape overflow".to_string())?;
            let dst_count = element_count(target_shape)
                .ok_or_else(|| "reshape target shape overflow".to_string())?;
            if src_count != dst_count {
                return Err(format!(
                    "reshape element mismatch: source {:?} ({} elements) to {:?} ({} elements)",
                    current_shape, src_count, target_shape, dst_count
                ));
            }
            Ok(ShapeFact::Tensor(target_shape.to_vec()))
        }
        ShapeFact::Unknown => Ok(ShapeFact::Tensor(target_shape.to_vec())),
        ShapeFact::NonTensor => Err("reshape expects tensor input".to_string()),
    }
}

fn infer_concat(
    inputs: &[ValueId],
    axis: usize,
    shapes: &HashMap<ValueId, ShapeFact>,
) -> Result<ShapeFact, String> {
    if inputs.len() < 2 {
        return Err("concat requires at least 2 inputs".to_string());
    }
    let mut resolved = Vec::with_capacity(inputs.len());
    for value in inputs {
        match shape_of(*value, shapes) {
            ShapeFact::Tensor(shape) => resolved.push(shape),
            ShapeFact::Unknown => return Ok(ShapeFact::Unknown),
            ShapeFact::NonTensor => return Err("concat expects tensor inputs".to_string()),
        }
    }
    let rank = resolved[0].len();
    if axis >= rank {
        return Err(format!(
            "concat axis {} out of bounds for rank {} tensor",
            axis, rank
        ));
    }

    let mut output = resolved[0].clone();
    let mut axis_sum = 0usize;
    for shape in resolved {
        if shape.len() != rank {
            return Err(format!(
                "concat rank mismatch: expected rank {}, got {:?}",
                rank, shape
            ));
        }
        for (dim, (expected, actual)) in output.iter().zip(shape.iter()).enumerate() {
            if dim != axis && expected != actual {
                return Err(format!(
                    "concat shape mismatch at dim {}: expected {}, got {}",
                    dim, expected, actual
                ));
            }
        }
        axis_sum = axis_sum
            .checked_add(shape[axis])
            .ok_or_else(|| "concat axis size overflow".to_string())?;
    }
    output[axis] = axis_sum;
    Ok(ShapeFact::Tensor(output))
}

fn infer_gather(
    input: ValueId,
    indices: &[usize],
    axis: usize,
    shapes: &HashMap<ValueId, ShapeFact>,
) -> Result<ShapeFact, String> {
    if indices.is_empty() {
        return Err("gather requires at least one index".to_string());
    }
    match shape_of(input, shapes) {
        ShapeFact::Tensor(mut shape) => {
            if axis >= shape.len() {
                return Err(format!(
                    "gather axis {} out of bounds for rank {} tensor",
                    axis,
                    shape.len()
                ));
            }
            let axis_dim = shape[axis];
            for index in indices {
                if *index >= axis_dim {
                    return Err(format!(
                        "gather index {} out of bounds for axis {} with size {}",
                        index, axis, axis_dim
                    ));
                }
            }
            shape[axis] = indices.len();
            Ok(ShapeFact::Tensor(shape))
        }
        ShapeFact::Unknown => Ok(ShapeFact::Unknown),
        ShapeFact::NonTensor => Err("gather expects tensor input".to_string()),
    }
}

fn infer_slice(
    input: ValueId,
    starts: &[usize],
    ends: &[usize],
    axes: &[usize],
    shapes: &HashMap<ValueId, ShapeFact>,
) -> Result<ShapeFact, String> {
    if starts.is_empty() || ends.is_empty() || axes.is_empty() {
        return Err("slice starts/ends/axes must be non-empty".to_string());
    }
    if starts.len() != ends.len() || starts.len() != axes.len() {
        return Err("slice starts/ends/axes lengths must match".to_string());
    }
    match shape_of(input, shapes) {
        ShapeFact::Tensor(mut shape) => {
            let rank = shape.len();
            let mut seen_axes = std::collections::HashSet::new();
            for idx in 0..axes.len() {
                let axis = axes[idx];
                if axis >= rank {
                    return Err(format!(
                        "slice axis {} out of bounds for rank {} tensor",
                        axis, rank
                    ));
                }
                if !seen_axes.insert(axis) {
                    return Err(format!("slice axis {} specified more than once", axis));
                }
                let start = starts[idx];
                let end = ends[idx];
                if start >= end {
                    return Err(format!(
                        "slice requires start < end for each axis, got {} >= {}",
                        start, end
                    ));
                }
                if end > shape[axis] {
                    return Err(format!(
                        "slice end {} out of bounds for axis {} with size {}",
                        end, axis, shape[axis]
                    ));
                }
                shape[axis] = end - start;
            }
            Ok(ShapeFact::Tensor(shape))
        }
        ShapeFact::Unknown => Ok(ShapeFact::Unknown),
        ShapeFact::NonTensor => Err("slice expects tensor input".to_string()),
    }
}

fn infer_reduce(
    input: ValueId,
    axis: Option<usize>,
    keepdims: bool,
    shapes: &HashMap<ValueId, ShapeFact>,
    op_name: &str,
) -> Result<ShapeFact, String> {
    match shape_of(input, shapes) {
        ShapeFact::Tensor(shape) => match axis {
            None => {
                if keepdims {
                    if shape.is_empty() {
                        Ok(ShapeFact::Tensor(vec![1]))
                    } else {
                        Ok(ShapeFact::Tensor(vec![1; shape.len()]))
                    }
                } else {
                    Ok(ShapeFact::Tensor(vec![1]))
                }
            }
            Some(a) => {
                if a >= shape.len() {
                    return Err(format!(
                        "{op_name} axis {} out of bounds for rank {} tensor",
                        a,
                        shape.len()
                    ));
                }
                let mut out = shape;
                if keepdims {
                    out[a] = 1;
                } else {
                    out.remove(a);
                    if out.is_empty() {
                        out.push(1);
                    }
                }
                Ok(ShapeFact::Tensor(out))
            }
        },
        ShapeFact::Unknown => Ok(ShapeFact::Unknown),
        ShapeFact::NonTensor => Err(format!("{op_name} expects tensor input")),
    }
}

fn element_count(shape: &[usize]) -> Option<usize> {
    let mut count = 1usize;
    for dim in shape {
        count = count.checked_mul(*dim)?;
    }
    Some(count)
}

fn infer_phi(
    values: &[ValueId],
    shapes: &HashMap<ValueId, ShapeFact>,
) -> Result<ShapeFact, String> {
    let mut current = ShapeFact::Unknown;
    for value in values {
        current = unify_shape(current, shape_of(*value, shapes))?;
    }
    Ok(current)
}

fn unify_shape(current: ShapeFact, next: ShapeFact) -> Result<ShapeFact, String> {
    match (current, next) {
        (ShapeFact::Unknown, _) | (_, ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
        (ShapeFact::NonTensor, ShapeFact::NonTensor) => Ok(ShapeFact::NonTensor),
        (ShapeFact::Tensor(a), ShapeFact::Tensor(b)) => {
            if a == b {
                Ok(ShapeFact::Tensor(a))
            } else {
                Err(format!("Shape mismatch in phi: {:?} vs {:?}", a, b))
            }
        }
        _ => Err("Type mismatch in phi: tensor vs non-tensor".to_string()),
    }
}

fn shape_of(value: ValueId, shapes: &HashMap<ValueId, ShapeFact>) -> ShapeFact {
    shapes.get(&value).cloned().unwrap_or(ShapeFact::Unknown)
}

pub fn broadcast_shapes(s1: &[usize], s2: &[usize]) -> Option<Vec<usize>> {
    let max_len = s1.len().max(s2.len());
    let mut out = vec![0; max_len];
    for i in 0..max_len {
        let dim1 = if i < max_len - s1.len() {
            1
        } else {
            s1[i - (max_len - s1.len())]
        };
        let dim2 = if i < max_len - s2.len() {
            1
        } else {
            s2[i - (max_len - s2.len())]
        };
        if dim1 == dim2 {
            out[i] = dim1;
        } else if dim1 == 1 {
            out[i] = dim2;
        } else if dim2 == 1 {
            out[i] = dim1;
        } else {
            return None;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, ShapeFact, graph_fingerprint, infer_shapes};

    #[test]
    fn infers_shapes_for_matmul() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2, 3],
                    data: vec![1.0; 6],
                },
            )
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![3, 4],
                    data: vec![1.0; 12],
                },
            )
            .expect("add op should succeed");
        let (_, c) = graph
            .add_op(block, Op::MatMul(a, b))
            .expect("add op should succeed");

        let shapes = infer_shapes(&graph).expect("shape inference should pass");
        assert_eq!(shapes.get(&c), Some(&ShapeFact::Tensor(vec![2, 4])));
    }

    #[test]
    fn rejects_shape_mismatch_in_matmul() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2, 2],
                    data: vec![1.0; 4],
                },
            )
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![3, 2],
                    data: vec![1.0; 6],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::MatMul(a, b))
            .expect("add op should succeed");

        let err = infer_shapes(&graph).expect_err("must fail shape inference");
        assert!(err.message.contains("Shape mismatch in MatMul"));
    }

    #[test]
    fn infers_shapes_from_bound_inputs_and_parameters() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, w) = graph
            .add_op(block, Op::Parameter("w".to_string()))
            .expect("add op should succeed");
        let (_, y) = graph
            .add_op(block, Op::MatMul(x, w))
            .expect("add op should succeed");

        graph.bind_input_shape("x", vec![1, 4]);
        graph.bind_parameter_shape("w", vec![4, 2]);

        let shapes = infer_shapes(&graph).expect("shape inference should pass");
        assert_eq!(shapes.get(&x), Some(&ShapeFact::Tensor(vec![1, 4])));
        assert_eq!(shapes.get(&w), Some(&ShapeFact::Tensor(vec![4, 2])));
        assert_eq!(shapes.get(&y), Some(&ShapeFact::Tensor(vec![1, 2])));
    }

    #[test]
    fn infer_shapes_does_not_mutate_graph() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, y) = graph
            .add_op(block, Op::Relu(x))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Output(y))
            .expect("add op should succeed");
        graph.bind_input_shape("x", vec![1, 2]);

        let before = graph_fingerprint(&graph);
        let _ = infer_shapes(&graph).expect("shape inference should pass");
        let after = graph_fingerprint(&graph);

        assert_eq!(before, after, "shape inference must not mutate IR graph");
    }
}
