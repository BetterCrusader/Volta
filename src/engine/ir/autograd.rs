use std::collections::HashMap;

use crate::ir::{Graph, Op, ShapeFact, ValueId, infer_shapes, verify_graph};

#[derive(Debug, Clone)]
pub struct AutogradError {
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct GradientGraph {
    pub forward: Graph,
    pub backward: Graph,
    pub gradients: HashMap<ValueId, ValueId>,
    pub loss_grad_input: ValueId,
}

/// Builds the reverse-mode automatic differentiation graph for the given forward graph.
///
/// # Errors
///
/// Returns `Err(AutogradError)` if:
/// - `forward` fails graph verification
/// - any gradient operation produces an incompatible shape
/// - the backward graph cannot be constructed due to unsupported ops
pub fn build_reverse_graph(
    forward: &Graph,
    loss_value: ValueId,
    parameters: &[ValueId],
) -> Result<GradientGraph, AutogradError> {
    verify_graph(forward).map_err(|err| AutogradError {
        message: format!("Forward graph failed verification: {}", err.message),
    })?;
    let shape_facts = infer_shapes(forward).map_err(|err| AutogradError {
        message: format!("Forward shape inference failed: {}", err.message),
    })?;

    let mut backward = Graph::new();
    backward.shape_signature = forward.shape_signature.clone();
    let block = backward.create_block();

    let mut forward_to_backward = HashMap::<ValueId, ValueId>::new();
    for node in &forward.nodes {
        let cloned = remap_forward_op(&node.op, &forward_to_backward)?;
        let (_, mapped_out) = backward
            .add_op(block, cloned)
            .map_err(|err| AutogradError {
                message: format!(
                    "Failed to clone forward node into backward graph: {}",
                    err.message
                ),
            })?;
        forward_to_backward.insert(node.output, mapped_out);
    }

    let mapped_loss = forward_to_backward
        .get(&loss_value)
        .copied()
        .ok_or_else(|| AutogradError {
            message: format!("Loss value {loss_value:?} is not present in forward graph"),
        })?;

    let (_, seed_grad) = backward
        .add_op(block, Op::Input("__loss_grad".to_string()))
        .map_err(|err| AutogradError {
            message: format!("Failed to add loss grad input: {}", err.message),
        })?;

    let mut grad_map: HashMap<ValueId, ValueId> = HashMap::new();
    grad_map.insert(mapped_loss, seed_grad);

    for node in forward.nodes.iter().rev() {
        let mapped_output = forward_to_backward
            .get(&node.output)
            .copied()
            .ok_or_else(|| AutogradError {
                message: format!("Missing mapped output for forward value {:?}", node.output),
            })?;

        let Some(upstream) = grad_map.get(&mapped_output).copied() else {
            continue;
        };

        match &node.op {
            Op::Plugin { operator, inputs } => {
                let upstream = grad_map
                    .get(&mapped_output)
                    .copied()
                    .ok_or_else(|| AutogradError {
                        message: format!(
                            "Plugin backward: missing upstream gradient for output {mapped_output:?}"
                        ),
                    })?;
                let backward_ops = operator.get_backward_ops(
                    inputs,
                    node.output,
                    &[upstream], // plugins currently only have one output
                );
                for (input_value, grad_op) in backward_ops {
                    let mapped_input = mapped(&forward_to_backward, input_value)?;
                    let (_, grad_contribution) =
                        backward
                            .add_op(block, grad_op)
                            .map_err(|err| AutogradError {
                                message: format!(
                                    "Failed to add backward op for plugin: {}",
                                    err.message
                                ),
                            })?;
                    accumulate_grad(
                        &mut backward,
                        block,
                        &mut grad_map,
                        mapped_input,
                        grad_contribution,
                    )?;
                }
            }
            Op::Add(a, b) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let b_m = mapped(&forward_to_backward, *b)?;
                let out_shape = tensor_shape_maybe(&shape_facts, node.output);
                let a_shape = tensor_shape_maybe(&shape_facts, *a);
                let b_shape = tensor_shape_maybe(&shape_facts, *b);

                let grad_a = if let (Some(out_shape), Some(a_shape)) = (&out_shape, &a_shape) {
                    unbroadcast_gradient(
                        &mut backward,
                        block,
                        upstream,
                        out_shape,
                        a_shape,
                        "add(lhs)",
                    )?
                } else {
                    upstream
                };
                let grad_b = if let (Some(out_shape), Some(b_shape)) = (&out_shape, &b_shape) {
                    unbroadcast_gradient(
                        &mut backward,
                        block,
                        upstream,
                        out_shape,
                        b_shape,
                        "add(rhs)",
                    )?
                } else {
                    upstream
                };

                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad_a)?;
                accumulate_grad(&mut backward, block, &mut grad_map, b_m, grad_b)?;
            }
            Op::Sub(a, b) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let b_m = mapped(&forward_to_backward, *b)?;
                let out_shape = tensor_shape_maybe(&shape_facts, node.output);
                let a_shape = tensor_shape_maybe(&shape_facts, *a);
                let b_shape = tensor_shape_maybe(&shape_facts, *b);

                let grad_a = if let (Some(out_shape), Some(a_shape)) = (&out_shape, &a_shape) {
                    unbroadcast_gradient(
                        &mut backward,
                        block,
                        upstream,
                        out_shape,
                        a_shape,
                        "sub(lhs)",
                    )?
                } else {
                    upstream
                };

                let (_, neg) =
                    backward
                        .add_op(block, Op::Neg(upstream))
                        .map_err(|err| AutogradError {
                            message: format!("Failed to build neg gradient: {}", err.message),
                        })?;

                let grad_b = if let (Some(out_shape), Some(b_shape)) = (&out_shape, &b_shape) {
                    unbroadcast_gradient(&mut backward, block, neg, out_shape, b_shape, "sub(rhs)")?
                } else {
                    neg
                };

                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad_a)?;
                accumulate_grad(&mut backward, block, &mut grad_map, b_m, grad_b)?;
            }
            Op::Mul(a, b) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let b_m = mapped(&forward_to_backward, *b)?;
                let out_shape = tensor_shape_maybe(&shape_facts, node.output);
                let a_shape = tensor_shape_maybe(&shape_facts, *a);
                let b_shape = tensor_shape_maybe(&shape_facts, *b);

                let (_, ga) = backward
                    .add_op(block, Op::Mul(upstream, b_m))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build mul grad(a): {}", err.message),
                    })?;
                let (_, gb) = backward
                    .add_op(block, Op::Mul(upstream, a_m))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build mul grad(b): {}", err.message),
                    })?;

                let grad_a = if let (Some(out_shape), Some(a_shape)) = (&out_shape, &a_shape) {
                    unbroadcast_gradient(&mut backward, block, ga, out_shape, a_shape, "mul(lhs)")?
                } else {
                    ga
                };
                let grad_b = if let (Some(out_shape), Some(b_shape)) = (&out_shape, &b_shape) {
                    unbroadcast_gradient(&mut backward, block, gb, out_shape, b_shape, "mul(rhs)")?
                } else {
                    gb
                };

                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad_a)?;
                accumulate_grad(&mut backward, block, &mut grad_map, b_m, grad_b)?;
            }
            Op::Div(a, b) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let b_m = mapped(&forward_to_backward, *b)?;
                let out_shape = tensor_shape_maybe(&shape_facts, node.output);
                let a_shape = tensor_shape_maybe(&shape_facts, *a);
                let b_shape = tensor_shape_maybe(&shape_facts, *b);

                let (_, ga) = backward
                    .add_op(block, Op::Div(upstream, b_m))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build div grad(a): {}", err.message),
                    })?;
                let (_, b2) =
                    backward
                        .add_op(block, Op::Mul(b_m, b_m))
                        .map_err(|err| AutogradError {
                            message: format!("Failed to build div b^2: {}", err.message),
                        })?;
                let (_, num) = backward
                    .add_op(block, Op::Mul(upstream, a_m))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build div numerator: {}", err.message),
                    })?;
                let (_, frac) =
                    backward
                        .add_op(block, Op::Div(num, b2))
                        .map_err(|err| AutogradError {
                            message: format!("Failed to build div fraction: {}", err.message),
                        })?;
                let (_, gb) =
                    backward
                        .add_op(block, Op::Neg(frac))
                        .map_err(|err| AutogradError {
                            message: format!("Failed to build div grad(b): {}", err.message),
                        })?;

                let grad_a = if let (Some(out_shape), Some(a_shape)) = (&out_shape, &a_shape) {
                    unbroadcast_gradient(&mut backward, block, ga, out_shape, a_shape, "div(lhs)")?
                } else {
                    ga
                };
                let grad_b = if let (Some(out_shape), Some(b_shape)) = (&out_shape, &b_shape) {
                    unbroadcast_gradient(&mut backward, block, gb, out_shape, b_shape, "div(rhs)")?
                } else {
                    gb
                };

                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad_a)?;
                accumulate_grad(&mut backward, block, &mut grad_map, b_m, grad_b)?;
            }
            Op::MatMul(a, b) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let b_m = mapped(&forward_to_backward, *b)?;
                let (_, b_t) =
                    backward
                        .add_op(block, Op::Transpose(b_m))
                        .map_err(|err| AutogradError {
                            message: format!("Failed to build transpose(b): {}", err.message),
                        })?;
                let (_, a_t) =
                    backward
                        .add_op(block, Op::Transpose(a_m))
                        .map_err(|err| AutogradError {
                            message: format!("Failed to build transpose(a): {}", err.message),
                        })?;
                let (_, ga) = backward
                    .add_op(block, Op::MatMul(upstream, b_t))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build matmul grad(a): {}", err.message),
                    })?;
                let (_, gb) = backward
                    .add_op(block, Op::MatMul(a_t, upstream))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build matmul grad(b): {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, ga)?;
                accumulate_grad(&mut backward, block, &mut grad_map, b_m, gb)?;
            }
            Op::Gemm {
                lhs,
                rhs,
                bias,
                alpha,
                beta,
            } => {
                let lhs_m = mapped(&forward_to_backward, *lhs)?;
                let rhs_m = mapped(&forward_to_backward, *rhs)?;
                let lhs_t = backward
                    .add_op(block, Op::Transpose(lhs_m))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build transpose(lhs): {}", err.message),
                    })?
                    .1;
                let rhs_t = backward
                    .add_op(block, Op::Transpose(rhs_m))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build transpose(rhs): {}", err.message),
                    })?
                    .1;
                let (_raw_grad_lhs_id, raw_grad_lhs) = backward
                    .add_op(block, Op::MatMul(upstream, rhs_t))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build gemm grad(lhs): {}", err.message),
                    })?;
                let (_raw_grad_rhs_id, raw_grad_rhs) = backward
                    .add_op(block, Op::MatMul(lhs_t, upstream))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build gemm grad(rhs): {}", err.message),
                    })?;
                let grad_lhs = if *alpha != 1.0 {
                    let alpha_const = backward
                        .add_op(
                            block,
                            Op::ConstTensor {
                                shape: vec![],
                                data: vec![*alpha],
                            },
                        )
                        .map_err(|err| AutogradError {
                            message: format!("Failed to build alpha constant: {}", err.message),
                        })?
                        .1;
                    backward
                        .add_op(block, Op::Mul(raw_grad_lhs, alpha_const))
                        .map_err(|err| AutogradError {
                            message: format!("Failed to scale grad(lhs): {}", err.message),
                        })?
                        .1
                } else {
                    raw_grad_lhs
                };
                let grad_rhs = if *alpha != 1.0 {
                    let alpha_const = backward
                        .add_op(
                            block,
                            Op::ConstTensor {
                                shape: vec![],
                                data: vec![*alpha],
                            },
                        )
                        .map_err(|err| AutogradError {
                            message: format!("Failed to build alpha constant: {}", err.message),
                        })?
                        .1;
                    backward
                        .add_op(block, Op::Mul(raw_grad_rhs, alpha_const))
                        .map_err(|err| AutogradError {
                            message: format!("Failed to scale grad(rhs): {}", err.message),
                        })?
                        .1
                } else {
                    raw_grad_rhs
                };
                accumulate_grad(&mut backward, block, &mut grad_map, lhs_m, grad_lhs)?;
                accumulate_grad(&mut backward, block, &mut grad_map, rhs_m, grad_rhs)?;
                if let Some(bias_val) = bias {
                    let bias_m = mapped(&forward_to_backward, *bias_val)?;
                    let scaled_grad = if *beta == 1.0 {
                        upstream
                    } else {
                        backward
                            .add_op(
                                block,
                                Op::ConstTensor {
                                    shape: vec![],
                                    data: vec![*beta],
                                },
                            )
                            .map_err(|err| AutogradError {
                                message: format!("Failed to build beta constant: {}", err.message),
                            })?
                            .1
                    };
                    let (_, grad_bias) = backward
                        .add_op(
                            block,
                            Op::ReduceSum {
                                input: scaled_grad,
                                axis: Some(0),
                                keepdims: false,
                            },
                        )
                        .map_err(|err| AutogradError {
                            message: format!(
                                "Failed to build gemm grad(bias) reduce: {}",
                                err.message
                            ),
                        })?;
                    accumulate_grad(&mut backward, block, &mut grad_map, bias_m, grad_bias)?;
                }
            }
            Op::Relu(a) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let (_, grad) = backward
                    .add_op(block, Op::ReluBackward(a_m, upstream))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build relu backward op: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad)?;
            }
            Op::Log(a) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let (_, grad) = backward
                    .add_op(block, Op::Div(upstream, a_m))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build log grad: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad)?;
            }
            Op::Exp(a) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let (_, grad) = backward
                    .add_op(block, Op::Mul(upstream, mapped_output))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build exp grad: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad)?;
            }
            Op::Sigmoid(a) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let (_, grad) = backward
                    .add_op(block, Op::SigmoidBackward(a_m, upstream))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build sigmoid backward op: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad)?;
            }
            Op::Gelu(a) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let (_, grad) = backward
                    .add_op(block, Op::GeluBackward(a_m, upstream))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build gelu backward op: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad)?;
            }
            Op::GeluExact(a) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let (_, grad) = backward
                    .add_op(block, Op::GeluExactBackward(a_m, upstream))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build gelu_exact backward op: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad)?;
            }
            Op::SoftmaxCrossEntropyLossFromLogits { logits, targets } => {
                // Given L = SoftmaxCrossEntropyLossFromLogits(logits, targets)
                // The gradient w.r.t logits is: (softmax(logits) - targets) / batch_size
                // (Assuming upstream grad is 1.0, otherwise we scale by upstream)
                let logits_m = mapped(&forward_to_backward, *logits)?;
                let targets_m = mapped(&forward_to_backward, *targets)?;

                // 1. Compute softmax(logits)
                let (_, sm) = backward
                    .add_op(block, Op::Softmax(logits_m))
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build softmax for cross_entropy grad: {}",
                            err.message
                        ),
                    })?;

                // 2. Subtract targets: softmax(logits) - targets
                let (_, diff) = backward
                    .add_op(block, Op::Sub(sm, targets_m))
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build diff for cross_entropy grad: {}",
                            err.message
                        ),
                    })?;

                // 3. Divide by batch_size. Shape of logits is [batch_size, num_classes]
                let logits_shape = tensor_shape_for(&shape_facts, *logits)?;
                let batch_size = logits_shape[0] as f32;

                let bs_const = backward
                    .add_op(
                        block,
                        Op::ConstTensor {
                            shape: vec![],
                            data: vec![batch_size],
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build batch_size const: {}", err.message),
                    })?
                    .1;

                let (_, grad_unscaled) =
                    backward
                        .add_op(block, Op::Div(diff, bs_const))
                        .map_err(|err| AutogradError {
                            message: format!(
                                "Failed to build div for cross_entropy grad: {}",
                                err.message
                            ),
                        })?;

                // 4. Multiply by upstream gradient (usually 1.0 for loss)
                let (_, grad) = backward
                    .add_op(block, Op::Mul(grad_unscaled, upstream))
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build mul upstream for cross_entropy grad: {}",
                            err.message
                        ),
                    })?;

                accumulate_grad(&mut backward, block, &mut grad_map, logits_m, grad)?;

                // Targets don't need gradients here, but if they did, we would accumulate 0
            }
            Op::Reshape { input, .. } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let input_shape = tensor_shape_for(&shape_facts, *input)?;
                let (_, grad) = backward
                    .add_op(
                        block,
                        Op::Reshape {
                            input: upstream,
                            shape: input_shape,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build reshape backward op: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad)?;
            }
            Op::Flatten { input, .. } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let input_shape = tensor_shape_for(&shape_facts, *input)?;
                let (_, grad) = backward
                    .add_op(
                        block,
                        Op::Reshape {
                            input: upstream,
                            shape: input_shape,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build flatten backward reshape: {}",
                            err.message
                        ),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad)?;
            }
            Op::GlobalAveragePool { input } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let (_, grad) = backward
                    .add_op(
                        block,
                        Op::GlobalAveragePoolBackward {
                            input: input_m,
                            upstream,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build GlobalAveragePool backward: {}",
                            err.message
                        ),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad)?;
            }
            Op::GlobalAveragePoolBackward { .. } => {
                // backward of backward not implemented
            }
            Op::GroupNorm {
                input,
                weight,
                bias,
                num_groups,
                epsilon,
            } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let weight_m = mapped(&forward_to_backward, *weight)?;
                let (_, grad_input) = backward
                    .add_op(
                        block,
                        Op::GroupNormBackwardInput {
                            input: input_m,
                            upstream,
                            weight: weight_m,
                            num_groups: *num_groups,
                            epsilon: *epsilon,
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("GroupNorm bwd input: {}", e.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad_input)?;
                let (_, grad_weight) = backward
                    .add_op(
                        block,
                        Op::GroupNormBackwardWeight {
                            input: input_m,
                            upstream,
                            num_groups: *num_groups,
                            epsilon: *epsilon,
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("GroupNorm bwd weight: {}", e.message),
                    })?;
                let weight_m_param = mapped(&forward_to_backward, *weight)?;
                accumulate_grad(
                    &mut backward,
                    block,
                    &mut grad_map,
                    weight_m_param,
                    grad_weight,
                )?;
                let (_, grad_bias) = backward
                    .add_op(block, Op::GroupNormBackwardBias { upstream })
                    .map_err(|e| AutogradError {
                        message: format!("GroupNorm bwd bias: {}", e.message),
                    })?;
                let bias_m = mapped(&forward_to_backward, *bias)?;
                accumulate_grad(&mut backward, block, &mut grad_map, bias_m, grad_bias)?;
            }
            Op::GroupNormBackwardInput { .. }
            | Op::GroupNormBackwardWeight { .. }
            | Op::GroupNormBackwardBias { .. } => {}
            Op::InstanceNorm {
                input,
                weight,
                bias,
                epsilon,
            } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let weight_m = mapped(&forward_to_backward, *weight)?;
                let (_, grad_input) = backward
                    .add_op(
                        block,
                        Op::InstanceNormBackwardInput {
                            input: input_m,
                            upstream,
                            weight: weight_m,
                            epsilon: *epsilon,
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("InstanceNorm bwd input: {}", e.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad_input)?;
                let (_, grad_weight) = backward
                    .add_op(
                        block,
                        Op::InstanceNormBackwardWeight {
                            input: input_m,
                            upstream,
                            epsilon: *epsilon,
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("InstanceNorm bwd weight: {}", e.message),
                    })?;
                let weight_m_param = mapped(&forward_to_backward, *weight)?;
                accumulate_grad(
                    &mut backward,
                    block,
                    &mut grad_map,
                    weight_m_param,
                    grad_weight,
                )?;
                let (_, grad_bias) = backward
                    .add_op(block, Op::InstanceNormBackwardBias { upstream })
                    .map_err(|e| AutogradError {
                        message: format!("InstanceNorm bwd bias: {}", e.message),
                    })?;
                let bias_m = mapped(&forward_to_backward, *bias)?;
                accumulate_grad(&mut backward, block, &mut grad_map, bias_m, grad_bias)?;
            }
            Op::InstanceNormBackwardInput { .. }
            | Op::InstanceNormBackwardWeight { .. }
            | Op::InstanceNormBackwardBias { .. } => {}
            Op::ConvTranspose2D { input, .. } => {
                // Gradient of ConvTranspose2D: pass upstream through to input for now.
                // Full impl requires conv2d backward; weight grad not yet supported.
                let input_m = mapped(&forward_to_backward, *input)?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, upstream)?;
            }
            Op::Upsample2D {
                input,
                scale_h,
                scale_w,
                mode,
            } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                if *mode == 0 {
                    let sh = scale_h.round() as usize;
                    let sw = scale_w.round() as usize;
                    // We need orig_h and orig_w — they come from the input shape
                    // For now we emit UpsampleBackward with placeholder zeros (shape from verifier)
                    // In practice, caller must set correct orig_h/orig_w from shape facts
                    let (_, grad) = backward
                        .add_op(
                            block,
                            Op::Upsample2DBackward {
                                upstream,
                                orig_h: 0, // will be resolved at runtime from input shape
                                orig_w: 0,
                                scale_h: sh,
                                scale_w: sw,
                            },
                        )
                        .map_err(|e| AutogradError {
                            message: format!("Upsample2D bwd: {}", e.message),
                        })?;
                    accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad)?;
                } else {
                    // Bilinear backward: approximate as pass-through (TODO: proper backward)
                    accumulate_grad(&mut backward, block, &mut grad_map, input_m, upstream)?;
                }
            }
            Op::Upsample2DBackward { .. } => {}
            Op::MultiHeadAttention {
                q_input,
                k_input,
                v_input,
                w_q,
                w_k,
                w_v,
                w_o,
                num_heads,
                output_idx,
                ..
            } => {
                // Only emit backward compute for the primary output (output_idx == 0).
                // The other five outputs (attn_weights, q_proj, k_proj, v_proj, context)
                // are intermediate activations — no gradient flows from loss through them directly.
                if *output_idx != 0 {
                    continue;
                }

                // Find the sibling forward nodes for attn_weights (output_idx=1) and context (output_idx=5).
                // They share the same q_input.
                let attn_weights_fwd = forward.nodes.iter()
                    .find(|n| matches!(&n.op, Op::MultiHeadAttention { q_input: qi, output_idx: 1, .. } if *qi == *q_input))
                    .ok_or_else(|| AutogradError {
                        message: "MHA backward: cannot find attn_weights sibling (output_idx=1)".to_string(),
                    })?;
                let context_fwd = forward.nodes.iter()
                    .find(|n| matches!(&n.op, Op::MultiHeadAttention { q_input: qi, output_idx: 5, .. } if *qi == *q_input))
                    .ok_or_else(|| AutogradError {
                        message: "MHA backward: cannot find context sibling (output_idx=5)".to_string(),
                    })?;

                // Map all forward ValueIds to backward graph.
                let q_m = mapped(&forward_to_backward, *q_input)?;
                let k_m = mapped(&forward_to_backward, *k_input)?;
                let v_m = mapped(&forward_to_backward, *v_input)?;
                let wq_m = mapped(&forward_to_backward, *w_q)?;
                let wk_m = mapped(&forward_to_backward, *w_k)?;
                let wv_m = mapped(&forward_to_backward, *w_v)?;
                let wo_m = mapped(&forward_to_backward, *w_o)?;
                let aw_m = mapped(&forward_to_backward, attn_weights_fwd.output)?;
                let ctx_m = mapped(&forward_to_backward, context_fwd.output)?;

                // Emit 7 MultiHeadAttentionBackward nodes, one per gradient target.
                for grad_output_idx in 0..7_usize {
                    let (_, grad_value) = backward
                        .add_op(block, Op::MultiHeadAttentionBackward {
                            q_input: q_m,
                            k_input: k_m,
                            v_input: v_m,
                            w_q: wq_m,
                            w_k: wk_m,
                            w_v: wv_m,
                            w_o: wo_m,
                            attn_weights: aw_m,
                            context: ctx_m,
                            upstream,
                            num_heads: *num_heads,
                            output_idx: grad_output_idx,
                        })
                        .map_err(|e| AutogradError {
                            message: format!("MHA backward op: {}", e.message),
                        })?;

                    // Map gradient to the correct forward input.
                    let target = match grad_output_idx {
                        0 => q_m,
                        1 => k_m,
                        2 => v_m,
                        3 => wq_m,
                        4 => wk_m,
                        5 => wv_m,
                        6 => wo_m,
                        _ => unreachable!(),
                    };
                    accumulate_grad(&mut backward, block, &mut grad_map, target, grad_value)?;
                }
            }
            Op::SinusoidalPE { input } => {
                // PE is additive and constant: grad passes through as identity
                let input_m = mapped(&forward_to_backward, *input)?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, upstream)?;
            }
            Op::RoPE { input, offset } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let (_, grad) = backward
                    .add_op(
                        block,
                        Op::RoPEBackward {
                            upstream,
                            offset: *offset,
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("RoPE backward: {}", e.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad)?;
            }
            Op::RoPEBackward { .. } => {}
            Op::Embedding { weight, indices } => {
                // Gradient flows to weight (sparse), not to indices (discrete)
                let weight_m = mapped(&forward_to_backward, *weight)?;
                let indices_m = mapped(&forward_to_backward, *indices)?;
                let (_, grad_weight) = backward
                    .add_op(
                        block,
                        Op::EmbeddingBackward {
                            weight: weight_m,
                            indices: indices_m,
                            upstream,
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("Embedding backward: {}", e.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, weight_m, grad_weight)?;
                // No gradient for indices (they're discrete tokens)
            }
            Op::EmbeddingBackward { .. } => {}
            Op::LstmCell {
                x,
                h_prev,
                c_prev,
                weight_ih,
                weight_hh,
                bias,
                output_idx: _,
            } => {
                // For autograd we only support output_idx=0 (h_next) as the primary output.
                // The saved tensors (gates_raw, tanh_c_next, c_next) need to be computed first.
                // We emit LstmCellBackward ops for each trainable param.
                // First get the saved tensors from the forward graph.
                let x_m = mapped(&forward_to_backward, *x)?;
                let h_m = mapped(&forward_to_backward, *h_prev)?;
                let c_m = mapped(&forward_to_backward, *c_prev)?;
                let wih_m = mapped(&forward_to_backward, *weight_ih)?;
                let whh_m = mapped(&forward_to_backward, *weight_hh)?;
                let bias_m = mapped(&forward_to_backward, *bias)?;
                // Re-run forward to get gates_raw (output_idx=2) and tanh_c_next (output_idx=3) for backward.
                // Note: We must emit these as separate nodes in the backward graph.
                let (_, gates_node) = backward
                    .add_op(
                        block,
                        Op::LstmCell {
                            x: x_m,
                            h_prev: h_m,
                            c_prev: c_m,
                            weight_ih: wih_m,
                            weight_hh: whh_m,
                            bias: bias_m,
                            output_idx: 2,
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("LSTM bwd gates: {}", e.message),
                    })?;
                let (_, tanh_c_node) = backward
                    .add_op(
                        block,
                        Op::LstmCell {
                            x: x_m,
                            h_prev: h_m,
                            c_prev: c_m,
                            weight_ih: wih_m,
                            weight_hh: whh_m,
                            bias: bias_m,
                            output_idx: 3,
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("LSTM bwd tanh_c: {}", e.message),
                    })?;
                // Zero dc_next (we assume single-step LSTM, no cell grad from future)
                let (_, zero_dc) = backward
                    .add_op(
                        block,
                        Op::LstmCell {
                            x: x_m,
                            h_prev: h_m,
                            c_prev: c_m,
                            weight_ih: wih_m,
                            weight_hh: whh_m,
                            bias: bias_m,
                            output_idx: 1, // c_next shape for zeros reference
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("LSTM bwd c_next shape: {}", e.message),
                    })?;
                // We'll use the c_next value as dc_next (approximation: zeros via Mul by zero const)
                // Actually just use upstream as dh_next and c_prev zeros as dc_next.
                // Proper approach: use the actual c_next grad. For single-step, dc_next = 0.
                let (_, dc_zero) =
                    backward
                        .add_op(block, Op::Mul(zero_dc, zero_dc))
                        .map_err(|e| AutogradError {
                            message: format!("LSTM dc zero: {}", e.message),
                        })?;
                // Emit backward ops for each target
                for (target, param_m) in [
                    (0usize, x_m),
                    (1, h_m),
                    (2, c_m),
                    (3, wih_m),
                    (4, whh_m),
                    (5, bias_m),
                ] {
                    let (_, grad) = backward
                        .add_op(
                            block,
                            Op::LstmCellBackward {
                                x: x_m,
                                h_prev: h_m,
                                c_prev: c_m,
                                weight_ih: wih_m,
                                weight_hh: whh_m,
                                gates_raw: gates_node,
                                tanh_c_next: tanh_c_node,
                                dh_next: upstream,
                                dc_next: dc_zero,
                                grad_target: target,
                            },
                        )
                        .map_err(|e| AutogradError {
                            message: format!("LstmCellBackward: {}", e.message),
                        })?;
                    accumulate_grad(&mut backward, block, &mut grad_map, param_m, grad)?;
                }
            }
            Op::LstmCellBackward { .. } => {}
            Op::GruCell {
                x,
                h_prev,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                ..
            } => {
                let x_m = mapped(&forward_to_backward, *x)?;
                let h_m = mapped(&forward_to_backward, *h_prev)?;
                let wih_m = mapped(&forward_to_backward, *weight_ih)?;
                let whh_m = mapped(&forward_to_backward, *weight_hh)?;
                let bih_m = mapped(&forward_to_backward, *bias_ih)?;
                let bhh_m = mapped(&forward_to_backward, *bias_hh)?;
                // Re-compute saved gate tensors for backward
                let (_, z_node) = backward
                    .add_op(
                        block,
                        Op::GruCell {
                            x: x_m,
                            h_prev: h_m,
                            weight_ih: wih_m,
                            weight_hh: whh_m,
                            bias_ih: bih_m,
                            bias_hh: bhh_m,
                            output_idx: 1,
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("GRU bwd z: {}", e.message),
                    })?;
                let (_, r_node) = backward
                    .add_op(
                        block,
                        Op::GruCell {
                            x: x_m,
                            h_prev: h_m,
                            weight_ih: wih_m,
                            weight_hh: whh_m,
                            bias_ih: bih_m,
                            bias_hh: bhh_m,
                            output_idx: 2,
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("GRU bwd r: {}", e.message),
                    })?;
                let (_, n_node) = backward
                    .add_op(
                        block,
                        Op::GruCell {
                            x: x_m,
                            h_prev: h_m,
                            weight_ih: wih_m,
                            weight_hh: whh_m,
                            bias_ih: bih_m,
                            bias_hh: bhh_m,
                            output_idx: 3,
                        },
                    )
                    .map_err(|e| AutogradError {
                        message: format!("GRU bwd n: {}", e.message),
                    })?;
                let params = [
                    (0usize, x_m),
                    (1, h_m),
                    (2, wih_m),
                    (3, whh_m),
                    (4, bih_m),
                    (5, bhh_m),
                ];
                for (target, param_m) in params {
                    let (_, grad) = backward
                        .add_op(
                            block,
                            Op::GruCellBackward {
                                x: x_m,
                                h_prev: h_m,
                                weight_ih: wih_m,
                                weight_hh: whh_m,
                                z_gate: z_node,
                                r_gate: r_node,
                                n_gate: n_node,
                                dh_next: upstream,
                                grad_target: target,
                            },
                        )
                        .map_err(|e| AutogradError {
                            message: format!("GruCellBackward: {}", e.message),
                        })?;
                    accumulate_grad(&mut backward, block, &mut grad_map, param_m, grad)?;
                }
            }
            Op::GruCellBackward { .. } => {}
            Op::Dropout { input, .. } | Op::Identity(input) => {
                let input_m = mapped(&forward_to_backward, *input)?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, upstream)?;
            }
            Op::MaxPool {
                input,
                kernel_shape,
                strides,
                pads,
            } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let (_, grad) = backward
                    .add_op(
                        block,
                        Op::MaxPoolBackward {
                            input: input_m,
                            upstream,
                            kernel_shape: kernel_shape.clone(),
                            strides: strides.clone(),
                            pads: pads.clone(),
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build maxpool backward op: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad)?;
            }
            Op::AvgPool {
                input,
                kernel_shape,
                strides,
                pads,
            } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let (_, grad) = backward
                    .add_op(
                        block,
                        Op::AvgPoolBackward {
                            input: input_m,
                            upstream,
                            kernel_shape: kernel_shape.clone(),
                            strides: strides.clone(),
                            pads: pads.clone(),
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build avgpool backward op: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad)?;
            }
            Op::BatchNorm {
                input,
                weight,
                bias,
                mean,
                var,
            } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let weight_m = mapped(&forward_to_backward, *weight)?;
                let bias_m = mapped(&forward_to_backward, *bias)?;
                let mean_m = mapped(&forward_to_backward, *mean)?;
                let var_m = mapped(&forward_to_backward, *var)?;

                let (_, grad_x) = backward
                    .add_op(
                        block,
                        Op::BatchNormBackwardInput {
                            input: input_m,
                            upstream,
                            weight: weight_m,
                            var: var_m,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build batchnorm input backward op: {}",
                            err.message
                        ),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad_x)?;

                let (_, grad_w) = backward
                    .add_op(
                        block,
                        Op::BatchNormBackwardWeight {
                            input: input_m,
                            upstream,
                            mean: mean_m,
                            var: var_m,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build batchnorm weight backward op: {}",
                            err.message
                        ),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, weight_m, grad_w)?;

                let (_, grad_b) = backward
                    .add_op(block, Op::BatchNormBackwardBias { upstream })
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build batchnorm bias backward op: {}",
                            err.message
                        ),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, bias_m, grad_b)?;
            }
            Op::LayerNorm {
                input,
                weight,
                bias,
                epsilon,
            } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let weight_m = mapped(&forward_to_backward, *weight)?;
                let bias_m = mapped(&forward_to_backward, *bias)?;

                let (_, grad_x) = backward
                    .add_op(
                        block,
                        Op::LayerNormBackwardInput {
                            input: input_m,
                            upstream,
                            weight: weight_m,
                            epsilon: *epsilon,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build layernorm input backward op: {}",
                            err.message
                        ),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad_x)?;

                let (_, grad_w) = backward
                    .add_op(
                        block,
                        Op::LayerNormBackwardWeight {
                            input: input_m,
                            upstream,
                            epsilon: *epsilon,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build layernorm weight backward op: {}",
                            err.message
                        ),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, weight_m, grad_w)?;

                let (_, grad_b) = backward
                    .add_op(block, Op::LayerNormBackwardBias { upstream })
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build layernorm bias backward op: {}",
                            err.message
                        ),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, bias_m, grad_b)?;
            }
            Op::Concat { inputs, axis } => {
                let mut offset = 0usize;
                for input in inputs {
                    let input_shape = tensor_shape_for(&shape_facts, *input)?;
                    let width = *input_shape.get(*axis).ok_or_else(|| AutogradError {
                        message: format!(
                            "Concat axis {axis} out of bounds for input shape {input_shape:?}"
                        ),
                    })?;
                    let (_, grad_chunk) = backward
                        .add_op(
                            block,
                            Op::Slice {
                                input: upstream,
                                starts: vec![offset],
                                ends: vec![offset + width],
                                axes: vec![*axis],
                            },
                        )
                        .map_err(|err| AutogradError {
                            message: format!(
                                "Failed to build concat backward slice: {}",
                                err.message
                            ),
                        })?;
                    let input_m = mapped(&forward_to_backward, *input)?;
                    accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad_chunk)?;
                    offset = offset.checked_add(width).ok_or_else(|| AutogradError {
                        message: "Concat backward offset overflow".to_string(),
                    })?;
                }
            }
            Op::Gather {
                input,
                indices,
                axis,
            } => {
                let input_shape = tensor_shape_for(&shape_facts, *input)?;
                let axis_dim = *input_shape.get(*axis).ok_or_else(|| AutogradError {
                    message: format!(
                        "Gather axis {axis} out of bounds for input shape {input_shape:?}"
                    ),
                })?;

                // ── Scatter-sum backward ─────────────────────────────────────
                // For each input slot, sum the upstream gradient slices from
                // *every* position where `indices[pos] == slot`.  This handles
                // duplicate indices by accumulation (scatter-add semantics)
                // — the old uniqueness restriction is gone.
                let mut chunks = Vec::with_capacity(axis_dim);
                for slot in 0..axis_dim {
                    let positions: Vec<usize> = indices
                        .iter()
                        .enumerate()
                        .filter_map(|(pos, &idx)| if idx == slot { Some(pos) } else { None })
                        .collect();

                    let value = if positions.is_empty() {
                        let mut zero_shape = input_shape.clone();
                        zero_shape[*axis] = 1;
                        const_zeros(&mut backward, block, &zero_shape)?
                    } else {
                        // First occurrence.
                        let mut acc = {
                            let pos = positions[0];
                            let (_, slice) = backward
                                .add_op(
                                    block,
                                    Op::Slice {
                                        input: upstream,
                                        starts: vec![pos],
                                        ends: vec![pos + 1],
                                        axes: vec![*axis],
                                    },
                                )
                                .map_err(|err| AutogradError {
                                    message: format!(
                                        "Failed gather backward slice pos {}: {}",
                                        pos, err.message
                                    ),
                                })?;
                            slice
                        };
                        // Accumulate duplicate occurrences.
                        for &pos in &positions[1..] {
                            let (_, extra) = backward
                                .add_op(
                                    block,
                                    Op::Slice {
                                        input: upstream,
                                        starts: vec![pos],
                                        ends: vec![pos + 1],
                                        axes: vec![*axis],
                                    },
                                )
                                .map_err(|err| AutogradError {
                                    message: format!(
                                        "Failed gather backward dup slice pos {}: {}",
                                        pos, err.message
                                    ),
                                })?;
                            let (_, summed) =
                                backward.add_op(block, Op::Add(acc, extra)).map_err(|err| {
                                    AutogradError {
                                        message: format!(
                                            "Failed gather dup grad accumulate: {}",
                                            err.message
                                        ),
                                    }
                                })?;
                            acc = summed;
                        }
                        acc
                    };
                    chunks.push(value);
                }
                let (_, gathered_back) = backward
                    .add_op(
                        block,
                        Op::Concat {
                            inputs: chunks,
                            axis: *axis,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!("Failed gather backward concat: {}", err.message),
                    })?;
                let input_m = mapped(&forward_to_backward, *input)?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, gathered_back)?;
            }
            Op::Slice {
                input,
                starts,
                ends,
                axes,
            } => {
                let input_shape = tensor_shape_for(&shape_facts, *input)?;
                let mut current = tensor_shape_for(&shape_facts, node.output)?;
                let mut padded = upstream;

                for idx in 0..axes.len() {
                    let axis = axes[idx];
                    let start = starts[idx];
                    let end = ends[idx];
                    let input_dim = *input_shape.get(axis).ok_or_else(|| AutogradError {
                        message: format!(
                            "Slice axis {axis} out of bounds for input shape {input_shape:?}"
                        ),
                    })?;

                    if start > input_dim || end > input_dim || start >= end {
                        return Err(AutogradError {
                            message: format!(
                                "Invalid slice spec for backward at axis {axis}: start={start} end={end} dim={input_dim}"
                            ),
                        });
                    }

                    let prefix = start;
                    let suffix = input_dim - end;

                    if prefix > 0 {
                        let mut zero_shape = current.clone();
                        zero_shape[axis] = prefix;
                        let prefix_zero = const_zeros(&mut backward, block, &zero_shape)?;
                        let (_, concat) = backward
                            .add_op(
                                block,
                                Op::Concat {
                                    inputs: vec![prefix_zero, padded],
                                    axis,
                                },
                            )
                            .map_err(|err| AutogradError {
                                message: format!(
                                    "Failed to build slice backward prefix concat: {}",
                                    err.message
                                ),
                            })?;
                        padded = concat;
                        current[axis] += prefix;
                    }

                    if suffix > 0 {
                        let mut zero_shape = current.clone();
                        zero_shape[axis] = suffix;
                        let suffix_zero = const_zeros(&mut backward, block, &zero_shape)?;
                        let (_, concat) = backward
                            .add_op(
                                block,
                                Op::Concat {
                                    inputs: vec![padded, suffix_zero],
                                    axis,
                                },
                            )
                            .map_err(|err| AutogradError {
                                message: format!(
                                    "Failed to build slice backward suffix concat: {}",
                                    err.message
                                ),
                            })?;
                        padded = concat;
                        current[axis] += suffix;
                    }
                }

                let input_m = mapped(&forward_to_backward, *input)?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, padded)?;
            }
            Op::Output(v) => {
                let v_m = mapped(&forward_to_backward, *v)?;
                accumulate_grad(&mut backward, block, &mut grad_map, v_m, upstream)?;
            }
            // ── Neg backward: d/dx(-x) = -1, so grad_x = -upstream ────────────
            Op::Neg(a) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let (_, neg_grad) =
                    backward
                        .add_op(block, Op::Neg(upstream))
                        .map_err(|err| AutogradError {
                            message: format!("Failed to build neg backward: {}", err.message),
                        })?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, neg_grad)?;
            }
            // ── ReduceSum backward ─────────────────────────────────────────────
            // grad_input[i] = upstream_scalar for every i  (d(sum)/dx_i = 1).
            //
            // `Op::Reshape` would fail here because the upstream is shape [1]
            // and the input may have N > 1 elements.  Instead we build a static
            // ones-tensor matching input_shape and compute Mul(ones, upstream),
            // which fills every output position with the upstream scalar.
            Op::ReduceSum {
                input,
                axis,
                keepdims,
            } => {
                let input_shape = tensor_shape_for(&shape_facts, *input)?;
                let n_elems = input_shape.iter().product::<usize>();
                let input_m = mapped(&forward_to_backward, *input)?;

                let upstream_aligned = match axis {
                    None => upstream,
                    Some(a) if !*keepdims => {
                        let mut reshape_shape = input_shape.clone();
                        let axis_idx = *a;
                        if axis_idx >= reshape_shape.len() {
                            return Err(AutogradError {
                                message: format!(
                                    "reduce_sum backward axis {} out of bounds for shape {:?}",
                                    axis_idx, input_shape
                                ),
                            });
                        }
                        reshape_shape[axis_idx] = 1;
                        let (_, aligned) = backward
                            .add_op(
                                block,
                                Op::Reshape {
                                    input: upstream,
                                    shape: reshape_shape,
                                },
                            )
                            .map_err(|err| AutogradError {
                                message: format!(
                                    "reduce_sum backward reshape upstream: {}",
                                    err.message
                                ),
                            })?;
                        aligned
                    }
                    Some(_) => upstream,
                };

                let ones_data = vec![1.0_f32; n_elems];
                let (_, ones) = backward
                    .add_op(
                        block,
                        Op::ConstTensor {
                            data: ones_data,
                            shape: input_shape,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!("reduce_sum backward ones: {}", err.message),
                    })?;

                let (_, broadcast_grad) = backward
                    .add_op(block, Op::Mul(ones, upstream_aligned))
                    .map_err(|err| AutogradError {
                    message: format!("reduce_sum backward mul: {}", err.message),
                })?;

                accumulate_grad(&mut backward, block, &mut grad_map, input_m, broadcast_grad)?;
            }
            Op::ReduceMax {
                input,
                axis,
                keepdims,
            } => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let output_max_m = mapped(&forward_to_backward, node.output)?;
                let (_, grad_x) = backward
                    .add_op(
                        block,
                        Op::ReduceMaxBackward {
                            input: input_m,
                            output_max: output_max_m,
                            upstream,
                            axis: *axis,
                            keepdims: *keepdims,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build reduce_max backward op: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad_x)?;
            }
            Op::ReduceMean {
                input,
                axis,
                keepdims,
            } => {
                let input_shape = tensor_shape_for(&shape_facts, *input)?;
                let n_elems = input_shape.iter().product::<usize>();
                let input_m = mapped(&forward_to_backward, *input)?;

                let upstream_aligned = match axis {
                    None => upstream,
                    Some(a) if !*keepdims => {
                        let mut reshape_shape = input_shape.clone();
                        let axis_idx = *a;
                        if axis_idx >= reshape_shape.len() {
                            return Err(AutogradError {
                                message: format!(
                                    "reduce_mean backward axis {} out of bounds for shape {:?}",
                                    axis_idx, input_shape
                                ),
                            });
                        }
                        reshape_shape[axis_idx] = 1;
                        let (_, aligned) = backward
                            .add_op(
                                block,
                                Op::Reshape {
                                    input: upstream,
                                    shape: reshape_shape,
                                },
                            )
                            .map_err(|err| AutogradError {
                                message: format!(
                                    "reduce_mean backward reshape upstream: {}",
                                    err.message
                                ),
                            })?;
                        aligned
                    }
                    Some(_) => upstream,
                };

                let ones_data = vec![1.0_f32; n_elems];
                let (_, ones) = backward
                    .add_op(
                        block,
                        Op::ConstTensor {
                            data: ones_data,
                            shape: input_shape.clone(),
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!("reduce_mean backward ones: {}", err.message),
                    })?;

                let (_, broadcast_grad) = backward
                    .add_op(block, Op::Mul(ones, upstream_aligned))
                    .map_err(|err| AutogradError {
                    message: format!("reduce_mean backward mul: {}", err.message),
                })?;

                let divisor = match axis {
                    None => n_elems,
                    Some(a) => *input_shape.get(*a).ok_or_else(|| AutogradError {
                        message: format!(
                            "reduce_mean backward axis {} out of bounds for shape {:?}",
                            a, input_shape
                        ),
                    })?,
                };
                if divisor == 0 {
                    return Err(AutogradError {
                        message: "reduce_mean backward divisor is zero".to_string(),
                    });
                }

                let (_, inv_scale) = backward
                    .add_op(
                        block,
                        Op::ConstTensor {
                            shape: vec![1],
                            data: vec![1.0_f32 / divisor as f32],
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!("reduce_mean backward scale const: {}", err.message),
                    })?;

                let (_, grad_input) = backward
                    .add_op(block, Op::Mul(broadcast_grad, inv_scale))
                    .map_err(|err| AutogradError {
                        message: format!("reduce_mean backward scale mul: {}", err.message),
                    })?;

                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad_input)?;
            }
            // ── Softmax backward ──────────────────────────────────────────────
            // For s = softmax(x), dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
            // In IR ops:
            //   dot = ReduceSum(s * upstream)
            //   grad_x = s * (upstream - broadcast(dot))
            Op::Softmax(input) => {
                let input_shape = tensor_shape_for(&shape_facts, *input)?;
                let n_elems = input_shape.iter().product::<usize>();
                let input_m = mapped(&forward_to_backward, *input)?;
                let softmax_m = mapped(&forward_to_backward, node.output)?;

                // s * upstream
                let (_, s_times_g) = backward
                    .add_op(block, Op::Mul(softmax_m, upstream))
                    .map_err(|err| AutogradError {
                        message: format!("softmax backward s*g: {}", err.message),
                    })?;

                // dot = sum(s * upstream) => scalar shaped [1]
                let (_, dot) = backward
                    .add_op(
                        block,
                        Op::ReduceSum {
                            input: s_times_g,
                            axis: None,
                            keepdims: false,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!("softmax backward reducesum: {}", err.message),
                    })?;

                // broadcast dot to full shape: ones * dot
                let ones_data = vec![1.0_f32; n_elems];
                let (_, ones) = backward
                    .add_op(
                        block,
                        Op::ConstTensor {
                            data: ones_data,
                            shape: input_shape,
                        },
                    )
                    .map_err(|err| AutogradError {
                        message: format!("softmax backward ones: {}", err.message),
                    })?;
                let (_, dot_broadcast) =
                    backward
                        .add_op(block, Op::Mul(ones, dot))
                        .map_err(|err| AutogradError {
                            message: format!("softmax backward dot broadcast: {}", err.message),
                        })?;

                // upstream - dot_broadcast
                let (_, g_minus_dot) = backward
                    .add_op(block, Op::Sub(upstream, dot_broadcast))
                    .map_err(|err| AutogradError {
                        message: format!("softmax backward sub: {}", err.message),
                    })?;

                // s * (upstream - dot_broadcast)
                let (_, grad_x) = backward
                    .add_op(block, Op::Mul(softmax_m, g_minus_dot))
                    .map_err(|err| AutogradError {
                        message: format!("softmax backward grad_x: {}", err.message),
                    })?;

                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad_x)?;
            }
            Op::Transpose(input) => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let (_, grad_x) =
                    backward
                        .add_op(block, Op::Transpose(upstream))
                        .map_err(|err| AutogradError {
                            message: format!("transpose backward: {}", err.message),
                        })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad_x)?;
            }
            Op::ConstInt(_)
            | Op::ConstFloat(_)
            | Op::ConstTensor { .. }
            | Op::ElementwiseChain { .. }
            | Op::ReluBackward(_, _)
            | Op::SigmoidBackward(_, _)
            | Op::GeluBackward(_, _)
            | Op::GeluExactBackward(_, _)
            | Op::ReduceMaxBackward { .. }
            | Op::GemmBackward { .. }
            | Op::Conv2DBackwardInput(_, _, _)
            | Op::Conv2DBackwardWeight(_, _, _)
            | Op::MaxPoolBackward { .. }
            | Op::AvgPoolBackward { .. }
            | Op::BatchNormBackwardInput { .. }
            | Op::BatchNormBackwardWeight { .. }
            | Op::BatchNormBackwardBias { .. }
            | Op::LayerNormBackwardInput { .. }
            | Op::LayerNormBackwardWeight { .. }
            | Op::LayerNormBackwardBias { .. }
            | Op::Parameter(_)
            | Op::Input(_)
            | Op::Phi(_) => {}
            Op::Conv2D(input, weight) => {
                let input_m = mapped(&forward_to_backward, *input)?;
                let weight_m = mapped(&forward_to_backward, *weight)?;

                // grad w.r.t input
                let (_, grad_input) = backward
                    .add_op(block, Op::Conv2DBackwardInput(input_m, weight_m, upstream))
                    .map_err(|err| AutogradError {
                        message: format!(
                            "Failed to build Conv2D backward input op: {}",
                            err.message
                        ),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, input_m, grad_input)?;

                // grad w.r.t weight
                let (_, grad_weight) = backward
                    .add_op(block, Op::Conv2DBackwardWeight(input_m, weight_m, upstream))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build Conv2DBackwardWeight: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, weight_m, grad_weight)?;
            }
            Op::Removed => {}
            Op::CustomCall { .. } => {
                // CustomCall has no defined backward pass; gradients are not propagated.
            }
            Op::DepthwiseSeparableConv { .. } => {
                // DepthwiseSeparableConv backward not yet implemented; gradients are not propagated.
            }
            Op::QuantizeLinear { input, .. } => {
                // Straight-through estimator: pass gradient through quantization unchanged.
                let upstream = grad_map
                    .get(&mapped_output)
                    .copied()
                    .ok_or_else(|| AutogradError {
                        message: format!(
                            "QuantizeLinear backward: missing upstream gradient for output {mapped_output:?}"
                        ),
                    })?;
                let mapped_input = mapped(&forward_to_backward, *input)?;
                accumulate_grad(&mut backward, block, &mut grad_map, mapped_input, upstream)?;
            }
            Op::DequantizeLinear { input, .. } => {
                // Dequantize is a scale operation; pass gradient through as straight-through.
                let upstream = grad_map
                    .get(&mapped_output)
                    .copied()
                    .ok_or_else(|| AutogradError {
                        message: format!(
                            "DequantizeLinear backward: missing upstream gradient for output {mapped_output:?}"
                        ),
                    })?;
                let mapped_input = mapped(&forward_to_backward, *input)?;
                accumulate_grad(&mut backward, block, &mut grad_map, mapped_input, upstream)?;
            }
            Op::MultiHeadAttentionBackward { .. } => {
                // MHABackward is a generated backward op — it does not need a second-order gradient.
            }
        }
    }

    let mut gradients = HashMap::new();
    for parameter in parameters {
        let mapped_param = mapped(&forward_to_backward, *parameter)?;
        if let Some(gradient) = grad_map.get(&mapped_param).copied() {
            gradients.insert(*parameter, gradient);
        }
    }

    let result = GradientGraph {
        forward: forward.clone(),
        backward,
        gradients,
        loss_grad_input: seed_grad,
    };

    verify_graph(&result.backward).expect("backward graph verification failed");
    Ok(result)
}

fn mapped(
    forward_to_backward: &HashMap<ValueId, ValueId>,
    source: ValueId,
) -> Result<ValueId, AutogradError> {
    forward_to_backward
        .get(&source)
        .copied()
        .ok_or_else(|| AutogradError {
            message: format!("Missing mapped ValueId for forward value {source:?}"),
        })
}

fn tensor_shape_for(
    shape_facts: &HashMap<ValueId, ShapeFact>,
    value: ValueId,
) -> Result<Vec<usize>, AutogradError> {
    match shape_facts.get(&value) {
        Some(ShapeFact::Tensor(shape)) => Ok(shape.clone()),
        Some(ShapeFact::Unknown) => Err(AutogradError {
            message: format!("Missing concrete tensor shape for value {value:?}"),
        }),
        Some(ShapeFact::NonTensor) => Err(AutogradError {
            message: format!("Expected tensor shape for value {value:?}, got non-tensor"),
        }),
        None => Err(AutogradError {
            message: format!("Missing shape fact for value {value:?}"),
        }),
    }
}

fn tensor_shape_maybe(
    shape_facts: &HashMap<ValueId, ShapeFact>,
    value: ValueId,
) -> Option<Vec<usize>> {
    match shape_facts.get(&value) {
        Some(ShapeFact::Tensor(shape)) => Some(shape.clone()),
        _ => None,
    }
}

fn unbroadcast_gradient(
    backward: &mut Graph,
    block: crate::ir::BasicBlockId,
    grad: ValueId,
    grad_shape: &[usize],
    target_shape: &[usize],
    op: &str,
) -> Result<ValueId, AutogradError> {
    if grad_shape == target_shape {
        return Ok(grad);
    }

    if target_shape.len() > grad_shape.len() {
        return Err(AutogradError {
            message: format!(
                "{op}: cannot unbroadcast rank {} gradient to rank {} target",
                grad_shape.len(),
                target_shape.len()
            ),
        });
    }

    let mut reduced = grad;
    let mut current_shape = grad_shape.to_vec();

    // Remove prepended dimensions introduced during forward broadcast alignment.
    while current_shape.len() > target_shape.len() {
        let (_, next) = backward
            .add_op(
                block,
                Op::ReduceSum {
                    input: reduced,
                    axis: Some(0),
                    keepdims: false,
                },
            )
            .map_err(|err| AutogradError {
                message: format!("{op}: failed to reduce prepended axis: {}", err.message),
            })?;
        reduced = next;
        current_shape.remove(0);
        if current_shape.is_empty() {
            current_shape.push(1);
        }
    }

    // Reduce axes where the original input had dimension 1.
    let mut axes_to_reduce = Vec::new();
    for (axis, (&curr, &target)) in current_shape.iter().zip(target_shape.iter()).enumerate() {
        if target == 1 && curr != 1 {
            axes_to_reduce.push(axis);
        } else if curr != target {
            return Err(AutogradError {
                message: format!(
                    "{op}: cannot unbroadcast shape {:?} to {:?}",
                    current_shape, target_shape
                ),
            });
        }
    }

    for axis in axes_to_reduce.into_iter().rev() {
        let (_, next) = backward
            .add_op(
                block,
                Op::ReduceSum {
                    input: reduced,
                    axis: Some(axis),
                    keepdims: false,
                },
            )
            .map_err(|err| AutogradError {
                message: format!("{op}: failed to reduce broadcast axis: {}", err.message),
            })?;
        reduced = next;
        current_shape.remove(axis);
        if current_shape.is_empty() {
            current_shape.push(1);
        }
    }

    if current_shape != target_shape {
        let (_, reshaped) = backward
            .add_op(
                block,
                Op::Reshape {
                    input: reduced,
                    shape: target_shape.to_vec(),
                },
            )
            .map_err(|err| AutogradError {
                message: format!("{op}: failed to reshape reduced gradient: {}", err.message),
            })?;
        reduced = reshaped;
    }

    Ok(reduced)
}

fn const_zeros(
    graph: &mut Graph,
    block: crate::ir::BasicBlockId,
    shape: &[usize],
) -> Result<ValueId, AutogradError> {
    let element_count = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim));
    let size = element_count.ok_or_else(|| AutogradError {
        message: format!("Zero tensor shape overflow for shape {shape:?}"),
    })?;
    let (_, value) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: shape.to_vec(),
                data: vec![0.0; size],
            },
        )
        .map_err(|err| AutogradError {
            message: format!("Failed to build zero tensor: {}", err.message),
        })?;
    Ok(value)
}

fn remap_forward_op(
    op: &Op,
    forward_to_backward: &HashMap<ValueId, ValueId>,
) -> Result<Op, AutogradError> {
    let mut mapped_op = op.clone();
    let mut error: Option<AutogradError> = None;
    mapped_op.remap_inputs(|source| {
        if let Some(target) = forward_to_backward.get(&source).copied() {
            target
        } else {
            error = Some(AutogradError {
                message: format!("Forward node references unmapped input value {source:?} during reverse graph build"),
            });
            source
        }
    });
    if let Some(err) = error {
        Err(err)
    } else {
        Ok(mapped_op)
    }
}

fn accumulate_grad(
    graph: &mut Graph,
    block: crate::ir::BasicBlockId,
    grad_map: &mut HashMap<ValueId, ValueId>,
    target: ValueId,
    contribution: ValueId,
) -> Result<(), AutogradError> {
    if let Some(existing) = grad_map.get(&target).copied() {
        let (_, sum) = graph
            .add_op(block, Op::Add(existing, contribution))
            .map_err(|err| AutogradError {
                message: format!("Failed to accumulate gradient: {}", err.message),
            })?;
        grad_map.insert(target, sum);
    } else {
        grad_map.insert(target, contribution);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::ir::{
        ExecutionContext, Graph, Op, RuntimeValue, build_reverse_graph, execute_value_with_context,
    };

    #[test]
    fn builds_separate_backward_graph_without_mutating_forward() {
        let mut forward = Graph::new();
        let block = forward.create_block();
        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, w) = forward
            .add_op(block, Op::Parameter("w".to_string()))
            .expect("add op should succeed");
        let (_, y) = forward
            .add_op(block, Op::Input("y".to_string()))
            .expect("add op should succeed");
        let (_, pred) = forward
            .add_op(block, Op::MatMul(x, w))
            .expect("add op should succeed");
        let (_, diff) = forward
            .add_op(block, Op::Sub(pred, y))
            .expect("add op should succeed");
        let (_, sq) = forward
            .add_op(block, Op::Mul(diff, diff))
            .expect("add op should succeed");
        let (_, loss) = forward
            .add_op(block, Op::Output(sq))
            .expect("add op should succeed");

        let before_nodes = forward.nodes.len();
        let backward = build_reverse_graph(&forward, loss, &[w]).expect("autograd should pass");
        assert_eq!(forward.nodes.len(), before_nodes);
        assert!(backward.backward.nodes.len() > before_nodes);
        assert!(backward.gradients.contains_key(&w));

        let mut context = ExecutionContext::default();
        context.inputs.insert(
            "x".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 1], vec![2.0]).unwrap(),
            )),
        );
        context.inputs.insert(
            "y".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 1], vec![1.0]).unwrap(),
            )),
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 1], vec![1.0]).unwrap(),
            )),
        );
        context.parameters.insert(
            "w".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 1], vec![0.5]).unwrap(),
            )),
        );

        let grad_value = backward.gradients.get(&w).copied().expect("grad exists");
        let grad = execute_value_with_context(&backward.backward, grad_value, &context)
            .expect("grad execute should pass");
        let RuntimeValue::Tensor(__tensor) = grad else {
            panic!("expected tensor grad");
        };
        let data = &__tensor.data;
        assert!(!data.is_empty());
    }

    #[test]
    fn reshape_backward_restores_input_shape() {
        let mut forward = Graph::new();
        let block = forward.create_block();
        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        forward.bind_input_shape("x", vec![1, 4]);
        let (_, y) = forward
            .add_op(
                block,
                Op::Reshape {
                    input: x,
                    shape: vec![2, 2],
                },
            )
            .expect("add op should succeed");
        let (_, out) = forward
            .add_op(block, Op::Output(y))
            .expect("add op should succeed");

        let backward = build_reverse_graph(&forward, out, &[x]).expect("autograd should pass");
        let grad_value = backward.gradients.get(&x).copied().expect("grad exists");

        let mut context = ExecutionContext::default();
        context.inputs.insert(
            "x".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
            )),
        );

        let grad = execute_value_with_context(&backward.backward, grad_value, &context)
            .expect("grad execute should pass");
        assert_eq!(
            grad,
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 4], vec![1.0, 1.0, 1.0, 1.0]).unwrap()
            ))
        );
    }

    #[test]
    fn concat_backward_splits_upstream_gradient() {
        let mut forward = Graph::new();
        let block = forward.create_block();
        let (_, x1) = forward
            .add_op(block, Op::Input("x1".to_string()))
            .expect("add op should succeed");
        let (_, x2) = forward
            .add_op(block, Op::Input("x2".to_string()))
            .expect("add op should succeed");
        forward.bind_input_shape("x1", vec![1, 2]);
        forward.bind_input_shape("x2", vec![1, 2]);
        let (_, cat) = forward
            .add_op(
                block,
                Op::Concat {
                    inputs: vec![x1, x2],
                    axis: 1,
                },
            )
            .expect("add op should succeed");
        let (_, out) = forward
            .add_op(block, Op::Output(cat))
            .expect("add op should succeed");

        let backward = build_reverse_graph(&forward, out, &[x1, x2]).expect("autograd should pass");
        let g1 = backward.gradients.get(&x1).copied().expect("x1 grad");
        let g2 = backward.gradients.get(&x2).copied().expect("x2 grad");

        let mut context = ExecutionContext::default();
        context.inputs.insert(
            "x1".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 2], vec![0.0, 0.0]).unwrap(),
            )),
        );
        context.inputs.insert(
            "x2".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 2], vec![0.0, 0.0]).unwrap(),
            )),
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        );

        let grad1 = execute_value_with_context(&backward.backward, g1, &context)
            .expect("x1 grad execute should pass");
        let grad2 = execute_value_with_context(&backward.backward, g2, &context)
            .expect("x2 grad execute should pass");

        assert_eq!(
            grad1,
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap()
            ))
        );
        assert_eq!(
            grad2,
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 2], vec![3.0, 4.0]).unwrap()
            ))
        );
    }

    #[test]
    fn gather_backward_reconstructs_sparse_gradient_for_unique_indices() {
        let mut forward = Graph::new();
        let block = forward.create_block();
        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        forward.bind_input_shape("x", vec![4]);
        let (_, gathered) = forward
            .add_op(
                block,
                Op::Gather {
                    input: x,
                    indices: vec![2, 0],
                    axis: 0,
                },
            )
            .expect("add op should succeed");
        let (_, out) = forward
            .add_op(block, Op::Output(gathered))
            .expect("add op should succeed");

        let backward = build_reverse_graph(&forward, out, &[x]).expect("autograd should pass");
        let gx = backward.gradients.get(&x).copied().expect("x grad");

        let mut context = ExecutionContext::default();
        context.inputs.insert(
            "x".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            )),
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![2], vec![10.0, 20.0]).unwrap(),
            )),
        );

        let grad =
            execute_value_with_context(&backward.backward, gx, &context).expect("grad execute");
        assert_eq!(
            grad,
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![4], vec![20.0, 0.0, 10.0, 0.0]).unwrap()
            ))
        );
    }

    #[test]
    fn gather_backward_accumulates_duplicate_indices_via_scatter_sum() {
        // Gather indices [1, 1] from a [4]-element input.
        // The backward pass must SUM the upstream gradients for slot 1
        // (scatter-add semantics), not reject the duplicate.
        let mut forward = Graph::new();
        let block = forward.create_block();
        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        forward.bind_input_shape("x", vec![4]);
        let (_, gathered) = forward
            .add_op(
                block,
                Op::Gather {
                    input: x,
                    indices: vec![1, 1], // duplicate index intentionally
                    axis: 0,
                },
            )
            .expect("add op should succeed");
        let (_, out) = forward
            .add_op(block, Op::Output(gathered))
            .expect("add op should succeed");

        // Must succeed — duplicate indices are now allowed via accumulation.
        let grad_graph = build_reverse_graph(&forward, out, &[x])
            .expect("gather with duplicate indices should build backward graph");

        // The backward graph must contain at least one Op::Add (the accumulation
        // of the two gradient slices for slot 1).
        let has_add = grad_graph
            .backward
            .nodes
            .iter()
            .any(|n| matches!(n.op, Op::Add(_, _)));
        assert!(
            has_add,
            "expected Op::Add for gradient accumulation of duplicate index"
        );
    }

    #[test]
    fn flatten_backward_restores_input_shape() {
        let mut forward = Graph::new();
        let block = forward.create_block();
        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        forward.bind_input_shape("x", vec![2, 3, 4]);
        let (_, y) = forward
            .add_op(block, Op::Flatten { input: x, axis: 1 })
            .expect("add op should succeed");
        let (_, out) = forward
            .add_op(block, Op::Output(y))
            .expect("add op should succeed");

        let backward = build_reverse_graph(&forward, out, &[x]).expect("autograd should pass");
        let gx = backward.gradients.get(&x).copied().expect("x grad");

        let mut context = ExecutionContext::default();
        context.inputs.insert(
            "x".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![2, 3, 4], vec![0.0; 24]).unwrap(),
            )),
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![2, 12], vec![1.0; 24]).unwrap(),
            )),
        );

        let grad = execute_value_with_context(&backward.backward, gx, &context)
            .expect("grad execute should pass");
        let RuntimeValue::Tensor(t) = grad else {
            panic!("expected tensor grad");
        };
        assert_eq!(t.shape, vec![2, 3, 4]);
    }

    #[test]
    fn maxpool_autograd_produces_input_shaped_gradient() {
        let mut forward = Graph::new();
        let block = forward.create_block();
        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        forward.bind_input_shape("x", vec![1, 1, 3, 3]);
        let (_, y) = forward
            .add_op(
                block,
                Op::MaxPool {
                    input: x,
                    kernel_shape: vec![2, 2],
                    strides: vec![1, 1],
                    pads: vec![],
                },
            )
            .expect("add op should succeed");
        let (_, out) = forward
            .add_op(block, Op::Output(y))
            .expect("add op should succeed");

        let backward = build_reverse_graph(&forward, out, &[x]).expect("autograd should pass");
        let gx = backward.gradients.get(&x).copied().expect("x grad");

        let mut context = ExecutionContext::default();
        context.inputs.insert(
            "x".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(
                    vec![1, 1, 3, 3],
                    vec![1.0, 2.0, 3.0, 4.0, 10.0, 6.0, 7.0, 8.0, 9.0],
                )
                .unwrap(),
            )),
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
            )),
        );

        let grad = execute_value_with_context(&backward.backward, gx, &context)
            .expect("grad execute should pass");
        let RuntimeValue::Tensor(t) = grad else {
            panic!("expected tensor grad");
        };
        assert_eq!(t.shape, vec![1, 1, 3, 3]);
    }

    #[test]
    fn avgpool_autograd_produces_input_shaped_gradient() {
        let mut forward = Graph::new();
        let block = forward.create_block();
        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        forward.bind_input_shape("x", vec![1, 1, 3, 3]);
        let (_, y) = forward
            .add_op(
                block,
                Op::AvgPool {
                    input: x,
                    kernel_shape: vec![2, 2],
                    strides: vec![1, 1],
                    pads: vec![],
                },
            )
            .expect("add op should succeed");
        let (_, out) = forward
            .add_op(block, Op::Output(y))
            .expect("add op should succeed");

        let backward = build_reverse_graph(&forward, out, &[x]).expect("autograd should pass");
        let gx = backward.gradients.get(&x).copied().expect("x grad");

        let mut context = ExecutionContext::default();
        context.inputs.insert(
            "x".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 1, 3, 3], vec![1.0; 9]).unwrap(),
            )),
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 1, 2, 2], vec![1.0; 4]).unwrap(),
            )),
        );

        let grad = execute_value_with_context(&backward.backward, gx, &context)
            .expect("grad execute should pass");
        let RuntimeValue::Tensor(t) = grad else {
            panic!("expected tensor grad");
        };
        assert_eq!(t.shape, vec![1, 1, 3, 3]);
    }

    #[test]
    fn batchnorm_autograd_emits_weight_and_bias_gradients() {
        let mut forward = Graph::new();
        let block = forward.create_block();
        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, w) = forward
            .add_op(block, Op::Parameter("bn_w".to_string()))
            .expect("add op should succeed");
        let (_, b) = forward
            .add_op(block, Op::Parameter("bn_b".to_string()))
            .expect("add op should succeed");
        let (_, m) = forward
            .add_op(block, Op::Input("bn_mean".to_string()))
            .expect("add op should succeed");
        let (_, v) = forward
            .add_op(block, Op::Input("bn_var".to_string()))
            .expect("add op should succeed");
        forward.bind_input_shape("x", vec![1, 2, 1, 1]);
        forward.bind_input_shape("bn_mean", vec![2]);
        forward.bind_input_shape("bn_var", vec![2]);
        forward.bind_parameter_shape("bn_w", vec![2]);
        forward.bind_parameter_shape("bn_b", vec![2]);

        let (_, y) = forward
            .add_op(
                block,
                Op::BatchNorm {
                    input: x,
                    weight: w,
                    bias: b,
                    mean: m,
                    var: v,
                },
            )
            .expect("add op should succeed");
        let (_, out) = forward
            .add_op(block, Op::Output(y))
            .expect("add op should succeed");

        let backward = build_reverse_graph(&forward, out, &[w, b]).expect("autograd should pass");
        assert!(backward.gradients.contains_key(&w));
        assert!(backward.gradients.contains_key(&b));
    }

    #[test]
    fn layernorm_autograd_emits_input_weight_and_bias_gradients() {
        let mut forward = Graph::new();
        let block = forward.create_block();
        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, w) = forward
            .add_op(block, Op::Parameter("ln_w".to_string()))
            .expect("add op should succeed");
        let (_, b) = forward
            .add_op(block, Op::Parameter("ln_b".to_string()))
            .expect("add op should succeed");

        forward.bind_input_shape("x", vec![2, 5]);
        forward.bind_parameter_shape("ln_w", vec![5]);
        forward.bind_parameter_shape("ln_b", vec![5]);

        let (_, y) = forward
            .add_op(
                block,
                Op::LayerNorm {
                    input: x,
                    weight: w,
                    bias: b,
                    epsilon: 1e-5,
                },
            )
            .expect("add op should succeed");
        let (_, out) = forward
            .add_op(block, Op::Output(y))
            .expect("add op should succeed");

        let backward =
            build_reverse_graph(&forward, out, &[x, w, b]).expect("autograd should pass");
        assert!(backward.gradients.contains_key(&x));
        assert!(backward.gradients.contains_key(&w));
        assert!(backward.gradients.contains_key(&b));
    }

    #[test]
    fn slice_backward_pads_gradient_to_input_shape() {
        let mut forward = Graph::new();
        let block = forward.create_block();
        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        forward.bind_input_shape("x", vec![1, 6]);
        let (_, sliced) = forward
            .add_op(
                block,
                Op::Slice {
                    input: x,
                    starts: vec![2],
                    ends: vec![5],
                    axes: vec![1],
                },
            )
            .expect("add op should succeed");
        let (_, out) = forward
            .add_op(block, Op::Output(sliced))
            .expect("add op should succeed");

        let backward = build_reverse_graph(&forward, out, &[x]).expect("autograd should pass");
        let gx = backward.gradients.get(&x).copied().expect("x grad");

        let mut context = ExecutionContext::default();
        context.inputs.insert(
            "x".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 6], vec![0.0; 6]).unwrap(),
            )),
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 3], vec![7.0, 8.0, 9.0]).unwrap(),
            )),
        );

        let grad =
            execute_value_with_context(&backward.backward, gx, &context).expect("grad execute");
        assert_eq!(
            grad,
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1, 6], vec![0.0, 0.0, 7.0, 8.0, 9.0, 0.0])
                    .unwrap()
            ))
        );
    }

    #[test]
    fn conv2d_autograd_computes_input_and_weight_gradients() {
        let mut forward = Graph::new();
        let block = forward.create_block();

        // 3x3 input
        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add x should succeed");
        forward.bind_input_shape("x", vec![3, 3]);

        // 2x2 kernel
        let (_, w) = forward
            .add_op(block, Op::Parameter("w".to_string()))
            .expect("add w should succeed");
        forward.bind_parameter_shape("w", vec![2, 2]);

        // Output will be 2x2
        let (_, conv) = forward
            .add_op(block, Op::Conv2D(x, w))
            .expect("conv2d should succeed");

        let (_, out) = forward
            .add_op(block, Op::Output(conv))
            .expect("output should succeed");

        let backward = build_reverse_graph(&forward, out, &[x, w]).expect("autograd should pass");
        let gx = backward.gradients.get(&x).copied().expect("x grad exists");
        let gw = backward.gradients.get(&w).copied().expect("w grad exists");

        let mut context = ExecutionContext::default();
        context.inputs.insert(
            "x".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(
                    vec![3, 3],
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                )
                .unwrap(),
            )),
        );
        context.parameters.insert(
            "w".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![2, 2], vec![1.0, 0.5, 0.0, -1.0]).unwrap(),
            )),
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
            )),
        );

        let grad_x =
            execute_value_with_context(&backward.backward, gx, &context).expect("gx execute");
        let grad_w =
            execute_value_with_context(&backward.backward, gw, &context).expect("gw execute");

        let RuntimeValue::Tensor(tx) = grad_x else {
            panic!("expected tensor")
        };
        let RuntimeValue::Tensor(tw) = grad_w else {
            panic!("expected tensor")
        };

        assert_eq!(tx.shape, vec![3, 3]);
        assert_eq!(tw.shape, vec![2, 2]);

        // Manual calculation for weight gradient (correlation):
        // gw[0,0] = x[0,0]*g[0,0] + x[0,1]*g[0,1] + x[1,0]*g[1,0] + x[1,1]*g[1,1] = 1*1 + 2*1 + 4*1 + 5*1 = 12
        // gw[0,1] = x[0,1]*g[0,0] + x[0,2]*g[0,1] + x[1,1]*g[1,0] + x[1,2]*g[1,1] = 2*1 + 3*1 + 5*1 + 6*1 = 16
        // gw[1,0] = x[1,0]*g[0,0] + x[1,1]*g[0,1] + x[2,0]*g[1,0] + x[2,1]*g[1,1] = 4*1 + 5*1 + 7*1 + 8*1 = 24
        // gw[1,1] = x[1,1]*g[0,0] + x[1,2]*g[0,1] + x[2,1]*g[1,0] + x[2,2]*g[1,1] = 5*1 + 6*1 + 8*1 + 9*1 = 28
        assert_eq!(*tw.data, vec![12.0, 16.0, 24.0, 28.0]);
    }

    #[test]
    fn build_reverse_graph_shared_output() {
        // Build a simple graph: x -> relu -> (a) -> add(a, a) -> loss
        // The value `a` is used as both inputs to the add — shared output feeding two ops.
        let mut g = Graph::new();
        let block = g.create_block();
        let (_, x) = g.add_op(block, Op::Input("x".to_string())).unwrap();
        let (_, a) = g.add_op(block, Op::Relu(x)).unwrap();
        // Use Add(a, a) so `a` fans out to two input edges
        let (_, loss) = g.add_op(block, Op::Add(a, a)).unwrap();
        g.add_op(block, Op::Output(loss)).unwrap();

        let result = build_reverse_graph(&g, loss, &[x]);
        assert!(result.is_ok(), "build_reverse_graph panicked or errored on shared-output graph: {:?}", result.err());
    }

    /// Helper: build a forward graph with 6 MHA output nodes plus a ReduceSum scalar loss.
    /// Returns (graph, block, q_input_id, k_input_id, v_input_id, w_q_id, w_k_id, w_v_id,
    ///          w_o_id, mha_output_id, loss_id)
    #[allow(clippy::too_many_arguments)]
    fn build_mha_graph(
        d_model: usize,
        num_heads: usize,
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        causal: bool,
    ) -> (
        Graph,
        crate::ir::BasicBlockId,
        crate::ir::ValueId, // q_input
        crate::ir::ValueId, // k_input
        crate::ir::ValueId, // v_input
        crate::ir::ValueId, // w_q
        crate::ir::ValueId, // w_k
        crate::ir::ValueId, // w_v
        crate::ir::ValueId, // w_o
        crate::ir::ValueId, // mha output (output_idx=0)
        crate::ir::ValueId, // loss (ReduceSum of output)
    ) {
        let mut g = Graph::new();
        let block = g.create_block();

        let (_, q_in) = g.add_op(block, Op::Input("q_input".to_string())).unwrap();
        let (_, k_in) = g.add_op(block, Op::Input("k_input".to_string())).unwrap();
        let (_, v_in) = g.add_op(block, Op::Input("v_input".to_string())).unwrap();
        let (_, wq) = g.add_op(block, Op::Parameter("w_q".to_string())).unwrap();
        let (_, wk) = g.add_op(block, Op::Parameter("w_k".to_string())).unwrap();
        let (_, wv) = g.add_op(block, Op::Parameter("w_v".to_string())).unwrap();
        let (_, wo) = g.add_op(block, Op::Parameter("w_o".to_string())).unwrap();

        // Zero biases as constants
        let (_, bq) = g.add_op(block, Op::ConstTensor { shape: vec![d_model], data: vec![0.0; d_model] }).unwrap();
        let (_, bk) = g.add_op(block, Op::ConstTensor { shape: vec![d_model], data: vec![0.0; d_model] }).unwrap();
        let (_, bv) = g.add_op(block, Op::ConstTensor { shape: vec![d_model], data: vec![0.0; d_model] }).unwrap();
        let (_, bo) = g.add_op(block, Op::ConstTensor { shape: vec![d_model], data: vec![0.0; d_model] }).unwrap();

        let make_mha = |g: &mut Graph, idx: usize| {
            g.add_op(block, Op::MultiHeadAttention {
                q_input: q_in, k_input: k_in, v_input: v_in,
                w_q: wq, w_k: wk, w_v: wv, w_o: wo,
                bias_q: bq, bias_k: bk, bias_v: bv, bias_o: bo,
                num_heads, causal, output_idx: idx,
            }).unwrap().1
        };

        // output_idx=0 is the primary output; 1..5 are saved activations
        let mha_out = make_mha(&mut g, 0);
        let _mha_aw  = make_mha(&mut g, 1);
        let _mha_qp  = make_mha(&mut g, 2);
        let _mha_kp  = make_mha(&mut g, 3);
        let _mha_vp  = make_mha(&mut g, 4);
        let _mha_ctx = make_mha(&mut g, 5);

        // Bind input shapes for shape inference
        g.bind_input_shape("q_input", vec![batch, seq_q, d_model]);
        g.bind_input_shape("k_input", vec![batch, seq_k, d_model]);
        g.bind_input_shape("v_input", vec![batch, seq_k, d_model]);

        let (_, loss) = g.add_op(block, Op::ReduceSum { input: mha_out, axis: None, keepdims: false }).unwrap();

        (g, block, q_in, k_in, v_in, wq, wk, wv, wo, mha_out, loss)
    }

    #[test]
    fn mha_backward_reduces_loss() {
        // d_model=4, num_heads=2, batch=1, seq=2, lr=0.01
        // Run 100 SGD steps on w_q; check final_loss < initial_loss.
        let d_model = 4_usize;
        let num_heads = 2_usize;
        let batch = 1_usize;
        let seq = 2_usize;
        let lr = 0.01_f32;

        let (forward, _, q_in_id, _k_in_id, _v_in_id, wq_id, _wk_id, _wv_id, _wo_id, _mha_out_id, loss_id) =
            build_mha_graph(d_model, num_heads, batch, seq, seq, false);

        let grad_graph = build_reverse_graph(&forward, loss_id, &[wq_id])
            .expect("build_reverse_graph should succeed");

        let wq_grad_value = *grad_graph.gradients.get(&wq_id).expect("w_q gradient must exist");

        // Initial w_q: small random-like values via a deterministic pattern
        let mut wq_data: Vec<f32> = (0..d_model * d_model)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        // Inputs: fixed throughout training
        let q_data: Vec<f32> = (0..batch * seq * d_model).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let make_tensor = |data: Vec<f32>, shape: Vec<usize>| {
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(shape, data).unwrap(),
            ))
        };

        // Execute forward to get initial loss
        let mut context = ExecutionContext::default();
        context.inputs.insert("q_input".to_string(), make_tensor(q_data.clone(), vec![batch, seq, d_model]));
        context.inputs.insert("k_input".to_string(), make_tensor(q_data.clone(), vec![batch, seq, d_model]));
        context.inputs.insert("v_input".to_string(), make_tensor(q_data.clone(), vec![batch, seq, d_model]));
        context.inputs.insert("__loss_grad".to_string(), make_tensor(vec![1.0], vec![1]));
        context.parameters.insert("w_q".to_string(), make_tensor(wq_data.clone(), vec![d_model, d_model]));
        context.parameters.insert("w_k".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));
        context.parameters.insert("w_v".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));
        context.parameters.insert("w_o".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));

        let initial_loss = {
            let rv = execute_value_with_context(&grad_graph.backward, {
                // Get the loss value id mapped in backward graph
                let mapped_loss = *forward.nodes.iter()
                    .find(|n| n.output == loss_id)
                    .map(|_| grad_graph.backward.nodes.iter()
                        .rev()
                        .find(|n| matches!(&n.op, Op::ReduceSum { .. }))
                        .map(|n| &n.output)
                        .unwrap())
                    .unwrap();
                mapped_loss
            }, &context).expect("execute initial loss");
            if let RuntimeValue::Tensor(t) = rv { t.data[0] } else { panic!("expected tensor") }
        };

        // SGD loop
        for _ in 0..100 {
            let mut ctx = ExecutionContext::default();
            ctx.inputs.insert("q_input".to_string(), make_tensor(q_data.clone(), vec![batch, seq, d_model]));
            ctx.inputs.insert("k_input".to_string(), make_tensor(q_data.clone(), vec![batch, seq, d_model]));
            ctx.inputs.insert("v_input".to_string(), make_tensor(q_data.clone(), vec![batch, seq, d_model]));
            ctx.inputs.insert("__loss_grad".to_string(), make_tensor(vec![1.0], vec![1]));
            ctx.parameters.insert("w_q".to_string(), make_tensor(wq_data.clone(), vec![d_model, d_model]));
            ctx.parameters.insert("w_k".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));
            ctx.parameters.insert("w_v".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));
            ctx.parameters.insert("w_o".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));

            // Get gradient for w_q
            let grad_rv = execute_value_with_context(&grad_graph.backward, wq_grad_value, &ctx)
                .expect("execute grad");
            let RuntimeValue::Tensor(grad_t) = grad_rv else { panic!("expected tensor grad") };

            // SGD update: w_q -= lr * grad
            let grad_c = grad_t.make_contiguous().unwrap();
            for (w, g) in wq_data.iter_mut().zip(grad_c.data.iter()) {
                *w -= lr * g;
            }
        }

        // Compute final loss
        let mut ctx = ExecutionContext::default();
        ctx.inputs.insert("q_input".to_string(), make_tensor(q_data.clone(), vec![batch, seq, d_model]));
        ctx.inputs.insert("k_input".to_string(), make_tensor(q_data.clone(), vec![batch, seq, d_model]));
        ctx.inputs.insert("v_input".to_string(), make_tensor(q_data.clone(), vec![batch, seq, d_model]));
        ctx.inputs.insert("__loss_grad".to_string(), make_tensor(vec![1.0], vec![1]));
        ctx.parameters.insert("w_q".to_string(), make_tensor(wq_data.clone(), vec![d_model, d_model]));
        ctx.parameters.insert("w_k".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));
        ctx.parameters.insert("w_v".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));
        ctx.parameters.insert("w_o".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));

        let final_loss = {
            let rv = execute_value_with_context(&grad_graph.backward, {
                grad_graph.backward.nodes.iter()
                    .rev()
                    .find(|n| matches!(&n.op, Op::ReduceSum { .. }))
                    .map(|n| n.output)
                    .unwrap()
            }, &ctx).expect("execute final loss");
            if let RuntimeValue::Tensor(t) = rv { t.data[0] } else { panic!("expected tensor") }
        };

        assert!(
            final_loss < initial_loss,
            "Expected loss to decrease after 100 SGD steps; initial={initial_loss}, final={final_loss}"
        );
    }

    fn eye(n: usize) -> Vec<f32> {
        let mut data = vec![0.0_f32; n * n];
        for i in 0..n { data[i * n + i] = 1.0; }
        data
    }

    #[test]
    fn mha_gradient_matches_reference() {
        // batch=1, seq=2, d_model=4, num_heads=2, identity weights
        // Differentiate w.r.t. v_input: grad = d(sum(MHA_output))/d(v_input).
        // With identity weights and sum loss, dv[j] = sum_i(aw[i,j]) per element (non-trivial).
        // Compare analytical gradient against numerical finite-difference reference.
        let d_model = 4_usize;
        let num_heads = 2_usize;
        let batch = 1_usize;
        let seq = 2_usize;

        // Build graph differentiating w.r.t. v_input
        let (forward, _, _q_in_id, _k_in_id, v_in_id, _wq_id, _wk_id, _wv_id, _wo_id, _mha_out_id, loss_id) =
            build_mha_graph(d_model, num_heads, batch, seq, seq, false);

        let grad_graph = build_reverse_graph(&forward, loss_id, &[v_in_id])
            .expect("build_reverse_graph should succeed");

        let dv_grad_value = *grad_graph.gradients.get(&v_in_id).expect("v_input gradient must exist");

        // Use specific non-trivial inputs
        let q_data: Vec<f32> = vec![1.0, 0.5, 0.0, 0.0,  0.0, 1.0, 0.5, 0.0]; // [1,2,4]
        let v_data: Vec<f32> = vec![0.5, 1.0, 0.0, 0.0,  0.0, 0.5, 1.0, 0.0]; // [1,2,4]

        let make_tensor = |data: Vec<f32>, shape: Vec<usize>| {
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(shape, data).unwrap(),
            ))
        };

        let mut ctx = ExecutionContext::default();
        ctx.inputs.insert("q_input".to_string(), make_tensor(q_data.clone(), vec![batch, seq, d_model]));
        ctx.inputs.insert("k_input".to_string(), make_tensor(q_data.clone(), vec![batch, seq, d_model]));
        ctx.inputs.insert("v_input".to_string(), make_tensor(v_data.clone(), vec![batch, seq, d_model]));
        ctx.inputs.insert("__loss_grad".to_string(), make_tensor(vec![1.0], vec![1]));
        ctx.parameters.insert("w_q".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));
        ctx.parameters.insert("w_k".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));
        ctx.parameters.insert("w_v".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));
        ctx.parameters.insert("w_o".to_string(), make_tensor(eye(d_model), vec![d_model, d_model]));

        let dv_rv = execute_value_with_context(&grad_graph.backward, dv_grad_value, &ctx)
            .expect("execute dv_input gradient");
        let RuntimeValue::Tensor(dv_t) = dv_rv else { panic!("expected tensor") };
        let dv_c = dv_t.make_contiguous().unwrap();
        let dv_actual: Vec<f32> = dv_c.data.to_vec();
        // eprintln!("dv_actual shape={:?} values={:?}", dv_t.shape, &dv_actual);

        // Numerical gradient: perturb v_input, hold q/k fixed
        let eps = 1e-3_f32;
        let loss_fn_v = |v: Vec<f32>| -> f32 {
            use crate::engine::ir::kernels::attention::multi_head_attention;
            let qk = crate::ir::tensor::Tensor::new(vec![batch, seq, d_model], q_data.clone()).unwrap();
            let vt = crate::ir::tensor::Tensor::new(vec![batch, seq, d_model], v).unwrap();
            let w = crate::ir::tensor::Tensor::new(vec![d_model, d_model], eye(d_model)).unwrap();
            let b = crate::ir::tensor::Tensor::new(vec![d_model], vec![0.0; d_model]).unwrap();
            let out = multi_head_attention(&qk, &qk, &vt, &w, &w, &w, &w, &b, &b, &b, &b, num_heads, false)
                .expect("mha forward");
            let oc = out.output.make_contiguous().unwrap();
            oc.data.iter().sum()
        };

        let mut dv_numerical = vec![0.0_f32; batch * seq * d_model];
        for i in 0..batch * seq * d_model {
            let mut v_plus = v_data.clone();
            let mut v_minus = v_data.clone();
            v_plus[i] += eps;
            v_minus[i] -= eps;
            dv_numerical[i] = (loss_fn_v(v_plus) - loss_fn_v(v_minus)) / (2.0 * eps);
        }

        // Verify the gradient is non-trivially non-zero (test is meaningful)
        let max_numerical = dv_numerical.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()));
        assert!(max_numerical > 0.01, "numerical gradient should be non-zero, max={max_numerical}");

        // Check that analytical gradient matches numerical within tolerance 5e-3
        for (i, (actual, expected)) in dv_actual.iter().zip(dv_numerical.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 5e-3,
                "dv_input[{i}]: actual={actual}, numerical={expected}, diff={}",
                (actual - expected).abs()
            );
        }
    }
}
