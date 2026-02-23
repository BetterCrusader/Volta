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
            Op::GeluExact(_) => {
                return Err(AutogradError {
                    message: "GeluExact backward is not implemented yet".to_string(),
                });
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
            Op::ReduceMax { .. } => {
                return Err(AutogradError {
                    message: "ReduceMax backward is not implemented yet".to_string(),
                });
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
            | Op::GemmBackward { .. }
            | Op::Conv2D(_, _)
            | Op::Parameter(_)
            | Op::Input(_)
            | Op::Phi(_)
            | Op::Removed => {}
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
}
