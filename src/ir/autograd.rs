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
            message: format!(
                "Loss value {:?} is not present in forward graph",
                loss_value
            ),
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
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, upstream)?;
                accumulate_grad(&mut backward, block, &mut grad_map, b_m, upstream)?;
            }
            Op::Sub(a, b) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let b_m = mapped(&forward_to_backward, *b)?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, upstream)?;
                let (_, neg) =
                    backward
                        .add_op(block, Op::Neg(upstream))
                        .map_err(|err| AutogradError {
                            message: format!("Failed to build neg gradient: {}", err.message),
                        })?;
                accumulate_grad(&mut backward, block, &mut grad_map, b_m, neg)?;
            }
            Op::Mul(a, b) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let b_m = mapped(&forward_to_backward, *b)?;
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
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, ga)?;
                accumulate_grad(&mut backward, block, &mut grad_map, b_m, gb)?;
            }
            Op::Div(a, b) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let b_m = mapped(&forward_to_backward, *b)?;
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
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, ga)?;
                accumulate_grad(&mut backward, block, &mut grad_map, b_m, gb)?;
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
            Op::Relu(a) => {
                let a_m = mapped(&forward_to_backward, *a)?;
                let (_, grad) = backward
                    .add_op(block, Op::ReluBackward(a_m, upstream))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build relu backward op: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad)?;
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
                            "Concat axis {} out of bounds for input shape {:?}",
                            axis, input_shape
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
                        "Gather axis {} out of bounds for input shape {:?}",
                        axis, input_shape
                    ),
                })?;
                let mut seen = std::collections::HashSet::new();
                for index in indices {
                    if !seen.insert(*index) {
                        return Err(AutogradError {
                            message: format!(
                                "Gather backward currently requires unique indices, duplicate index {} detected",
                                index
                            ),
                        });
                    }
                }

                let mut chunks = Vec::with_capacity(axis_dim);
                for slot in 0..axis_dim {
                    let maybe_pos = indices.iter().position(|idx| *idx == slot);
                    let value = if let Some(pos) = maybe_pos {
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
                                    "Failed to build gather backward selected slice: {}",
                                    err.message
                                ),
                            })?;
                        slice
                    } else {
                        let mut zero_shape = input_shape.clone();
                        zero_shape[*axis] = 1;
                        const_zeros(&mut backward, block, &zero_shape)?
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
                        message: format!("Failed to build gather backward concat: {}", err.message),
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
                            "Slice axis {} out of bounds for input shape {:?}",
                            axis, input_shape
                        ),
                    })?;

                    if start > input_dim || end > input_dim || start >= end {
                        return Err(AutogradError {
                            message: format!(
                                "Invalid slice spec for backward at axis {}: start={} end={} dim={}",
                                axis, start, end, input_dim
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
            Op::ConstInt(_)
            | Op::ConstFloat(_)
            | Op::ConstTensor { .. }
            | Op::Neg(_)
            | Op::ElementwiseChain { .. }
            | Op::Reshape { .. }
            | Op::Concat { .. }
            | Op::Gather { .. }
            | Op::Slice { .. }
            | Op::Transpose(_)
            | Op::ReluBackward(_, _)
            | Op::Softmax(_)
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

    debug_assert!(verify_graph(&result.backward).is_ok());
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
            message: format!("Missing mapped ValueId for forward value {:?}", source),
        })
}

fn tensor_shape_for(
    shape_facts: &HashMap<ValueId, ShapeFact>,
    value: ValueId,
) -> Result<Vec<usize>, AutogradError> {
    match shape_facts.get(&value) {
        Some(ShapeFact::Tensor(shape)) => Ok(shape.clone()),
        Some(ShapeFact::Unknown) => Err(AutogradError {
            message: format!("Missing concrete tensor shape for value {:?}", value),
        }),
        Some(ShapeFact::NonTensor) => Err(AutogradError {
            message: format!(
                "Expected tensor shape for value {:?}, got non-tensor",
                value
            ),
        }),
        None => Err(AutogradError {
            message: format!("Missing shape fact for value {:?}", value),
        }),
    }
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
        message: format!("Zero tensor shape overflow for shape {:?}", shape),
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
                message: format!(
                    "Forward node references unmapped input value {:?} during reverse graph build",
                    source
                ),
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
            RuntimeValue::Tensor {
                shape: vec![1, 1],
                data: vec![2.0],
            },
        );
        context.inputs.insert(
            "y".to_string(),
            RuntimeValue::Tensor {
                shape: vec![1, 1],
                data: vec![1.0],
            },
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor {
                shape: vec![1, 1],
                data: vec![1.0],
            },
        );
        context.parameters.insert(
            "w".to_string(),
            RuntimeValue::Tensor {
                shape: vec![1, 1],
                data: vec![0.5],
            },
        );

        let grad_value = backward.gradients.get(&w).copied().expect("grad exists");
        let grad = execute_value_with_context(&backward.backward, grad_value, &context)
            .expect("grad execute should pass");
        let RuntimeValue::Tensor { data, .. } = grad else {
            panic!("expected tensor grad");
        };
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
            RuntimeValue::Tensor {
                shape: vec![1, 4],
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor {
                shape: vec![2, 2],
                data: vec![1.0, 1.0, 1.0, 1.0],
            },
        );

        let grad = execute_value_with_context(&backward.backward, grad_value, &context)
            .expect("grad execute should pass");
        assert_eq!(
            grad,
            RuntimeValue::Tensor {
                shape: vec![1, 4],
                data: vec![1.0, 1.0, 1.0, 1.0],
            }
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
            RuntimeValue::Tensor {
                shape: vec![1, 2],
                data: vec![0.0, 0.0],
            },
        );
        context.inputs.insert(
            "x2".to_string(),
            RuntimeValue::Tensor {
                shape: vec![1, 2],
                data: vec![0.0, 0.0],
            },
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor {
                shape: vec![1, 4],
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
        );

        let grad1 = execute_value_with_context(&backward.backward, g1, &context)
            .expect("x1 grad execute should pass");
        let grad2 = execute_value_with_context(&backward.backward, g2, &context)
            .expect("x2 grad execute should pass");

        assert_eq!(
            grad1,
            RuntimeValue::Tensor {
                shape: vec![1, 2],
                data: vec![1.0, 2.0],
            }
        );
        assert_eq!(
            grad2,
            RuntimeValue::Tensor {
                shape: vec![1, 2],
                data: vec![3.0, 4.0],
            }
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
            RuntimeValue::Tensor {
                shape: vec![4],
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor {
                shape: vec![2],
                data: vec![10.0, 20.0],
            },
        );

        let grad =
            execute_value_with_context(&backward.backward, gx, &context).expect("grad execute");
        assert_eq!(
            grad,
            RuntimeValue::Tensor {
                shape: vec![4],
                data: vec![20.0, 0.0, 10.0, 0.0],
            }
        );
    }

    #[test]
    fn gather_backward_rejects_duplicate_indices() {
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
                    indices: vec![1, 1],
                    axis: 0,
                },
            )
            .expect("add op should succeed");
        let (_, out) = forward
            .add_op(block, Op::Output(gathered))
            .expect("add op should succeed");

        let err = build_reverse_graph(&forward, out, &[x]).expect_err("must fail loudly");
        assert!(err.message.contains("unique indices"));
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
            RuntimeValue::Tensor {
                shape: vec![1, 6],
                data: vec![0.0; 6],
            },
        );
        context.inputs.insert(
            "__loss_grad".to_string(),
            RuntimeValue::Tensor {
                shape: vec![1, 3],
                data: vec![7.0, 8.0, 9.0],
            },
        );

        let grad =
            execute_value_with_context(&backward.backward, gx, &context).expect("grad execute");
        assert_eq!(
            grad,
            RuntimeValue::Tensor {
                shape: vec![1, 6],
                data: vec![0.0, 0.0, 7.0, 8.0, 9.0, 0.0],
            }
        );
    }
}
