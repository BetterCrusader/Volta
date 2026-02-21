use std::collections::HashMap;

use crate::ir::{Graph, Op, ValueId, verify_graph};

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

        match node.op {
            Op::Add(a, b) => {
                let a_m = mapped(&forward_to_backward, a)?;
                let b_m = mapped(&forward_to_backward, b)?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, upstream)?;
                accumulate_grad(&mut backward, block, &mut grad_map, b_m, upstream)?;
            }
            Op::Sub(a, b) => {
                let a_m = mapped(&forward_to_backward, a)?;
                let b_m = mapped(&forward_to_backward, b)?;
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
                let a_m = mapped(&forward_to_backward, a)?;
                let b_m = mapped(&forward_to_backward, b)?;
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
                let a_m = mapped(&forward_to_backward, a)?;
                let b_m = mapped(&forward_to_backward, b)?;
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
                let a_m = mapped(&forward_to_backward, a)?;
                let b_m = mapped(&forward_to_backward, b)?;
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
                let a_m = mapped(&forward_to_backward, a)?;
                let (_, grad) = backward
                    .add_op(block, Op::ReluBackward(a_m, upstream))
                    .map_err(|err| AutogradError {
                        message: format!("Failed to build relu backward op: {}", err.message),
                    })?;
                accumulate_grad(&mut backward, block, &mut grad_map, a_m, grad)?;
            }
            Op::Output(v) => {
                let v_m = mapped(&forward_to_backward, v)?;
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
}
