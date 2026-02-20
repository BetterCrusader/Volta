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
        let inferred = infer_shape_for_op(&node.op, &shapes).map_err(|message| ShapeError {
            message: format!("{} (at node {})", message, node.id.0),
        })?;
        shapes.insert(node.output, inferred);
    }

    Ok(shapes)
}

fn infer_shape_for_op(op: &Op, shapes: &HashMap<ValueId, ShapeFact>) -> Result<ShapeFact, String> {
    match op {
        Op::ConstInt(_) | Op::ConstFloat(_) => Ok(ShapeFact::NonTensor),
        Op::ConstTensor { shape, .. } => Ok(ShapeFact::Tensor(shape.clone())),
        Op::Parameter(_) | Op::Input(_) => Ok(ShapeFact::Unknown),
        Op::Output(value) => Ok(shape_of(*value, shapes)),
        Op::Add(left, right)
        | Op::Sub(left, right)
        | Op::Mul(left, right)
        | Op::Div(left, right) => infer_elementwise(*left, *right, shapes),
        Op::Neg(value) => Ok(shape_of(*value, shapes)),
        Op::ElementwiseChain { input, .. } => infer_tensor_unary(*input, shapes),
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
        Op::Relu(value) | Op::Softmax(value) => infer_tensor_unary(*value, shapes),
        Op::ReluBackward(input, grad) => infer_same_tensor(*input, *grad, shapes, "relu_backward"),
        Op::Conv2D(input, weight) => infer_conv2d(*input, *weight, shapes),
        Op::Phi(values) => infer_phi(values, shapes),
        Op::Removed => Ok(ShapeFact::Unknown),
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
            if a == b {
                Ok(ShapeFact::Tensor(a))
            } else {
                Err(format!(
                    "Shape mismatch in elementwise op: {:?} vs {:?}",
                    a, b
                ))
            }
        }
        (ShapeFact::Unknown, x) | (x, ShapeFact::Unknown) => Ok(x),
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
        (ShapeFact::Unknown, ShapeFact::Tensor(b)) | (ShapeFact::Tensor(b), ShapeFact::Unknown) => {
            Ok(ShapeFact::Tensor(b))
        }
        (ShapeFact::Unknown, ShapeFact::Unknown) => Ok(ShapeFact::Unknown),
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
        (ShapeFact::Unknown, x) | (x, ShapeFact::Unknown) => Ok(x),
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

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, ShapeFact, infer_shapes};

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
}
