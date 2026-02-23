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
        Op::SigmoidBackward(input, grad) | Op::GeluBackward(input, grad) => {
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
            if let Some(out_shape) = broadcast_shapes(&a, &b) {
                Ok(ShapeFact::Tensor(out_shape))
            } else {
                Err(format!("Shape mismatch in elementwise op: {a:?} vs {b:?}"))
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
