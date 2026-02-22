use std::collections::HashMap;

use crate::ir::tensor::Tensor;
use crate::ir::{ElementwiseUnaryOp, Graph, NodeId, Op, ValueId, build_schedule};

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeValue {
    Int(i64),
    Float(f64),
    Tensor { shape: Vec<usize>, data: Vec<f32> },
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionContext {
    pub inputs: HashMap<String, RuntimeValue>,
    pub parameters: HashMap<String, RuntimeValue>,
}

#[derive(Debug, Clone)]
pub struct InterpreterError {
    pub message: String,
    pub node: Option<NodeId>,
}

pub fn execute(graph: &Graph) -> Result<Option<RuntimeValue>, InterpreterError> {
    execute_with_context(graph, &ExecutionContext::default())
}

pub fn execute_with_context(
    graph: &Graph,
    context: &ExecutionContext,
) -> Result<Option<RuntimeValue>, InterpreterError> {
    let schedule = build_schedule(graph).map_err(|err| InterpreterError {
        message: format!("Failed to build schedule: {}", err.message),
        node: None,
    })?;

    if let Some(target) = terminal_value_id(graph) {
        execute_value_with_context_scheduled(graph, target, context, &schedule.ordered_nodes)
            .map(Some)
    } else {
        Ok(None)
    }
}

pub fn execute_value(graph: &Graph, target: ValueId) -> Result<RuntimeValue, InterpreterError> {
    execute_value_with_context(graph, target, &ExecutionContext::default())
}

pub fn execute_value_with_context(
    graph: &Graph,
    target: ValueId,
    context: &ExecutionContext,
) -> Result<RuntimeValue, InterpreterError> {
    let schedule = build_schedule(graph).map_err(|err| InterpreterError {
        message: format!("Failed to build schedule: {}", err.message),
        node: None,
    })?;
    execute_value_with_context_scheduled(graph, target, context, &schedule.ordered_nodes)
}

pub fn execute_with_schedule_context(
    graph: &Graph,
    ordered_nodes: &[NodeId],
    context: &ExecutionContext,
) -> Result<Option<RuntimeValue>, InterpreterError> {
    if let Some(target) = terminal_value_id(graph) {
        execute_value_with_context_scheduled(graph, target, context, ordered_nodes).map(Some)
    } else {
        Ok(None)
    }
}

pub fn execute_value_with_schedule_context(
    graph: &Graph,
    target: ValueId,
    ordered_nodes: &[NodeId],
    context: &ExecutionContext,
) -> Result<RuntimeValue, InterpreterError> {
    execute_value_with_context_scheduled(graph, target, context, ordered_nodes)
}

fn execute_value_with_context_scheduled(
    graph: &Graph,
    target: ValueId,
    context: &ExecutionContext,
    ordered_nodes: &[NodeId],
) -> Result<RuntimeValue, InterpreterError> {
    let mut values = vec![None; graph.value_count()];

    for node_id in ordered_nodes {
        let node = &graph.nodes[node_id.0];
        let output_index = node.output.0;
        if output_index >= values.len() {
            return Err(error(
                format!("Output ValueId out of range: {}", output_index),
                Some(node.id),
            ));
        }

        if values[output_index].is_some() {
            return Err(error(
                format!(
                    "SSA violation: ValueId {} assigned more than once",
                    output_index
                ),
                Some(node.id),
            ));
        }

        if matches!(node.op, Op::Removed) {
            continue;
        }

        let computed = evaluate_op(&node.op, &values, node.id, context)?;
        values[output_index] = Some(computed);
    }

    read_value(&values, target, None)
}

fn terminal_value_id(graph: &Graph) -> Option<ValueId> {
    for node in graph.nodes.iter().rev() {
        if let Op::Output(value) = node.op {
            return Some(value);
        }
    }
    graph.last_value_id()
}

fn evaluate_op(
    op: &Op,
    values: &[Option<RuntimeValue>],
    node_id: NodeId,
    context: &ExecutionContext,
) -> Result<RuntimeValue, InterpreterError> {
    match op {
        Op::ConstInt(value) => Ok(RuntimeValue::Int(*value)),
        Op::ConstFloat(value) => Ok(RuntimeValue::Float(*value)),
        Op::ConstTensor { shape, data } => {
            let tensor = Tensor::new(shape.clone(), data.clone())
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(tensor))
        }
        Op::Parameter(name) => context
            .parameters
            .get(name)
            .cloned()
            .ok_or_else(|| error(format!("Missing parameter: '{name}'"), Some(node_id))),
        Op::Input(name) => context
            .inputs
            .get(name)
            .cloned()
            .ok_or_else(|| error(format!("Missing input: '{name}'"), Some(node_id))),
        Op::Output(value) => read_value(values, *value, Some(node_id)),
        Op::Add(left, right) => {
            let left_value = read_value(values, *left, Some(node_id))?;
            let right_value = read_value(values, *right, Some(node_id))?;
            add_values(left_value, right_value, node_id)
        }
        Op::Sub(left, right) => {
            let left_value = read_value(values, *left, Some(node_id))?;
            let right_value = read_value(values, *right, Some(node_id))?;
            sub_values(left_value, right_value, node_id)
        }
        Op::Mul(left, right) => {
            let left_value = read_value(values, *left, Some(node_id))?;
            let right_value = read_value(values, *right, Some(node_id))?;
            mul_values(left_value, right_value, node_id)
        }
        Op::Div(left, right) => {
            let left_value = read_value(values, *left, Some(node_id))?;
            let right_value = read_value(values, *right, Some(node_id))?;
            div_values(left_value, right_value, node_id)
        }
        Op::Neg(value) => {
            let value = read_value(values, *value, Some(node_id))?;
            neg_value(value, node_id)
        }
        Op::ElementwiseChain { input, ops } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let out = apply_elementwise_chain(input_tensor, ops, node_id)?;
            Ok(runtime_from_tensor(out))
        }
        Op::Reshape { input, shape } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let out = input_tensor
                .reshape(shape.clone())
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Concat { inputs, axis } => {
            if inputs.len() < 2 {
                return Err(error(
                    "concat requires at least two inputs".to_string(),
                    Some(node_id),
                ));
            }
            let mut tensors = Vec::with_capacity(inputs.len());
            for input in inputs {
                tensors.push(read_tensor(values, *input, node_id)?);
            }
            let out =
                Tensor::concat(&tensors, *axis).map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Gather {
            input,
            indices,
            axis,
        } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let out = input_tensor
                .gather(indices, *axis)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Slice {
            input,
            starts,
            ends,
            axes,
        } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let out = input_tensor
                .slice(starts, ends, axes)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Transpose(value) => {
            let value = read_tensor(values, *value, node_id)?;
            let transposed = value
                .transpose_2d()
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(transposed))
        }
        Op::MatMul(left, right) => {
            let lhs = read_tensor(values, *left, node_id)?;
            let rhs = read_tensor(values, *right, node_id)?;
            let out = lhs
                .matmul(&rhs)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Relu(value) => {
            let tensor = read_tensor(values, *value, node_id)?;
            let out = tensor
                .relu()
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::ReluBackward(input, grad) => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let grad_tensor = read_tensor(values, *grad, node_id)?;
            let out = input_tensor
                .relu_backward(&grad_tensor)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Softmax(value) => {
            let tensor = read_tensor(values, *value, node_id)?;
            softmax(tensor, node_id)
        }
        Op::Log(value) => {
            let tensor = read_tensor(values, *value, node_id)?;
            let out = tensor
                .log_elementwise()
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Exp(value) => {
            let tensor = read_tensor(values, *value, node_id)?;
            let out = tensor
                .exp_elementwise()
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::ReduceSum { input, axis } => {
            let tensor = read_tensor(values, *input, node_id)?;
            let out = tensor
                .reduce_sum(*axis)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Conv2D(input, weight) => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let weight_tensor = read_tensor(values, *weight, node_id)?;
            conv2d(input_tensor, weight_tensor, node_id)
        }
        Op::Phi(_) => Err(error(
            "Unsupported op in phase 1: phi".to_string(),
            Some(node_id),
        )),
        Op::Removed => Err(error(
            "Removed node cannot be executed".to_string(),
            Some(node_id),
        )),
    }
}

fn read_tensor(
    values: &[Option<RuntimeValue>],
    id: ValueId,
    node: NodeId,
) -> Result<Tensor, InterpreterError> {
    match read_value(values, id, Some(node))? {
        RuntimeValue::Tensor { shape, data } => {
            Tensor::new(shape, data).map_err(|err| error(err.message, Some(node)))
        }
        _ => Err(error(
            "Type mismatch: expected tensor input".to_string(),
            Some(node),
        )),
    }
}

fn softmax(tensor: Tensor, node: NodeId) -> Result<RuntimeValue, InterpreterError> {
    if tensor.shape.len() != 1 {
        return Err(error(
            format!("Softmax expects 1D tensor, got shape {:?}", tensor.shape),
            Some(node),
        ));
    }
    if tensor.data.is_empty() {
        return Err(error(
            "Softmax expects non-empty tensor".to_string(),
            Some(node),
        ));
    }

    let max = tensor
        .data
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut exps = Vec::with_capacity(tensor.data.len());
    let mut sum = 0.0_f32;
    for value in tensor.data {
        let exp_value = (value - max).exp();
        exps.push(exp_value);
        sum += exp_value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(error(
            "Softmax numeric instability: invalid sum".to_string(),
            Some(node),
        ));
    }

    // Normalize in-place — reuse the `exps` Vec instead of allocating a second
    // one for the divided values.
    for v in &mut exps {
        *v /= sum;
    }
    Ok(RuntimeValue::Tensor {
        shape: tensor.shape,
        data: exps,
    })
}

fn conv2d(input: Tensor, weight: Tensor, node: NodeId) -> Result<RuntimeValue, InterpreterError> {
    if input.shape.len() != 2 || weight.shape.len() != 2 {
        return Err(error(
            "Conv2D expects 2D input and 2D kernel".to_string(),
            Some(node),
        ));
    }

    let in_h = input.shape[0];
    let in_w = input.shape[1];
    let k_h = weight.shape[0];
    let k_w = weight.shape[1];
    if k_h == 0 || k_w == 0 || k_h > in_h || k_w > in_w {
        return Err(error(
            format!(
                "Shape mismatch in Conv2D: input {:?}, kernel {:?}",
                input.shape, weight.shape
            ),
            Some(node),
        ));
    }

    let out_h = in_h - k_h + 1;
    let out_w = in_w - k_w + 1;
    let mut out = vec![0.0_f32; out_h * out_w];
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut acc = 0.0_f32;
            for ky in 0..k_h {
                for kx in 0..k_w {
                    let in_index = (oy + ky) * in_w + (ox + kx);
                    let kernel_index = ky * k_w + kx;
                    acc += input.data[in_index] * weight.data[kernel_index];
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }

    Ok(RuntimeValue::Tensor {
        shape: vec![out_h, out_w],
        data: out,
    })
}

fn apply_elementwise_chain(
    mut tensor: Tensor,
    ops: &[ElementwiseUnaryOp],
    node: NodeId,
) -> Result<Tensor, InterpreterError> {
    for op in ops {
        tensor = match op {
            ElementwiseUnaryOp::Neg => tensor
                .scale(-1.0)
                .map_err(|err| error(err.message, Some(node)))?,
            ElementwiseUnaryOp::Relu => tensor
                .relu()
                .map_err(|err| error(err.message, Some(node)))?,
        };
    }
    Ok(tensor)
}

fn read_value(
    values: &[Option<RuntimeValue>],
    id: ValueId,
    node: Option<NodeId>,
) -> Result<RuntimeValue, InterpreterError> {
    values
        .get(id.0)
        .ok_or_else(|| error(format!("ValueId out of range: {}", id.0), node))?
        .clone()
        .ok_or_else(|| error(format!("ValueId {} not computed", id.0), node))
}

fn add_values(
    left: RuntimeValue,
    right: RuntimeValue,
    node: NodeId,
) -> Result<RuntimeValue, InterpreterError> {
    match (left, right) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => a
            .checked_add(b)
            .map(RuntimeValue::Int)
            .ok_or_else(|| error("Integer overflow in add".to_string(), Some(node))),
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
            finite_float(a + b, "Float overflow in add", node)
        }
        (
            RuntimeValue::Tensor {
                shape: s1,
                data: d1,
            },
            RuntimeValue::Tensor {
                shape: s2,
                data: d2,
            },
        ) => {
            if s1 != s2 {
                return Err(error(
                    format!("Shape mismatch in add: {:?} vs {:?}", s1, s2),
                    Some(node),
                ));
            }
            let mut out = Vec::with_capacity(d1.len());
            for (a, b) in d1.iter().zip(d2.iter()) {
                out.push(*a + *b);
            }
            Ok(RuntimeValue::Tensor {
                shape: s1,
                data: out,
            })
        }
        _ => Err(error(
            "Type mismatch in add: expected matching numeric types".to_string(),
            Some(node),
        )),
    }
}

fn sub_values(
    left: RuntimeValue,
    right: RuntimeValue,
    node: NodeId,
) -> Result<RuntimeValue, InterpreterError> {
    match (left, right) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => a
            .checked_sub(b)
            .map(RuntimeValue::Int)
            .ok_or_else(|| error("Integer overflow in sub".to_string(), Some(node))),
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
            finite_float(a - b, "Float overflow in sub", node)
        }
        (
            RuntimeValue::Tensor {
                shape: s1,
                data: d1,
            },
            RuntimeValue::Tensor {
                shape: s2,
                data: d2,
            },
        ) => {
            if s1 != s2 {
                return Err(error(
                    format!("Shape mismatch in sub: {:?} vs {:?}", s1, s2),
                    Some(node),
                ));
            }
            let mut out = Vec::with_capacity(d1.len());
            for (a, b) in d1.iter().zip(d2.iter()) {
                out.push(*a - *b);
            }
            Ok(RuntimeValue::Tensor {
                shape: s1,
                data: out,
            })
        }
        _ => Err(error(
            "Type mismatch in sub: expected matching numeric types".to_string(),
            Some(node),
        )),
    }
}

fn mul_values(
    left: RuntimeValue,
    right: RuntimeValue,
    node: NodeId,
) -> Result<RuntimeValue, InterpreterError> {
    match (left, right) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => a
            .checked_mul(b)
            .map(RuntimeValue::Int)
            .ok_or_else(|| error("Integer overflow in mul".to_string(), Some(node))),
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
            finite_float(a * b, "Float overflow in mul", node)
        }
        (
            RuntimeValue::Tensor {
                shape: s1,
                data: d1,
            },
            RuntimeValue::Tensor {
                shape: s2,
                data: d2,
            },
        ) => {
            if s1 != s2 {
                return Err(error(
                    format!("Shape mismatch in mul: {:?} vs {:?}", s1, s2),
                    Some(node),
                ));
            }
            let mut out = Vec::with_capacity(d1.len());
            for (a, b) in d1.iter().zip(d2.iter()) {
                out.push(*a * *b);
            }
            Ok(RuntimeValue::Tensor {
                shape: s1,
                data: out,
            })
        }
        _ => Err(error(
            "Type mismatch in mul: expected matching numeric types".to_string(),
            Some(node),
        )),
    }
}

fn div_values(
    left: RuntimeValue,
    right: RuntimeValue,
    node: NodeId,
) -> Result<RuntimeValue, InterpreterError> {
    match (left, right) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => {
            if b == 0 {
                return Err(error("Division by zero".to_string(), Some(node)));
            }
            a.checked_div(b)
                .map(RuntimeValue::Int)
                .ok_or_else(|| error("Integer overflow in div".to_string(), Some(node)))
        }
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
            if b == 0.0 {
                return Err(error("Division by zero".to_string(), Some(node)));
            }
            finite_float(a / b, "Float overflow in div", node)
        }
        (
            RuntimeValue::Tensor {
                shape: s1,
                data: d1,
            },
            RuntimeValue::Tensor {
                shape: s2,
                data: d2,
            },
        ) => {
            if s1 != s2 {
                return Err(error(
                    format!("Shape mismatch in div: {:?} vs {:?}", s1, s2),
                    Some(node),
                ));
            }
            let mut out = Vec::with_capacity(d1.len());
            for (a, b) in d1.iter().zip(d2.iter()) {
                if *b == 0.0 {
                    return Err(error("Division by zero".to_string(), Some(node)));
                }
                out.push(*a / *b);
            }
            Ok(RuntimeValue::Tensor {
                shape: s1,
                data: out,
            })
        }
        _ => Err(error(
            "Type mismatch in div: expected matching numeric types".to_string(),
            Some(node),
        )),
    }
}

fn neg_value(value: RuntimeValue, node: NodeId) -> Result<RuntimeValue, InterpreterError> {
    match value {
        RuntimeValue::Int(v) => v
            .checked_neg()
            .map(RuntimeValue::Int)
            .ok_or_else(|| error("Integer overflow in neg".to_string(), Some(node))),
        RuntimeValue::Float(v) => finite_float(-v, "Float overflow in neg", node),
        RuntimeValue::Tensor { shape, mut data } => {
            // Negate in-place: avoids a full heap allocation for just flipping
            // the sign bit on every element.
            for v in &mut data {
                *v = -*v;
            }
            Ok(RuntimeValue::Tensor { shape, data })
        }
    }
}

fn runtime_from_tensor(tensor: Tensor) -> RuntimeValue {
    RuntimeValue::Tensor {
        shape: tensor.shape,
        data: tensor.data,
    }
}

fn finite_float(value: f64, message: &str, node: NodeId) -> Result<RuntimeValue, InterpreterError> {
    if value.is_finite() {
        Ok(RuntimeValue::Float(value))
    } else {
        Err(error(message.to_string(), Some(node)))
    }
}

fn error(message: String, node: Option<NodeId>) -> InterpreterError {
    InterpreterError { message, node }
}

#[cfg(test)]
mod tests {
    use crate::ir::{
        ExecutionContext, Graph, Op, RuntimeValue, ValueId, execute, execute_value,
        execute_with_context,
    };

    #[test]
    fn executes_linear_integer_arithmetic() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        let (_, v2) = graph
            .add_op(block, Op::Add(v0, v1))
            .expect("add op should succeed");
        let (_, v3) = graph
            .add_op(block, Op::ConstInt(3))
            .expect("add op should succeed");
        let (_, v4) = graph
            .add_op(block, Op::Mul(v2, v3))
            .expect("add op should succeed");
        let (_, v5) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Div(v4, v5))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(result, Some(RuntimeValue::Int(4)));
    }

    #[test]
    fn executes_float_arithmetic() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstFloat(1.5))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstFloat(2.5))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(v0, v1))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(result, Some(RuntimeValue::Float(4.0)));
    }

    #[test]
    fn rejects_mixed_numeric_types() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstFloat(2.0))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(v0, v1))
            .expect("add op should succeed");

        let err = execute(&graph).expect_err("execute must fail");
        assert!(err.message.contains("Type mismatch"));
    }

    #[test]
    fn rejects_division_by_zero() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(8))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstInt(0))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Div(v0, v1))
            .expect("add op should succeed");

        let err = execute(&graph).expect_err("execute must fail");
        assert!(err.message.contains("Division by zero"));
    }

    #[test]
    fn fails_on_uninitialized_input_value() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        graph
            .add_op(block, Op::Add(ValueId(99), ValueId(1)))
            .expect("add op should succeed");

        let err = execute(&graph).expect_err("execute must fail");
        assert!(
            err.message.contains("not computed")
                || err.message.contains("out of range")
                || err.message.contains("missing producer")
        );
    }

    #[test]
    fn fails_on_double_assignment_to_same_value_id() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");

        let duplicate_output = graph.nodes[0].output;
        graph.nodes[1].output = duplicate_output;

        let err = execute(&graph).expect_err("execute must fail");
        assert!(err.message.contains("assigned more than once"));
    }

    #[test]
    fn execute_value_can_read_non_terminal_result() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(5))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstInt(7))
            .expect("add op should succeed");
        let (_, add_value) = graph
            .add_op(block, Op::Add(v0, v1))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Mul(add_value, v1))
            .expect("add op should succeed");

        let result = execute_value(&graph, add_value).expect("execute_value should succeed");
        assert_eq!(result, RuntimeValue::Int(12));
    }

    #[test]
    fn executes_small_matmul_2x2() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, left) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2, 2],
                    data: vec![1.0, 2.0, 3.0, 4.0],
                },
            )
            .expect("add op should succeed");
        let (_, right) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2, 2],
                    data: vec![5.0, 6.0, 7.0, 8.0],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::MatMul(left, right))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(
            result,
            Some(RuntimeValue::Tensor {
                shape: vec![2, 2],
                data: vec![19.0, 22.0, 43.0, 50.0]
            })
        );
    }

    #[test]
    fn relu_clamps_negative_values() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![4],
                    data: vec![-2.0, -0.5, 1.0, 3.5],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Relu(input))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(
            result,
            Some(RuntimeValue::Tensor {
                shape: vec![4],
                data: vec![0.0, 0.0, 1.0, 3.5]
            })
        );
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![3],
                    data: vec![1.0, 2.0, 3.0],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Softmax(input))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        let RuntimeValue::Tensor { data, .. } = result.expect("output exists") else {
            panic!("expected tensor");
        };

        let sum = data.iter().copied().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn conv2d_minimal_example() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![3, 3],
                    data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                },
            )
            .expect("add op should succeed");
        let (_, kernel) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2, 2],
                    data: vec![1.0, 0.0, 0.0, -1.0],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Conv2D(input, kernel))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(
            result,
            Some(RuntimeValue::Tensor {
                shape: vec![2, 2],
                data: vec![-4.0, -4.0, -4.0, -4.0]
            })
        );
    }

    #[test]
    fn tensor_shape_mismatch_reports_error() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, left) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2, 3],
                    data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                },
            )
            .expect("add op should succeed");
        let (_, right) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2, 2],
                    data: vec![1.0, 2.0, 3.0, 4.0],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::MatMul(left, right))
            .expect("add op should succeed");

        let err = execute(&graph).expect_err("execute must fail");
        assert!(err.message.contains("Shape mismatch"));
    }

    #[test]
    fn input_and_output_ops_use_execution_context() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Output(input))
            .expect("add op should succeed");

        let mut context = ExecutionContext::default();
        context.inputs.insert(
            "x".to_string(),
            RuntimeValue::Tensor {
                shape: vec![1],
                data: vec![42.0],
            },
        );

        let result = execute_with_context(&graph, &context).expect("execute should pass");
        assert_eq!(
            result,
            Some(RuntimeValue::Tensor {
                shape: vec![1],
                data: vec![42.0],
            })
        );
    }
}
