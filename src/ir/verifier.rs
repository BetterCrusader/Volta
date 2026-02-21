use std::collections::{HashMap, HashSet};

use crate::ir::{
    CompilerFlags, Graph, Op, Pass, ValueId, build_schedule, infer_shapes, plan_allocation,
    verify_allocation, verify_schedule,
};

#[derive(Debug, Clone)]
pub struct VerifyError {
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueType {
    Int,
    Float,
    Tensor,
    Unknown,
}

pub fn verify_graph(graph: &Graph) -> Result<(), VerifyError> {
    if graph.blocks.is_empty() {
        return Err(err("Graph must contain at least one basic block"));
    }

    let mut seen_node_ids = HashSet::new();
    let mut block_node_set = HashSet::new();
    for block in &graph.blocks {
        for node_id in &block.nodes {
            if node_id.0 >= graph.nodes.len() {
                return Err(err(format!(
                    "Block {:?} references missing node id {}",
                    block.id, node_id.0
                )));
            }
            block_node_set.insert(node_id.0);
        }
    }

    let mut value_producer: HashMap<ValueId, usize> = HashMap::new();
    let mut value_types: HashMap<ValueId, ValueType> = HashMap::new();

    for (index, node) in graph.nodes.iter().enumerate() {
        if node.id.0 != index {
            return Err(err(format!(
                "Node id mismatch at index {index}: node.id={} ",
                node.id.0
            )));
        }

        if !seen_node_ids.insert(node.id.0) {
            return Err(err(format!("Duplicate node id {}", node.id.0)));
        }

        if !block_node_set.contains(&node.id.0) {
            return Err(err(format!(
                "Node {} is not referenced by any basic block",
                node.id.0
            )));
        }

        if value_producer.insert(node.output, index).is_some() {
            return Err(err(format!(
                "SSA violation: ValueId {} assigned more than once",
                node.output.0
            )));
        }

        for input in node.op.input_values() {
            let Some(producer_index) = value_producer.get(&input).copied() else {
                return Err(err(format!(
                    "Use-before-def or missing ValueId {} in node {}",
                    input.0, node.id.0
                )));
            };

            if producer_index >= index {
                return Err(err(format!(
                    "ValueId {} used before producer is available in node {}",
                    input.0, node.id.0
                )));
            }

            if matches!(graph.nodes[producer_index].op, Op::Removed) {
                return Err(err(format!(
                    "Node {} uses ValueId {} from removed node {}",
                    node.id.0, input.0, producer_index
                )));
            }
        }

        validate_arity(&node.op, node.id.0)?;

        let inferred = infer_type_for_op(&node.op, &value_types, node.id.0)?;
        value_types.insert(node.output, inferred);
    }

    infer_shapes(graph).map_err(|shape_err| err(shape_err.message))?;

    let schedule = build_schedule(graph).map_err(|sched_err| err(sched_err.message))?;
    verify_schedule(graph, &schedule).map_err(|sched_err| err(sched_err.message))?;

    let gradients = HashSet::new();
    let allocation = plan_allocation(graph, &schedule, &gradients)
        .map_err(|alloc_err| err(alloc_err.message))?;
    verify_allocation(graph, &schedule, &allocation).map_err(|alloc_err| err(alloc_err.message))?;

    Ok(())
}

pub fn run_verified_pass<P: Pass>(pass: &mut P, graph: &mut Graph) -> Result<(), VerifyError> {
    verify_graph(graph)?;
    pass.run(graph);
    verify_graph(graph)
}

pub fn verify_with_policy(graph: &Graph) -> Result<(), VerifyError> {
    let flags = CompilerFlags::from_env();
    if !flags.strict {
        return Ok(());
    }

    let result = verify_graph(graph);
    if flags.debug_verify
        && cfg!(debug_assertions)
        && let Err(err) = &result
    {
        panic!(
            "IR verification failed in debug-verify mode: {}",
            err.message
        );
    }
    result
}

fn infer_type_for_op(
    op: &Op,
    value_types: &HashMap<ValueId, ValueType>,
    node_id: usize,
) -> Result<ValueType, VerifyError> {
    match op {
        Op::ConstInt(_) => Ok(ValueType::Int),
        Op::ConstFloat(_) => Ok(ValueType::Float),
        Op::ConstTensor { .. } => Ok(ValueType::Tensor),
        Op::Parameter(_) | Op::Input(_) => Ok(ValueType::Unknown),
        Op::Output(value) => Ok(type_of(*value, value_types)),
        Op::Add(left, right)
        | Op::Sub(left, right)
        | Op::Mul(left, right)
        | Op::Div(left, right) => {
            let left_type = type_of(*left, value_types);
            let right_type = type_of(*right, value_types);
            require_same_strict(left_type, right_type, node_id, "binary op")
        }
        Op::Neg(value) => {
            let ty = type_of(*value, value_types);
            match ty {
                ValueType::Int | ValueType::Float | ValueType::Tensor | ValueType::Unknown => {
                    Ok(ty)
                }
            }
        }
        Op::ElementwiseChain { input, .. } => {
            let ty = type_of(*input, value_types);
            require_tensor_or_unknown(ty, node_id, "elementwise_chain")
        }
        Op::Reshape { input, .. } | Op::Gather { input, .. } | Op::Slice { input, .. } => {
            let ty = type_of(*input, value_types);
            require_tensor_or_unknown(ty, node_id, "tensor unary transform")
        }
        Op::Concat { inputs, .. } => {
            let mut current = ValueType::Unknown;
            for value in inputs {
                let next = type_of(*value, value_types);
                current = unify_types(current, next, node_id, "concat")?;
            }
            require_tensor_or_unknown(current, node_id, "concat")
        }
        Op::Transpose(value) | Op::Relu(value) | Op::Softmax(value) => {
            let ty = type_of(*value, value_types);
            require_tensor_or_unknown(ty, node_id, "tensor unary op")
        }
        Op::ReluBackward(input, grad) | Op::MatMul(input, grad) | Op::Conv2D(input, grad) => {
            let left = type_of(*input, value_types);
            let right = type_of(*grad, value_types);
            let same = require_same_strict(left, right, node_id, "tensor binary op")?;
            require_tensor_or_unknown(same, node_id, "tensor binary op")
        }
        Op::Phi(values) => {
            let mut current = ValueType::Unknown;
            for value in values {
                let next = type_of(*value, value_types);
                current = unify_types(current, next, node_id, "phi")?;
            }
            Ok(current)
        }
        Op::Removed => Ok(ValueType::Unknown),
    }
}

fn validate_arity(op: &Op, node_id: usize) -> Result<(), VerifyError> {
    match op {
        Op::Phi(values) => {
            if values.is_empty() {
                return Err(err(format!(
                    "Invalid arity at node {node_id}: phi requires at least one input"
                )));
            }
        }
        Op::ElementwiseChain { ops, .. } => {
            if ops.is_empty() {
                return Err(err(format!(
                    "Invalid arity at node {node_id}: elementwise chain requires at least one op"
                )));
            }
        }
        Op::Concat { inputs, .. } => {
            if inputs.len() < 2 {
                return Err(err(format!(
                    "Invalid arity at node {node_id}: concat requires at least two inputs"
                )));
            }
        }
        Op::Gather { indices, .. } => {
            if indices.is_empty() {
                return Err(err(format!(
                    "Invalid arity at node {node_id}: gather requires at least one index"
                )));
            }
        }
        Op::Slice {
            starts, ends, axes, ..
        } => {
            if starts.is_empty() || ends.is_empty() || axes.is_empty() {
                return Err(err(format!(
                    "Invalid arity at node {node_id}: slice starts/ends/axes must be non-empty"
                )));
            }
            if starts.len() != ends.len() || starts.len() != axes.len() {
                return Err(err(format!(
                    "Invalid arity at node {node_id}: slice starts/ends/axes lengths must match"
                )));
            }
        }
        _ => {}
    }
    Ok(())
}

fn require_same_strict(
    left: ValueType,
    right: ValueType,
    node_id: usize,
    label: &str,
) -> Result<ValueType, VerifyError> {
    match (left, right) {
        (ValueType::Unknown, x) | (x, ValueType::Unknown) => Ok(x),
        (a, b) if a == b => Ok(a),
        _ => Err(err(format!(
            "Type mismatch in {label} at node {node_id}: left={left:?}, right={right:?}"
        ))),
    }
}

fn require_tensor_or_unknown(
    ty: ValueType,
    node_id: usize,
    label: &str,
) -> Result<ValueType, VerifyError> {
    match ty {
        ValueType::Tensor | ValueType::Unknown => Ok(ty),
        _ => Err(err(format!(
            "Type mismatch in {label} at node {node_id}: expected tensor, got {ty:?}"
        ))),
    }
}

fn unify_types(
    current: ValueType,
    next: ValueType,
    node_id: usize,
    label: &str,
) -> Result<ValueType, VerifyError> {
    match (current, next) {
        (ValueType::Unknown, x) | (x, ValueType::Unknown) => Ok(x),
        (a, b) if a == b => Ok(a),
        _ => Err(err(format!(
            "Type mismatch in {label} at node {node_id}: {current:?} vs {next:?}"
        ))),
    }
}

fn type_of(value: ValueId, value_types: &HashMap<ValueId, ValueType>) -> ValueType {
    value_types
        .get(&value)
        .copied()
        .unwrap_or(ValueType::Unknown)
}

fn err(message: impl Into<String>) -> VerifyError {
    VerifyError {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, verify_graph};

    #[test]
    fn rejects_use_before_definition() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        graph
            .add_op(
                block,
                Op::Add(crate::ir::ValueId(42), crate::ir::ValueId(0)),
            )
            .expect("add op should succeed");

        let err = verify_graph(&graph).expect_err("must fail verifier");
        assert!(err.message.contains("Use-before-def"));
    }

    #[test]
    fn rejects_mixed_numeric_types() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstFloat(1.0))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(a, b))
            .expect("add op should succeed");

        let err = verify_graph(&graph).expect_err("must fail verifier");
        assert!(err.message.contains("Type mismatch"));
    }

    #[test]
    fn accepts_valid_simple_graph() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        let (_, c) = graph
            .add_op(block, Op::Add(a, b))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Output(c))
            .expect("add op should succeed");

        verify_graph(&graph).expect("verifier must pass");
    }

    #[test]
    fn rejects_tensor_shape_mismatch() {
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

        let err = verify_graph(&graph).expect_err("must fail verifier");
        assert!(err.message.contains("Shape mismatch in MatMul"));
    }

    #[test]
    fn rejects_empty_elementwise_chain() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![1],
                    data: vec![1.0],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(
                block,
                Op::ElementwiseChain {
                    input,
                    ops: Vec::new(),
                },
            )
            .expect("add op should succeed");

        let err = verify_graph(&graph).expect_err("must fail verifier");
        assert!(err.message.contains("Invalid arity"));
    }
}
