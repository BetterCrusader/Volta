use std::collections::HashMap;

use crate::ir::{Graph, Op, Pass, ValueId, run_with_verifier_guard};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CseKey {
    ConstInt(i64),
    ConstFloat(u64),
    ConstTensor {
        shape: Vec<usize>,
        data_bits: Vec<u32>,
    },
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    Neg(ValueId),
    ElementwiseChain {
        input: ValueId,
        ops: Vec<crate::ir::ElementwiseUnaryOp>,
    },
    Transpose(ValueId),
    MatMul(ValueId, ValueId),
    Relu(ValueId),
    ReluBackward(ValueId, ValueId),
    Softmax(ValueId),
    Conv2D(ValueId, ValueId),
}

#[derive(Default)]
pub struct CsePass;

impl CsePass {
    pub fn new() -> Self {
        Self
    }
}

impl Pass for CsePass {
    fn run(&mut self, graph: &mut Graph) {
        run_with_verifier_guard(graph, |graph| {
            let mut canonical_for_value = vec![None; graph.value_count()];
            let mut existing: HashMap<CseKey, ValueId> = HashMap::new();

            for node in &mut graph.nodes {
                node.op
                    .remap_inputs(|value| resolve_alias(value, &canonical_for_value));

                if matches!(node.op, Op::Removed) {
                    canonical_for_value[node.output.0] = Some(node.output);
                    continue;
                }

                if let Some(key) = key_of(&node.op) {
                    if let Some(previous) = existing.get(&key).copied() {
                        node.op = Op::Removed;
                        canonical_for_value[node.output.0] = Some(previous);
                    } else {
                        existing.insert(key, node.output);
                        canonical_for_value[node.output.0] = Some(node.output);
                    }
                } else {
                    canonical_for_value[node.output.0] = Some(node.output);
                }
            }
        });
    }
}

fn resolve_alias(value: ValueId, canonical_for_value: &[Option<ValueId>]) -> ValueId {
    let mut current = value;
    while let Some(Some(next)) = canonical_for_value.get(current.0) {
        if *next == current {
            break;
        }
        current = *next;
    }
    current
}

fn key_of(op: &Op) -> Option<CseKey> {
    match op {
        Op::ConstInt(value) => Some(CseKey::ConstInt(*value)),
        Op::ConstFloat(value) => Some(CseKey::ConstFloat(value.to_bits())),
        Op::ConstTensor { shape, data } => Some(CseKey::ConstTensor {
            shape: shape.clone(),
            data_bits: data.iter().map(|v| v.to_bits()).collect(),
        }),
        Op::Add(left, right) => Some(CseKey::Add(*left, *right)),
        Op::Sub(left, right) => Some(CseKey::Sub(*left, *right)),
        Op::Mul(left, right) => Some(CseKey::Mul(*left, *right)),
        Op::Div(left, right) => Some(CseKey::Div(*left, *right)),
        Op::Neg(value) => Some(CseKey::Neg(*value)),
        Op::ElementwiseChain { input, ops } => Some(CseKey::ElementwiseChain {
            input: *input,
            ops: ops.clone(),
        }),
        Op::Transpose(value) => Some(CseKey::Transpose(*value)),
        Op::MatMul(left, right) => Some(CseKey::MatMul(*left, *right)),
        Op::Relu(value) => Some(CseKey::Relu(*value)),
        Op::ReluBackward(input, grad) => Some(CseKey::ReluBackward(*input, *grad)),
        Op::Softmax(value) => Some(CseKey::Softmax(*value)),
        Op::Conv2D(input, weight) => Some(CseKey::Conv2D(*input, *weight)),
        Op::Parameter(_) | Op::Input(_) | Op::Output(_) | Op::Phi(_) | Op::Removed => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{CsePass, Graph, Op, Pass, RuntimeValue, execute};

    #[test]
    fn merges_repeated_add_with_same_operands() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(3))
            .expect("add op should succeed");
        let (_, first) = graph
            .add_op(block, Op::Add(a, b))
            .expect("add op should succeed");
        let (_, second) = graph
            .add_op(block, Op::Add(a, b))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(first, second))
            .expect("add op should succeed");

        let mut pass = CsePass::new();
        pass.run(&mut graph);

        assert!(matches!(graph.nodes[3].op, Op::Removed));
        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(result, Some(RuntimeValue::Int(10)));
    }

    #[test]
    fn merges_identical_constants() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(7))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(7))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(a, b))
            .expect("add op should succeed");

        let mut pass = CsePass::new();
        pass.run(&mut graph);

        assert!(matches!(graph.nodes[1].op, Op::Removed));
        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(result, Some(RuntimeValue::Int(14)));
    }

    #[test]
    fn does_not_merge_if_operand_order_differs() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Sub(a, b))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Sub(b, a))
            .expect("add op should succeed");

        let mut pass = CsePass::new();
        pass.run(&mut graph);

        assert!(!matches!(graph.nodes[3].op, Op::Removed));
    }
}
