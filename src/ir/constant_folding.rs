use crate::ir::{Graph, Op, Pass, ValueId, run_with_verifier_guard};

#[derive(Debug, Clone, Copy)]
enum ConstValue {
    Int(i64),
    Float(f64),
}

#[derive(Default)]
pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    pub fn new() -> Self {
        Self
    }
}

impl Pass for ConstantFoldingPass {
    fn run(&mut self, graph: &mut Graph) {
        run_with_verifier_guard(graph, |graph| {
            let mut constants = vec![None; graph.value_count()];

            for node in &mut graph.nodes {
                let folded = fold_op(&node.op, &constants);
                if let Some(replacement) = folded {
                    node.op = replacement;
                }

                constants[node.output.0] = const_from_op(&node.op);
            }
        });
    }
}

fn fold_op(op: &Op, constants: &[Option<ConstValue>]) -> Option<Op> {
    match op {
        Op::Add(left, right) => fold_binary(*left, *right, constants, fold_add),
        Op::Sub(left, right) => fold_binary(*left, *right, constants, fold_sub),
        Op::Mul(left, right) => fold_binary(*left, *right, constants, fold_mul),
        Op::Div(left, right) => fold_binary(*left, *right, constants, fold_div),
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
        | Op::MatMul(_, _)
        | Op::Relu(_)
        | Op::ReluBackward(_, _)
        | Op::Softmax(_)
        | Op::Log(_)
        | Op::Exp(_)
        | Op::ReduceSum { .. }
        | Op::Conv2D(_, _)
        | Op::Parameter(_)
        | Op::Input(_)
        | Op::Output(_)
        | Op::Phi(_)
        | Op::Removed => None,
    }
}

fn fold_binary(
    left: ValueId,
    right: ValueId,
    constants: &[Option<ConstValue>],
    fold: fn(ConstValue, ConstValue) -> Option<ConstValue>,
) -> Option<Op> {
    let left_const = constants.get(left.0).and_then(|value| *value)?;
    let right_const = constants.get(right.0).and_then(|value| *value)?;
    let folded = fold(left_const, right_const)?;
    Some(op_from_const(folded))
}

fn fold_add(left: ConstValue, right: ConstValue) -> Option<ConstValue> {
    match (left, right) {
        (ConstValue::Int(a), ConstValue::Int(b)) => a.checked_add(b).map(ConstValue::Int),
        (ConstValue::Float(a), ConstValue::Float(b)) => finite_float(a + b),
        _ => None,
    }
}

fn fold_sub(left: ConstValue, right: ConstValue) -> Option<ConstValue> {
    match (left, right) {
        (ConstValue::Int(a), ConstValue::Int(b)) => a.checked_sub(b).map(ConstValue::Int),
        (ConstValue::Float(a), ConstValue::Float(b)) => finite_float(a - b),
        _ => None,
    }
}

fn fold_mul(left: ConstValue, right: ConstValue) -> Option<ConstValue> {
    match (left, right) {
        (ConstValue::Int(a), ConstValue::Int(b)) => a.checked_mul(b).map(ConstValue::Int),
        (ConstValue::Float(a), ConstValue::Float(b)) => finite_float(a * b),
        _ => None,
    }
}

fn fold_div(left: ConstValue, right: ConstValue) -> Option<ConstValue> {
    match (left, right) {
        (ConstValue::Int(a), ConstValue::Int(b)) => {
            if b == 0 {
                return None;
            }
            a.checked_div(b).map(ConstValue::Int)
        }
        (ConstValue::Float(a), ConstValue::Float(b)) => {
            if b == 0.0 {
                return None;
            }
            finite_float(a / b)
        }
        _ => None,
    }
}

fn finite_float(value: f64) -> Option<ConstValue> {
    if value.is_finite() {
        Some(ConstValue::Float(value))
    } else {
        None
    }
}

fn const_from_op(op: &Op) -> Option<ConstValue> {
    match op {
        Op::ConstInt(value) => Some(ConstValue::Int(*value)),
        Op::ConstFloat(value) => Some(ConstValue::Float(*value)),
        Op::Add(_, _)
        | Op::Sub(_, _)
        | Op::Mul(_, _)
        | Op::Div(_, _)
        | Op::ConstTensor { .. }
        | Op::Neg(_)
        | Op::ElementwiseChain { .. }
        | Op::Reshape { .. }
        | Op::Concat { .. }
        | Op::Gather { .. }
        | Op::Slice { .. }
        | Op::Transpose(_)
        | Op::MatMul(_, _)
        | Op::Relu(_)
        | Op::ReluBackward(_, _)
        | Op::Softmax(_)
        | Op::Log(_)
        | Op::Exp(_)
        | Op::ReduceSum { .. }
        | Op::Conv2D(_, _)
        | Op::Parameter(_)
        | Op::Input(_)
        | Op::Output(_)
        | Op::Phi(_)
        | Op::Removed => None,
    }
}

fn op_from_const(value: ConstValue) -> Op {
    match value {
        ConstValue::Int(value) => Op::ConstInt(value),
        ConstValue::Float(value) => Op::ConstFloat(value),
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{
        ConstantFoldingPass, Graph, Op, Pass, RuntimeValue, execute, execute_value, print_graph,
    };

    #[test]
    fn folds_constant_arithmetic_without_changing_result() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(10))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        let (_, v2) = graph
            .add_op(block, Op::Div(v0, v1))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(v2, v1))
            .expect("add op should succeed");

        let before = execute(&graph).expect("execute should succeed");

        let mut pass = ConstantFoldingPass::new();
        pass.run(&mut graph);

        let after = execute(&graph).expect("execute should succeed");
        assert_eq!(before, after);

        let printed = print_graph(&graph);
        assert!(printed.contains("%2 = const 5"));
        assert!(printed.contains("%3 = const 7"));
    }

    #[test]
    fn does_not_fold_non_constant_operations() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::Input("a".to_string()))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::Input("b".to_string()))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(v0, v1))
            .expect("add op should succeed");

        let mut pass = ConstantFoldingPass::new();
        pass.run(&mut graph);

        assert!(matches!(graph.nodes[2].op, Op::Add(_, _)));
    }

    #[test]
    fn does_not_fold_division_by_zero() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(8))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstInt(0))
            .expect("add op should succeed");
        let (_, div_out) = graph
            .add_op(block, Op::Div(v0, v1))
            .expect("add op should succeed");

        let mut pass = ConstantFoldingPass::new();
        pass.run(&mut graph);

        assert!(matches!(graph.nodes[2].op, Op::Div(_, _)));
        let err = execute_value(&graph, div_out).expect_err("must still fail at runtime");
        assert!(err.message.contains("Division by zero"));
    }

    #[test]
    fn preserves_non_terminal_value_results() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(block, Op::ConstInt(3))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::ConstInt(4))
            .expect("add op should succeed");
        let (_, mid) = graph
            .add_op(block, Op::Mul(a, b))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(mid, a))
            .expect("add op should succeed");

        let before_mid = execute_value(&graph, mid).expect("execute_value should succeed");

        let mut pass = ConstantFoldingPass::new();
        pass.run(&mut graph);

        let after_mid = execute_value(&graph, mid).expect("execute_value should succeed");
        assert_eq!(before_mid, after_mid);
        assert_eq!(after_mid, RuntimeValue::Int(12));
    }
}
