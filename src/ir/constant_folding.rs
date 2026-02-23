use crate::ir::{Graph, Op, Pass, ValueId, run_with_verifier_guard};

#[derive(Debug, Clone, Copy)]
enum ConstValue {
    Int(i64),
    Float(f64),
}

#[derive(Default)]
pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    #[must_use]
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

/// Try to constant-fold or algebraically simplify an op.
///
/// Returns `Some(replacement_op)` when the op can be simplified, or `None`
/// when no simplification is possible.
fn fold_op(op: &Op, constants: &[Option<ConstValue>]) -> Option<Op> {
    match op {
        Op::Add(left, right) => {
            // Try full constant folding first.
            if let Some(folded) = fold_binary(*left, *right, constants, fold_add) {
                return Some(folded);
            }
            // Algebraic identity: x + 0 → x,  0 + x → x
            identity_add(*left, *right, constants)
        }
        Op::Sub(left, right) => {
            // Try full constant folding first.
            if let Some(folded) = fold_binary(*left, *right, constants, fold_sub) {
                return Some(folded);
            }
            // Algebraic identity: x - 0 → x
            identity_sub(*left, *right, constants)
        }
        Op::Mul(left, right) => {
            // Try full constant folding first.
            if let Some(folded) = fold_binary(*left, *right, constants, fold_mul) {
                return Some(folded);
            }
            // Algebraic identities: x * 1 → x,  1 * x → x,  x * 0 → 0,  0 * x → 0
            identity_mul(*left, *right, constants)
        }
        Op::Div(left, right) => {
            // Try full constant folding first.
            if let Some(folded) = fold_binary(*left, *right, constants, fold_div) {
                return Some(folded);
            }
            // Algebraic identity: x / 1 → x
            identity_div(*left, *right, constants)
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
        | Op::MatMul(_, _)
        | Op::Relu(_)
        | Op::ReluBackward(_, _)
        | Op::Softmax(_)
        | Op::Log(_)
        | Op::Exp(_)
        | Op::Sigmoid(_)
        | Op::SigmoidBackward(_, _)
        | Op::GeluExact(_)
        | Op::Gelu(_)
        | Op::GeluBackward(_, _)
        | Op::Gemm { .. }
        | Op::GemmBackward { .. }
        | Op::ReduceSum { .. }
        | Op::ReduceMax { .. }
        | Op::ReduceMean { .. }
        | Op::Conv2D(_, _)
        | Op::Parameter(_)
        | Op::Input(_)
        | Op::Output(_)
        | Op::Phi(_)
        | Op::Removed => None,
    }
}

// ── Algebraic identity rewrites ──────────────────────────────────────────────

/// `x + 0 → x`,  `0 + x → x`
fn identity_add(left: ValueId, right: ValueId, constants: &[Option<ConstValue>]) -> Option<Op> {
    let left_const = constants.get(left.0).and_then(|v| *v);
    let right_const = constants.get(right.0).and_then(|v| *v);

    if is_zero(right_const) {
        // x + 0 → output the value of x (represented as Output of x)
        return Some(Op::Output(left));
    }
    if is_zero(left_const) {
        // 0 + x → output the value of x
        return Some(Op::Output(right));
    }
    None
}

/// `x - 0 → x`
fn identity_sub(left: ValueId, right: ValueId, constants: &[Option<ConstValue>]) -> Option<Op> {
    let right_const = constants.get(right.0).and_then(|v| *v);
    if is_zero(right_const) {
        return Some(Op::Output(left));
    }
    None
}

/// `x * 1 → x`,  `1 * x → x`,  `x * 0 → 0`,  `0 * x → 0`
fn identity_mul(left: ValueId, right: ValueId, constants: &[Option<ConstValue>]) -> Option<Op> {
    let left_const = constants.get(left.0).and_then(|v| *v);
    let right_const = constants.get(right.0).and_then(|v| *v);

    if is_one(right_const) {
        return Some(Op::Output(left));
    }
    if is_one(left_const) {
        return Some(Op::Output(right));
    }
    if is_zero(right_const) {
        return Some(op_from_const(zero_for(left_const, right_const)));
    }
    if is_zero(left_const) {
        return Some(op_from_const(zero_for(left_const, right_const)));
    }
    None
}

/// `x / 1 → x`
fn identity_div(left: ValueId, right: ValueId, constants: &[Option<ConstValue>]) -> Option<Op> {
    let right_const = constants.get(right.0).and_then(|v| *v);
    if is_one(right_const) {
        return Some(Op::Output(left));
    }
    None
}

#[allow(clippy::float_cmp)]
#[must_use]
fn is_zero(c: Option<ConstValue>) -> bool {
    match c {
        Some(ConstValue::Int(0)) => true,
        Some(ConstValue::Float(v)) => v == 0.0,
        _ => false,
    }
}

#[allow(clippy::float_cmp)]
#[must_use]
fn is_one(c: Option<ConstValue>) -> bool {
    match c {
        Some(ConstValue::Int(1)) => true,
        Some(ConstValue::Float(v)) => v == 1.0,
        _ => false,
    }
}

/// Returns a zero constant of the same numeric kind as the operands.
/// Falls back to Int(0) when neither side is known.
fn zero_for(left: Option<ConstValue>, right: Option<ConstValue>) -> ConstValue {
    match (left, right) {
        (Some(ConstValue::Float(_)), _) | (_, Some(ConstValue::Float(_))) => ConstValue::Float(0.0),
        _ => ConstValue::Int(0),
    }
}

// ── Full constant folding helpers ────────────────────────────────────────────

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
        #[allow(clippy::float_cmp)]
        (ConstValue::Float(a), ConstValue::Float(b)) => {
            if b == 0.0 {
                return Some(ConstValue::Float(a));
            }
            if a == 0.0 {
                return Some(ConstValue::Float(b));
            }
            finite_float(a + b)
        }
        _ => None,
    }
}

fn fold_sub(left: ConstValue, right: ConstValue) -> Option<ConstValue> {
    match (left, right) {
        (ConstValue::Int(a), ConstValue::Int(b)) => a.checked_sub(b).map(ConstValue::Int),
        #[allow(clippy::float_cmp)]
        (ConstValue::Float(a), ConstValue::Float(b)) => {
            if b == 0.0 {
                return Some(ConstValue::Float(a));
            }
            // `0.0 - b` isn't strictly `-b` due to -0.0 semantics,
            // but for typical folds `x - 0.0 -> x` is safe.
            finite_float(a - b)
        }
        _ => None,
    }
}

fn fold_mul(left: ConstValue, right: ConstValue) -> Option<ConstValue> {
    match (left, right) {
        (ConstValue::Int(a), ConstValue::Int(b)) => a.checked_mul(b).map(ConstValue::Int),
        #[allow(clippy::float_cmp)]
        (ConstValue::Float(a), ConstValue::Float(b)) => {
            // Note: x * 0.0 is skipped deliberately (NaN * 0.0 = NaN)
            if b == 1.0 {
                return Some(ConstValue::Float(a));
            }
            if a == 1.0 {
                return Some(ConstValue::Float(b));
            }
            finite_float(a * b)
        }
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
        #[allow(clippy::float_cmp)]
        (ConstValue::Float(a), ConstValue::Float(b)) => {
            if b == 0.0 {
                return None;
            }
            if b == 1.0 {
                return Some(ConstValue::Float(a));
            }
            finite_float(a / b)
        }
        _ => None,
    }
}

#[must_use]
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
        | Op::Sigmoid(_)
        | Op::SigmoidBackward(_, _)
        | Op::GeluExact(_)
        | Op::Gelu(_)
        | Op::GeluBackward(_, _)
        | Op::Gemm { .. }
        | Op::GemmBackward { .. }
        | Op::ReduceSum { .. }
        | Op::ReduceMax { .. }
        | Op::ReduceMean { .. }
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

    // ── Regression tests for algebraic identity rewrites ──────────────────────

    #[test]
    fn folds_add_zero_identity() {
        // x + 0 → x
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::ConstInt(42))
            .expect("add op should succeed");
        let (_, zero) = graph
            .add_op(block, Op::ConstInt(0))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(x, zero))
            .expect("add op should succeed");

        let before = execute(&graph).expect("should succeed");

        let mut pass = ConstantFoldingPass::new();
        pass.run(&mut graph);

        let after = execute(&graph).expect("should succeed after folding");
        assert_eq!(before, after, "x + 0 must equal x");
        // The Add node should be gone (either fully folded or identity-redirected).
        let has_add = graph.nodes.iter().any(|n| matches!(n.op, Op::Add(_, _)));
        // After constant folding, 42+0 should be a single const, so no Add remains.
        assert!(!has_add, "Add node should be eliminated");
    }

    #[test]
    fn folds_mul_one_identity() {
        // x * 1 → x  (both sides)
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::ConstInt(7))
            .expect("add op should succeed");
        let (_, one) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Mul(x, one))
            .expect("add op should succeed");

        let before = execute(&graph).expect("should succeed");

        let mut pass = ConstantFoldingPass::new();
        pass.run(&mut graph);

        let after = execute(&graph).expect("should succeed after folding");
        assert_eq!(before, after, "x * 1 must equal x");
    }

    #[test]
    fn folds_mul_zero_identity() {
        // x * 0 → 0  (with a non-const x to force identity path, not full fold)
        let mut graph = Graph::new();
        let block = graph.create_block();
        // Use Input so x is not a constant → full fold won't fire, identity must.
        let (_, x) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, zero) = graph
            .add_op(block, Op::ConstInt(0))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Mul(x, zero))
            .expect("add op should succeed");

        let mut pass = ConstantFoldingPass::new();
        pass.run(&mut graph);

        // The Mul node should now be a const 0.
        let has_mul = graph.nodes.iter().any(|n| matches!(n.op, Op::Mul(_, _)));
        assert!(
            !has_mul,
            "x * 0 should collapse to const 0, Mul should be gone"
        );
    }

    #[test]
    fn folds_div_one_identity() {
        // x / 1 → x  (with a non-const x)
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, one) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Div(x, one))
            .expect("add op should succeed");

        let mut pass = ConstantFoldingPass::new();
        pass.run(&mut graph);

        let has_div = graph.nodes.iter().any(|n| matches!(n.op, Op::Div(_, _)));
        assert!(!has_div, "x / 1 should eliminate the Div node");
    }
}
