use crate::ir::{Graph, Op, Pass, ValueId, run_with_verifier_guard};

#[derive(Debug, Clone, Copy)]
enum NumericConst {
    Int(i64),
    Float(f64),
}

#[derive(Default)]
pub struct AlgebraicSimplificationPass;

impl AlgebraicSimplificationPass {
    pub fn new() -> Self {
        Self
    }
}

impl Pass for AlgebraicSimplificationPass {
    fn run(&mut self, graph: &mut Graph) {
        run_with_verifier_guard(graph, |graph| {
            let mut known = vec![None; graph.value_count()];
            for node in &mut graph.nodes {
                let replacement = simplify_op(&node.op, &known);
                if let Some(new_op) = replacement {
                    node.op = new_op;
                }
                known[node.output.0] = constant_of(&node.op, &known);
            }
        });
    }
}

fn simplify_op(op: &Op, known: &[Option<NumericConst>]) -> Option<Op> {
    match op {
        Op::Add(left, right) => {
            if is_zero(*right, known) {
                Some(Op::Output(*left))
            } else if is_zero(*left, known) {
                Some(Op::Output(*right))
            } else {
                None
            }
        }
        Op::Mul(left, right) => {
            if is_zero(*left, known) {
                Some(Op::Output(*left))
            } else if is_zero(*right, known) {
                Some(Op::Output(*right))
            } else if is_one(*right, known) {
                Some(Op::Output(*left))
            } else if is_one(*left, known) {
                Some(Op::Output(*right))
            } else {
                None
            }
        }
        Op::Sub(left, right) => {
            if is_zero(*right, known) {
                Some(Op::Output(*left))
            } else {
                None
            }
        }
        Op::Div(left, right) => {
            if is_one(*right, known) {
                Some(Op::Output(*left))
            } else {
                None
            }
        }
        Op::ConstInt(_)
        | Op::ConstFloat(_)
        | Op::ConstTensor { .. }
        | Op::Neg(_)
        | Op::ElementwiseChain { .. }
        | Op::Transpose(_)
        | Op::MatMul(_, _)
        | Op::Relu(_)
        | Op::ReluBackward(_, _)
        | Op::Softmax(_)
        | Op::Conv2D(_, _)
        | Op::Parameter(_)
        | Op::Input(_)
        | Op::Output(_)
        | Op::Phi(_)
        | Op::Removed => None,
    }
}

fn constant_of(op: &Op, known: &[Option<NumericConst>]) -> Option<NumericConst> {
    match op {
        Op::ConstInt(value) => Some(NumericConst::Int(*value)),
        Op::ConstFloat(value) => Some(NumericConst::Float(*value)),
        Op::Output(value) => known.get(value.0).and_then(|v| *v),
        _ => None,
    }
}

fn is_zero(value: ValueId, known: &[Option<NumericConst>]) -> bool {
    match known.get(value.0).and_then(|v| *v) {
        Some(NumericConst::Int(v)) => v == 0,
        Some(NumericConst::Float(v)) => v == 0.0,
        None => false,
    }
}

fn is_one(value: ValueId, known: &[Option<NumericConst>]) -> bool {
    match known.get(value.0).and_then(|v| *v) {
        Some(NumericConst::Int(v)) => v == 1,
        Some(NumericConst::Float(v)) => v == 1.0,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{AlgebraicSimplificationPass, Graph, Op, Pass, execute};

    #[test]
    fn simplifies_local_identities_without_semantic_change() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::ConstInt(9))
            .expect("add op should succeed");
        let (_, zero) = graph
            .add_op(block, Op::ConstInt(0))
            .expect("add op should succeed");
        let (_, one) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        let (_, a) = graph
            .add_op(block, Op::Add(x, zero))
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(block, Op::Mul(a, one))
            .expect("add op should succeed");
        let (_, c) = graph
            .add_op(block, Op::Sub(b, zero))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Div(c, one))
            .expect("add op should succeed");

        let before = execute(&graph).expect("execute should pass");
        let mut pass = AlgebraicSimplificationPass::new();
        pass.run(&mut graph);
        let after = execute(&graph).expect("execute should pass");
        assert_eq!(before, after);
    }

    #[test]
    fn simplifies_multiply_by_zero() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::ConstInt(123))
            .expect("add op should succeed");
        let (_, zero) = graph
            .add_op(block, Op::ConstInt(0))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Mul(x, zero))
            .expect("add op should succeed");

        let before = execute(&graph).expect("execute should pass");
        let mut pass = AlgebraicSimplificationPass::new();
        pass.run(&mut graph);
        let after = execute(&graph).expect("execute should pass");
        assert_eq!(before, after);
    }
}
