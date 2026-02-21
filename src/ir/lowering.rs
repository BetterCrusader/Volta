use std::collections::HashMap;

use crate::ast::{BinaryOp, Expr, Program, Stmt};
use crate::ir::verify_graph;
use crate::ir::{BasicBlockId, Graph, Op, ValueId};

pub struct LoweringContext {
    graph: Graph,
    entry_block: BasicBlockId,
    variables: HashMap<String, ValueId>,
}

impl Default for LoweringContext {
    fn default() -> Self {
        Self::new()
    }
}

impl LoweringContext {
    pub fn new() -> Self {
        let mut graph = Graph::new();
        let entry_block = graph.create_block();
        Self {
            graph,
            entry_block,
            variables: HashMap::new(),
        }
    }

    pub fn lower_program(mut self, program: &Program) -> Graph {
        for stmt in &program.statements {
            self.lower_stmt(stmt);
        }
        self.graph
    }

    fn lower_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::VarDecl { name, value, .. } | Stmt::Assign { name, value, .. } => {
                if let Some(value_id) = self.lower_expr(value) {
                    self.variables.insert(name.clone(), value_id);
                }
            }
            Stmt::Print { expr, .. } => {
                let _ = self.lower_expr(expr);
            }
            _ => {}
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> Option<ValueId> {
        match expr {
            Expr::Int { value, .. } => self.push_op(Op::ConstInt(*value)),
            Expr::Float { value, .. } => self.push_op(Op::ConstFloat(*value)),
            Expr::Ident { name, .. } => self.variables.get(name).copied(),
            Expr::Binary {
                left, op, right, ..
            } => {
                let left_value = self.lower_expr(left)?;
                let right_value = self.lower_expr(right)?;
                let lowered = match op {
                    BinaryOp::Add => Op::Add(left_value, right_value),
                    BinaryOp::Sub => Op::Sub(left_value, right_value),
                    BinaryOp::Mul => Op::Mul(left_value, right_value),
                    BinaryOp::Div => Op::Div(left_value, right_value),
                    _ => return None,
                };
                self.push_op(lowered)
            }
            _ => None,
        }
    }

    fn push_op(&mut self, op: Op) -> Option<ValueId> {
        self.graph
            .add_op(self.entry_block, op)
            .ok()
            .map(|(_, value)| value)
    }
}

pub fn lower_program(program: &Program) -> Graph {
    let graph = LoweringContext::new().lower_program(program);
    debug_assert!(verify_graph(&graph).is_ok());
    graph
}

#[cfg(test)]
mod tests {
    use crate::ast::Program;
    use crate::ir::{Op, lower_program};
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn parse(source: &str) -> Program {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        parser.parse_program().expect("parse failed")
    }

    #[test]
    fn lowers_arithmetic_into_linear_ssa_nodes() {
        let program = parse("x 1 + 2\ny x * 3\nprint y / 2\n");
        let graph = lower_program(&program);

        assert_eq!(graph.blocks.len(), 1);
        assert_eq!(graph.nodes.len(), 7);

        assert!(matches!(graph.nodes[0].op, Op::ConstInt(1)));
        assert!(matches!(graph.nodes[1].op, Op::ConstInt(2)));

        match graph.nodes[2].op {
            Op::Add(left, right) => {
                assert_eq!(left.0, 0);
                assert_eq!(right.0, 1);
            }
            _ => panic!("expected Add op"),
        }

        assert!(matches!(graph.nodes[3].op, Op::ConstInt(3)));

        match graph.nodes[4].op {
            Op::Mul(left, right) => {
                assert_eq!(left.0, 2);
                assert_eq!(right.0, 3);
            }
            _ => panic!("expected Mul op"),
        }

        assert!(matches!(graph.nodes[5].op, Op::ConstInt(2)));

        match graph.nodes[6].op {
            Op::Div(left, right) => {
                assert_eq!(left.0, 4);
                assert_eq!(right.0, 5);
            }
            _ => panic!("expected Div op"),
        }

        for (index, node) in graph.nodes.iter().enumerate() {
            assert_eq!(node.id.0, index);
            assert_eq!(node.output.0, index);
            match &node.op {
                Op::Add(left, right)
                | Op::Sub(left, right)
                | Op::Mul(left, right)
                | Op::Div(left, right) => {
                    assert!(left.0 < node.output.0);
                    assert!(right.0 < node.output.0);
                }
                Op::ElementwiseChain { input, .. } => {
                    assert!(input.0 < node.output.0);
                }
                Op::Reshape { input, .. } | Op::Gather { input, .. } | Op::Slice { input, .. } => {
                    assert!(input.0 < node.output.0);
                }
                Op::Concat { inputs, .. } => {
                    for input in inputs {
                        assert!(input.0 < node.output.0);
                    }
                }
                Op::ConstInt(_)
                | Op::ConstFloat(_)
                | Op::ConstTensor { .. }
                | Op::Neg(_)
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
                | Op::Removed => {}
            }
        }
    }

    #[test]
    fn skips_unsupported_expressions_without_breaking_graph() {
        let program = parse("x true\ny x == x\n");
        let graph = lower_program(&program);
        assert!(graph.nodes.is_empty());
    }
}
