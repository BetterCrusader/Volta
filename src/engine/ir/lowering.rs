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
    #[must_use]
    pub fn new() -> Self {
        let mut graph = Graph::new();
        let entry_block = graph.create_block();
        Self {
            graph,
            entry_block,
            variables: HashMap::new(),
        }
    }

    #[must_use]
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

    pub fn push_op(&mut self, op: Op) -> Option<ValueId> {
        self.graph
            .add_op(self.entry_block, op)
            .ok()
            .map(|(_, value)| value)
    }

    /// Extends the graph with neural network layers based on the provided configuration.
    ///
    /// The input must be a tensor of shape `[batch, in_features]`.
    /// Returns the `ValueId` representing the final output logits of the model.
    pub fn lower_model_to_graph(
        &mut self,
        layers: &[i64],
        activation: &str,
        input_val: ValueId,
        apply_final_activation: bool,
    ) -> ValueId {
        let mut current_input = input_val;

        for (i, w) in layers.windows(2).enumerate() {
            let in_features = w[0] as usize;
            let out_features = w[1] as usize;

            // 1. Generate Weights: [in_features, out_features]
            // In the real system, these would be matched with random tensors in the executor
            let weight_name = format!("weight_{in_features}_{out_features}");
            let weight_op = Op::Parameter(weight_name.clone());
            let weight = self.graph.add_op(self.entry_block, weight_op).unwrap().1;

            // 2. Generate Bias: [1, out_features]
            let bias_name = format!("bias_{out_features}");
            let bias_op = Op::Parameter(bias_name.clone());
            let bias = self.graph.add_op(self.entry_block, bias_op).unwrap().1;

            // 3. Matrix Multiplication
            let matmul_op = Op::MatMul(current_input, weight);
            let matmul = self.graph.add_op(self.entry_block, matmul_op).unwrap().1;

            // 4. Add Bias (simulated with elementwise add for brevity, assuming broadcast in backend eventually)
            let add_op = Op::Add(matmul, bias);
            let add = self.graph.add_op(self.entry_block, add_op).unwrap().1;

            // 5. Activation
            current_input = add;

            let is_last_layer = i == layers.len() - 2;
            if !is_last_layer {
                if activation == "relu" {
                    let relu_op = Op::Relu(current_input);
                    current_input = self.graph.add_op(self.entry_block, relu_op).unwrap().1;
                } else if activation == "sigmoid" {
                    let sig_op = Op::Sigmoid(current_input);
                    current_input = self.graph.add_op(self.entry_block, sig_op).unwrap().1;
                }
            } else if apply_final_activation {
                // For the last layer, if the activation is softmax (e.g. inference), apply it.
                if activation == "softmax" {
                    let soft_op = Op::Softmax(current_input);
                    current_input = self.graph.add_op(self.entry_block, soft_op).unwrap().1;
                }
            }
        }

        current_input
    }

    /// Constructs a Mean Squared Error (MSE) loss metric.
    pub fn build_mse_loss(&mut self, predictions: ValueId, target_labels: ValueId) -> ValueId {
        let error = self.push_op(Op::Sub(predictions, target_labels)).unwrap();
        let squared_error = self.push_op(Op::Mul(error, error)).unwrap();
        self.push_op(Op::ReduceMean {
            input: squared_error,
            axis: None,
            keepdims: false,
        })
        .unwrap()
    }

    /// Constructs a Softmax Cross Entropy loss metric.
    /// Expects raw logits and one-hot target labels.
    pub fn build_softmax_cross_entropy_loss(
        &mut self,
        logits: ValueId,
        target_labels: ValueId,
    ) -> ValueId {
        self.push_op(Op::SoftmaxCrossEntropyLossFromLogits {
            logits,
            targets: target_labels,
        })
        .unwrap()
    }

    /// Consumes the context and returns the constructed graph.
    #[must_use]
    pub fn into_graph(self) -> Graph {
        self.graph
    }
}

#[must_use]
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
                | Op::Log(_)
                | Op::Exp(_)
                | Op::Sigmoid(_)
                | Op::SigmoidBackward(_, _)
                | Op::GeluExact(_)
                | Op::Gelu(_)
                | Op::GeluBackward(_, _)
                | Op::GeluExactBackward(_, _)
                | Op::ReduceMaxBackward { .. }
                | Op::Gemm { .. }
                | Op::GemmBackward { .. }
                | Op::ReduceSum { .. }
                | Op::ReduceMax { .. }
                | Op::ReduceMean { .. }
                | Op::Conv2D(_, _)
                | Op::Conv2DBackwardInput(_, _, _)
                | Op::Conv2DBackwardWeight(_, _, _)
                | Op::MaxPool { .. }
                | Op::AvgPool { .. }
                | Op::BatchNorm { .. }
                | Op::Flatten { .. }
                | Op::GlobalAveragePool { .. }
                | Op::GlobalAveragePoolBackward { .. }
                | Op::GroupNorm { .. }
                | Op::GroupNormBackwardInput { .. }
                | Op::GroupNormBackwardWeight { .. }
                | Op::GroupNormBackwardBias { .. }
                | Op::InstanceNorm { .. }
                | Op::InstanceNormBackwardInput { .. }
                | Op::InstanceNormBackwardWeight { .. }
                | Op::InstanceNormBackwardBias { .. }
                | Op::Embedding { .. }
                | Op::EmbeddingBackward { .. }
                | Op::LstmCell { .. }
                | Op::LstmCellBackward { .. }
                | Op::GruCell { .. }
                | Op::GruCellBackward { .. }
                | Op::ConvTranspose2D { .. }
                | Op::Upsample2D { .. }
                | Op::Upsample2DBackward { .. }
                | Op::MultiHeadAttention { .. }
                | Op::SinusoidalPE { .. }
                | Op::RoPE { .. }
                | Op::RoPEBackward { .. }
                | Op::Dropout { .. }
                | Op::Identity(_)
                | Op::MaxPoolBackward { .. }
                | Op::AvgPoolBackward { .. }
                | Op::BatchNormBackwardInput { .. }
                | Op::BatchNormBackwardWeight { .. }
                | Op::BatchNormBackwardBias { .. }
                | Op::LayerNorm { .. }
                | Op::LayerNormBackwardInput { .. }
                | Op::LayerNormBackwardWeight { .. }
                | Op::LayerNormBackwardBias { .. }
                | Op::Parameter(_)
                | Op::Input(_)
                | Op::Output(_)
                | Op::Phi(_)
                | Op::SoftmaxCrossEntropyLossFromLogits { .. }
                | Op::Plugin { .. }
                | Op::Removed
                | Op::QuantizeLinear { .. }
                | Op::DequantizeLinear { .. }
                | Op::DepthwiseSeparableConv { .. }
                | Op::CustomCall { .. } => {}
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
