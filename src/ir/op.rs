use crate::ir::node::ValueId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ElementwiseUnaryOp {
    Neg,
    Relu,
}

#[derive(Debug, Clone)]
pub enum Op {
    ConstInt(i64),
    ConstFloat(f64),
    ConstTensor {
        shape: Vec<usize>,
        data: Vec<f32>,
    },
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    Neg(ValueId),
    ElementwiseChain {
        input: ValueId,
        ops: Vec<ElementwiseUnaryOp>,
    },
    Transpose(ValueId),
    MatMul(ValueId, ValueId),
    Relu(ValueId),
    ReluBackward(ValueId, ValueId),
    Softmax(ValueId),
    Conv2D(ValueId, ValueId),
    Parameter(String),
    Input(String),
    Output(ValueId),
    Phi(Vec<ValueId>),
    Removed,
}

impl Op {
    pub fn input_values(&self) -> Vec<ValueId> {
        match self {
            Op::Add(left, right)
            | Op::Sub(left, right)
            | Op::Mul(left, right)
            | Op::Div(left, right)
            | Op::MatMul(left, right)
            | Op::ReluBackward(left, right)
            | Op::Conv2D(left, right) => vec![*left, *right],
            Op::Relu(value) | Op::Softmax(value) | Op::Output(value) => vec![*value],
            Op::Neg(value) | Op::Transpose(value) => vec![*value],
            Op::ElementwiseChain { input, .. } => vec![*input],
            Op::Phi(values) => values.clone(),
            Op::ConstInt(_)
            | Op::ConstFloat(_)
            | Op::ConstTensor { .. }
            | Op::Parameter(_)
            | Op::Input(_)
            | Op::Removed => Vec::new(),
        }
    }

    pub fn remap_inputs(&mut self, mut remap: impl FnMut(ValueId) -> ValueId) {
        match self {
            Op::Add(left, right)
            | Op::Sub(left, right)
            | Op::Mul(left, right)
            | Op::Div(left, right)
            | Op::MatMul(left, right)
            | Op::ReluBackward(left, right)
            | Op::Conv2D(left, right) => {
                *left = remap(*left);
                *right = remap(*right);
            }
            Op::Relu(value)
            | Op::Softmax(value)
            | Op::Output(value)
            | Op::Neg(value)
            | Op::Transpose(value) => {
                *value = remap(*value);
            }
            Op::ElementwiseChain { input, .. } => {
                *input = remap(*input);
            }
            Op::Phi(values) => {
                for value in values {
                    *value = remap(*value);
                }
            }
            Op::ConstInt(_)
            | Op::ConstFloat(_)
            | Op::ConstTensor { .. }
            | Op::Parameter(_)
            | Op::Input(_)
            | Op::Removed => {}
        }
    }
}
