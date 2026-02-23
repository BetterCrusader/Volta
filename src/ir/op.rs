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
    Reshape {
        input: ValueId,
        shape: Vec<usize>,
    },
    Concat {
        inputs: Vec<ValueId>,
        axis: usize,
    },
    Gather {
        input: ValueId,
        indices: Vec<usize>,
        axis: usize,
    },
    Slice {
        input: ValueId,
        starts: Vec<usize>,
        ends: Vec<usize>,
        axes: Vec<usize>,
    },
    Transpose(ValueId),
    MatMul(ValueId, ValueId),
    Relu(ValueId),
    ReluBackward(ValueId, ValueId),
    Softmax(ValueId),
    Log(ValueId),
    Exp(ValueId),
    Sigmoid(ValueId),
    SigmoidBackward(ValueId, ValueId),
    Gelu(ValueId),
    GeluExact(ValueId),
    GeluBackward(ValueId, ValueId),
    ReduceSum {
        input: ValueId,
        axis: Option<usize>,
        keepdims: bool,
    },
    ReduceMax {
        input: ValueId,
        axis: Option<usize>,
        keepdims: bool,
    },
    ReduceMean {
        input: ValueId,
        axis: Option<usize>,
        keepdims: bool,
    },
    /// Generalised Matrix Multiply: `alpha * (lhs × rhs) + beta * bias`.
    ///
    /// Matches the ONNX `Gemm` operator and the standard BLAS SGEMM interface.
    /// When `bias` is `None` the operation degenerates to a scaled MatMul.
    Gemm {
        lhs: ValueId,
        rhs: ValueId,
        bias: Option<ValueId>,
        alpha: f32,
        beta: f32,
    },
    GemmBackward {
        lhs: ValueId,
        rhs: ValueId,
        bias: Option<ValueId>,
        alpha: f32,
        beta: f32,
    },
    Conv2D(ValueId, ValueId),
    Parameter(String),
    Input(String),
    Output(ValueId),
    Phi(Vec<ValueId>),
    Removed,
}

impl Op {
    #[must_use]
    pub fn input_values(&self) -> Vec<ValueId> {
        match self {
            Op::Add(left, right)
            | Op::Sub(left, right)
            | Op::Mul(left, right)
            | Op::Div(left, right)
            | Op::MatMul(left, right)
            | Op::SigmoidBackward(left, right)
            | Op::GeluBackward(left, right)
            | Op::ReluBackward(left, right)
            | Op::Conv2D(left, right) => vec![*left, *right],
            Op::Relu(value) | Op::Softmax(value) | Op::Output(value) => vec![*value],
            Op::Neg(value)
            | Op::Transpose(value)
            | Op::Log(value)
            | Op::Exp(value)
            | Op::Sigmoid(value)
            | Op::Gelu(value)
            | Op::GeluExact(value)
            | Op::GemmBackward { lhs: value, .. } => vec![*value],
            Op::ElementwiseChain { input, .. } => vec![*input],
            Op::Reshape { input, .. }
            | Op::Gather { input, .. }
            | Op::Slice { input, .. }
            | Op::ReduceSum { input, .. }
            | Op::ReduceMax { input, .. }
            | Op::ReduceMean { input, .. } => {
                vec![*input]
            }
            Op::Gemm { lhs, rhs, bias, .. } => {
                let mut inputs = vec![*lhs, *rhs];
                if let Some(b) = bias {
                    inputs.push(*b);
                }
                inputs
            }
            Op::Concat { inputs, .. } => inputs.clone(),
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
            | Op::SigmoidBackward(left, right)
            | Op::GeluBackward(left, right)
            | Op::Conv2D(left, right) => {
                *left = remap(*left);
                *right = remap(*right);
            }
            Op::Relu(value)
            | Op::Softmax(value)
            | Op::Output(value)
            | Op::Neg(value)
            | Op::Transpose(value)
            | Op::Log(value)
            | Op::Exp(value)
            | Op::Sigmoid(value)
            | Op::Gelu(value)
            | Op::GeluExact(value) => {
                *value = remap(*value);
            }
            Op::ElementwiseChain { input, .. } => {
                *input = remap(*input);
            }
            Op::Reshape { input, .. }
            | Op::Gather { input, .. }
            | Op::Slice { input, .. }
            | Op::ReduceSum { input, .. }
            | Op::ReduceMax { input, .. }
            | Op::ReduceMean { input, .. } => {
                *input = remap(*input);
            }
            Op::Gemm { lhs, rhs, bias, .. } => {
                *lhs = remap(*lhs);
                *rhs = remap(*rhs);
                if let Some(b) = bias {
                    *b = remap(*b);
                }
            }
            Op::GemmBackward { lhs, rhs, bias, .. } => {
                *lhs = remap(*lhs);
                *rhs = remap(*rhs);
                if let Some(b) = bias {
                    *b = remap(*b);
                }
            }
            Op::Concat { inputs, .. } => {
                for input in inputs {
                    *input = remap(*input);
                }
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
