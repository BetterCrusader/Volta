use crate::engine::ir::kernels::activations::relu_backward;
use crate::engine::ir::tensor::Tensor;
use crate::ir::{ElementwiseUnaryOp, Graph, Op, Pass, run_with_verifier_guard};

#[derive(Default)]
pub struct TensorConstantPropagationPass;

impl TensorConstantPropagationPass {
    pub fn new() -> Self {
        Self
    }
}

impl Pass for TensorConstantPropagationPass {
    fn run(&mut self, graph: &mut Graph) {
        run_with_verifier_guard(graph, |graph| {
            let mut known_tensors: Vec<Option<Tensor>> = vec![None; graph.value_count()];

            for node in &mut graph.nodes {
                if let Some(folded) = fold_tensor_op(&node.op, &known_tensors) {
                    node.op = folded;
                }

                if let Some(tensor) = tensor_from_const_op(&node.op) {
                    known_tensors[node.output.0] = Some(tensor);
                }
            }
        });
    }

    fn name(&self) -> &str {
        "TensorConstantPropagationPass"
    }
}

fn fold_tensor_op(op: &Op, known_tensors: &[Option<Tensor>]) -> Option<Op> {
    match op {
        Op::Add(a, b) => {
            let lhs = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            let rhs = known_tensors.get(b.0).and_then(|v| v.as_ref())?;
            const_tensor_op(lhs.add(rhs).ok()?)
        }
        Op::Sub(a, b) => {
            let lhs = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            let rhs = known_tensors.get(b.0).and_then(|v| v.as_ref())?;
            const_tensor_op(lhs.sub(rhs).ok()?)
        }
        Op::Mul(a, b) => {
            let lhs = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            let rhs = known_tensors.get(b.0).and_then(|v| v.as_ref())?;
            const_tensor_op(lhs.mul_elementwise(rhs).ok()?)
        }
        Op::MatMul(a, b) => {
            let lhs = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            let rhs = known_tensors.get(b.0).and_then(|v| v.as_ref())?;
            const_tensor_op(lhs.matmul(rhs).ok()?)
        }
        Op::Gemm {
            lhs,
            rhs,
            bias,
            alpha,
            beta,
        } => {
            let lhs_t = known_tensors.get(lhs.0).and_then(|v| v.as_ref())?;
            let rhs_t = known_tensors.get(rhs.0).and_then(|v| v.as_ref())?;
            let bias_t = if let Some(b) = bias {
                Some(known_tensors.get(b.0).and_then(|v| v.as_ref())?)
            } else {
                None
            };

            const_tensor_op(lhs_t.gemm(rhs_t, bias_t, *alpha, *beta).ok()?)
        }
        Op::Relu(a) => {
            let input = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            const_tensor_op(input.relu().ok()?)
        }
        Op::Sigmoid(a) => {
            let input = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            const_tensor_op(input.sigmoid().ok()?)
        }
        Op::ReluBackward(a, b) => {
            let input = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            let grad = known_tensors.get(b.0).and_then(|v| v.as_ref())?;
            const_tensor_op(relu_backward(input, grad).ok()?)
        }
        Op::ElementwiseChain { input, ops } => {
            let mut tensor = known_tensors.get(input.0).and_then(|v| v.clone())?;
            for unary in ops {
                tensor = match unary {
                    ElementwiseUnaryOp::Neg => tensor.scale(-1.0).ok()?,
                    ElementwiseUnaryOp::Relu => tensor.relu().ok()?,
                    ElementwiseUnaryOp::Sigmoid => tensor.sigmoid().ok()?,
                    ElementwiseUnaryOp::Gelu => tensor.gelu().ok()?,
                    ElementwiseUnaryOp::GeluExact => tensor.gelu_exact().ok()?,
                    ElementwiseUnaryOp::Exp => tensor.exp_elementwise().ok()?,
                    ElementwiseUnaryOp::Log => tensor.log_elementwise().ok()?,
                    ElementwiseUnaryOp::LeakyRelu(alpha) => {
                        // Toy implementation for constant folding
                        let mut data = (*tensor.data).clone();
                        for v in &mut data {
                            if *v < 0.0 {
                                *v *= alpha;
                            }
                        }
                        Tensor::new(tensor.shape.clone(), data).ok()?
                    }
                };
            }
            const_tensor_op(tensor)
        }
        Op::Reshape { input, shape } => {
            let input = known_tensors.get(input.0).and_then(|v| v.as_ref())?;
            const_tensor_op(input.reshape(shape.clone()).ok()?)
        }
        Op::Transpose(input) => {
            let input = known_tensors.get(input.0).and_then(|v| v.as_ref())?;
            const_tensor_op(input.transpose_2d().ok()?)
        }
        Op::Concat { inputs, axis } => {
            let mut tensors = Vec::new();
            for id in inputs {
                tensors.push(known_tensors.get(id.0).and_then(|v| v.clone())?);
            }
            const_tensor_op(Tensor::concat(&tensors, *axis).ok()?)
        }
        Op::Slice {
            input,
            starts,
            ends,
            axes,
        } => {
            let input = known_tensors.get(input.0).and_then(|v| v.as_ref())?;
            const_tensor_op(input.slice(starts, ends, axes).ok()?)
        }
        Op::Flatten { input, axis } => {
            let input = known_tensors.get(input.0).and_then(|v| v.as_ref())?;
            let left = input.shape[..*axis].iter().product::<usize>().max(1);
            let right = input.shape[*axis..].iter().product::<usize>().max(1);
            const_tensor_op(input.reshape(vec![left, right]).ok()?)
        }
        _ => None,
    }
}

fn tensor_from_const_op(op: &Op) -> Option<Tensor> {
    match op {
        Op::ConstTensor { shape, data } => Tensor::new(shape.clone(), data.clone()).ok(),
        _ => None,
    }
}

fn const_tensor_op(tensor: Tensor) -> Option<Op> {
    Some(Op::ConstTensor {
        shape: tensor.shape,
        data: tensor.data.to_vec(),
    })
}
