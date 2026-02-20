use crate::ir::tensor::Tensor;
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
                if let Some(replacement) = fold_tensor_op(&node.op, &known_tensors) {
                    node.op = replacement;
                }

                known_tensors[node.output.0] = tensor_from_const_op(&node.op);
            }
        });
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
        Op::Div(a, b) => {
            let lhs = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            let rhs = known_tensors.get(b.0).and_then(|v| v.as_ref())?;
            let mut out = Vec::with_capacity(lhs.data.len());
            if lhs.shape != rhs.shape {
                return None;
            }
            for (x, y) in lhs.data.iter().zip(rhs.data.iter()) {
                if *y == 0.0 {
                    return None;
                }
                out.push(*x / *y);
            }
            const_tensor_op(Tensor::new(lhs.shape.clone(), out).ok()?)
        }
        Op::Neg(a) => {
            let input = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            const_tensor_op(input.scale(-1.0).ok()?)
        }
        Op::Transpose(a) => {
            let input = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            const_tensor_op(input.transpose_2d().ok()?)
        }
        Op::MatMul(a, b) => {
            let lhs = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            let rhs = known_tensors.get(b.0).and_then(|v| v.as_ref())?;
            const_tensor_op(lhs.matmul(rhs).ok()?)
        }
        Op::Relu(a) => {
            let input = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            const_tensor_op(input.relu().ok()?)
        }
        Op::ReluBackward(a, b) => {
            let input = known_tensors.get(a.0).and_then(|v| v.as_ref())?;
            let grad = known_tensors.get(b.0).and_then(|v| v.as_ref())?;
            const_tensor_op(input.relu_backward(grad).ok()?)
        }
        Op::ElementwiseChain { input, ops } => {
            let mut tensor = known_tensors.get(input.0).and_then(|v| v.clone())?;
            for op in ops {
                tensor = match op {
                    ElementwiseUnaryOp::Neg => tensor.scale(-1.0).ok()?,
                    ElementwiseUnaryOp::Relu => tensor.relu().ok()?,
                };
            }
            const_tensor_op(tensor)
        }
        Op::ConstInt(_)
        | Op::ConstFloat(_)
        | Op::ConstTensor { .. }
        | Op::Softmax(_)
        | Op::Conv2D(_, _)
        | Op::Parameter(_)
        | Op::Input(_)
        | Op::Output(_)
        | Op::Phi(_)
        | Op::Removed => None,
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
        data: tensor.data,
    })
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, Pass, RuntimeValue, TensorConstantPropagationPass, execute};

    #[test]
    fn folds_tensor_elementwise_and_matmul_constants() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![1, 2],
                    data: vec![1.0, 2.0],
                },
            )
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2, 1],
                    data: vec![3.0, 4.0],
                },
            )
            .expect("add op should succeed");
        let (_, mm) = graph
            .add_op(block, Op::MatMul(a, b))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Relu(mm))
            .expect("add op should succeed");

        let mut pass = TensorConstantPropagationPass::new();
        pass.run(&mut graph);

        assert!(matches!(graph.nodes[2].op, Op::ConstTensor { .. }));
        assert!(matches!(graph.nodes[3].op, Op::ConstTensor { .. }));

        let result = execute(&graph).expect("execute should pass");
        assert_eq!(
            result,
            Some(RuntimeValue::Tensor {
                shape: vec![1, 1],
                data: vec![11.0]
            })
        );
    }
}
