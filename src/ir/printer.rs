use crate::ir::{Graph, Op, ValueId};

pub fn print_graph(graph: &Graph) -> String {
    let mut lines = Vec::new();
    for node in &graph.nodes {
        lines.push(format!("%{} = {}", node.output.0, format_op(&node.op)));
    }
    lines.join("\n")
}

fn format_op(op: &Op) -> String {
    match op {
        Op::ConstInt(value) => format!("const {value}"),
        Op::ConstFloat(value) => format!("const {value}"),
        Op::ConstTensor { shape, .. } => format!("const_tensor {:?}", shape),
        Op::Add(left, right) => format!("add {} {}", fmt_value(*left), fmt_value(*right)),
        Op::Sub(left, right) => format!("sub {} {}", fmt_value(*left), fmt_value(*right)),
        Op::Mul(left, right) => format!("mul {} {}", fmt_value(*left), fmt_value(*right)),
        Op::Div(left, right) => format!("div {} {}", fmt_value(*left), fmt_value(*right)),
        Op::Neg(value) => format!("neg {}", fmt_value(*value)),
        Op::ElementwiseChain { input, ops } => {
            let chain = ops
                .iter()
                .map(|op| match op {
                    crate::ir::ElementwiseUnaryOp::Neg => "neg",
                    crate::ir::ElementwiseUnaryOp::Relu => "relu",
                })
                .collect::<Vec<_>>()
                .join("->");
            format!("elementwise_chain {} [{}]", fmt_value(*input), chain)
        }
        Op::Transpose(value) => format!("transpose {}", fmt_value(*value)),
        Op::MatMul(left, right) => format!("matmul {} {}", fmt_value(*left), fmt_value(*right)),
        Op::Relu(value) => format!("relu {}", fmt_value(*value)),
        Op::ReluBackward(input, grad) => {
            format!("relu_backward {} {}", fmt_value(*input), fmt_value(*grad))
        }
        Op::Softmax(value) => format!("softmax {}", fmt_value(*value)),
        Op::Conv2D(input, weight) => {
            format!("conv2d {} {}", fmt_value(*input), fmt_value(*weight))
        }
        Op::Parameter(name) => format!("parameter {name}"),
        Op::Input(name) => format!("input {name}"),
        Op::Output(value) => format!("output {}", fmt_value(*value)),
        Op::Phi(values) => {
            let formatted = values
                .iter()
                .map(|id| fmt_value(*id))
                .collect::<Vec<_>>()
                .join(" ");
            format!("phi {formatted}")
        }
        Op::Removed => "removed".to_string(),
    }
}

fn fmt_value(value: ValueId) -> String {
    format!("%{}", value.0)
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op};

    use super::print_graph;

    #[test]
    fn prints_ssa_style_lines() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(5))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstInt(3))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(v0, v1))
            .expect("add op should succeed");

        let actual = print_graph(&graph);
        let expected = "%0 = const 5\n%1 = const 3\n%2 = add %0 %1";
        assert_eq!(actual, expected);
    }

    #[test]
    fn snapshot_print_for_fused_tensor_chain() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2],
                    data: vec![1.0, -2.0],
                },
            )
            .expect("add op should succeed");
        let (_, relu) = graph
            .add_op(block, Op::Relu(input))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Neg(relu))
            .expect("add op should succeed");

        let actual = print_graph(&graph);
        let expected = "%0 = const_tensor [2]\n%1 = relu %0\n%2 = neg %1";
        assert_eq!(actual, expected);
    }
}
