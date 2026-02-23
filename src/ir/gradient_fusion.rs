use crate::ir::{ElementwiseFusionPass, Graph, Pass, run_with_verifier_guard};

#[derive(Default)]
pub struct GradientFusionPass;

impl GradientFusionPass {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Pass for GradientFusionPass {
    fn run(&mut self, graph: &mut Graph) {
        run_with_verifier_guard(graph, |graph| {
            // Reuse deterministic elementwise fusion for gradient subgraphs.
            let mut elementwise = ElementwiseFusionPass::new();
            elementwise.run(graph);
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{GradientFusionPass, Graph, Op, Pass};

    #[test]
    fn fuses_elementwise_gradient_chain() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2],
                    data: vec![1.0, -3.0],
                },
            )
            .expect("add op should succeed");
        let (_, relu) = graph
            .add_op(block, Op::Relu(input))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Neg(relu))
            .expect("add op should succeed");

        let mut pass = GradientFusionPass::new();
        pass.run(&mut graph);

        assert!(matches!(graph.nodes[1].op, Op::Removed));
        assert!(matches!(graph.nodes[2].op, Op::ElementwiseChain { .. }));
    }
}
