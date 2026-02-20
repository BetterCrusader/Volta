use crate::ir::{Graph, Pass, run_with_verifier_guard};

#[derive(Default)]
pub struct LayoutAwareOptimizationPass;

impl LayoutAwareOptimizationPass {
    pub fn new() -> Self {
        Self
    }
}

impl Pass for LayoutAwareOptimizationPass {
    fn run(&mut self, graph: &mut Graph) {
        run_with_verifier_guard(graph, |_graph| {
            // Current IR stores tensors in a canonical row-major layout.
            // This pass is a hook for future layout transforms and kernel-specific rewrites.
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, LayoutAwareOptimizationPass, Op, Pass, RuntimeValue, execute};

    #[test]
    fn preserves_semantics_for_current_layout_model() {
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
        graph
            .add_op(block, Op::Relu(input))
            .expect("add op should succeed");

        let before = execute(&graph).expect("execute should pass");
        let mut pass = LayoutAwareOptimizationPass::new();
        pass.run(&mut graph);
        let after = execute(&graph).expect("execute should pass");

        assert_eq!(before, after);
        assert_eq!(
            after,
            Some(RuntimeValue::Tensor {
                shape: vec![2],
                data: vec![1.0, 0.0]
            })
        );
    }
}
