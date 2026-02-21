use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::ir::{Graph, Op};

pub fn graph_fingerprint(graph: &Graph) -> u64 {
    let mut hasher = DefaultHasher::new();
    graph.nodes.len().hash(&mut hasher);
    graph.blocks.len().hash(&mut hasher);
    graph.shape_signature.inputs.hash(&mut hasher);
    graph.shape_signature.parameters.hash(&mut hasher);

    for node in &graph.nodes {
        node.id.0.hash(&mut hasher);
        node.output.0.hash(&mut hasher);
        hash_op(&node.op, &mut hasher);
    }

    hasher.finish()
}

fn hash_op(op: &Op, hasher: &mut DefaultHasher) {
    std::mem::discriminant(op).hash(hasher);
    match op {
        Op::ConstInt(value) => value.hash(hasher),
        Op::ConstFloat(value) => value.to_bits().hash(hasher),
        Op::ConstTensor { shape, data } => {
            shape.hash(hasher);
            for value in data {
                value.to_bits().hash(hasher);
            }
        }
        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::Div(a, b)
        | Op::ReluBackward(a, b)
        | Op::MatMul(a, b)
        | Op::Conv2D(a, b) => {
            a.0.hash(hasher);
            b.0.hash(hasher);
        }
        Op::Neg(v) | Op::Transpose(v) | Op::Relu(v) | Op::Softmax(v) | Op::Output(v) => {
            v.0.hash(hasher)
        }
        Op::ElementwiseChain { input, ops } => {
            input.0.hash(hasher);
            ops.hash(hasher);
        }
        Op::Reshape { input, shape } => {
            input.0.hash(hasher);
            shape.hash(hasher);
        }
        Op::Concat { inputs, axis } => {
            axis.hash(hasher);
            for input in inputs {
                input.0.hash(hasher);
            }
        }
        Op::Gather {
            input,
            indices,
            axis,
        } => {
            input.0.hash(hasher);
            indices.hash(hasher);
            axis.hash(hasher);
        }
        Op::Slice {
            input,
            starts,
            ends,
            axes,
        } => {
            input.0.hash(hasher);
            starts.hash(hasher);
            ends.hash(hasher);
            axes.hash(hasher);
        }
        Op::Parameter(name) | Op::Input(name) => name.hash(hasher),
        Op::Phi(values) => values.iter().for_each(|v| v.0.hash(hasher)),
        Op::Removed => {}
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, graph_fingerprint};

    #[test]
    fn fingerprint_is_deterministic() {
        let mut g1 = Graph::new();
        let b1 = g1.create_block();
        let (_, a1) = g1
            .add_op(b1, Op::ConstInt(1))
            .expect("add op should succeed");
        g1.add_op(b1, Op::Output(a1))
            .expect("add op should succeed");

        let mut g2 = Graph::new();
        let b2 = g2.create_block();
        let (_, a2) = g2
            .add_op(b2, Op::ConstInt(1))
            .expect("add op should succeed");
        g2.add_op(b2, Op::Output(a2))
            .expect("add op should succeed");

        assert_eq!(graph_fingerprint(&g1), graph_fingerprint(&g2));
    }
}
