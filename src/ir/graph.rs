use std::collections::BTreeMap;

use crate::ir::block::{BasicBlock, BasicBlockId};
use crate::ir::node::{Node, NodeId, ValueId};
use crate::ir::op::Op;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct ShapeSignature {
    pub inputs: BTreeMap<String, Vec<usize>>,
    pub parameters: BTreeMap<String, Vec<usize>>,
}

#[derive(Debug, Default, Clone)]
pub struct Graph {
    pub blocks: Vec<BasicBlock>,
    pub nodes: Vec<Node>,
    pub shape_signature: ShapeSignature,
}

#[derive(Debug, Clone)]
pub struct GraphError {
    pub message: String,
}

impl Graph {
    #[must_use]
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            nodes: Vec::new(),
            shape_signature: ShapeSignature::default(),
        }
    }

    pub fn bind_input_shape(&mut self, name: &str, shape: Vec<usize>) {
        self.shape_signature.inputs.insert(name.to_string(), shape);
    }

    pub fn bind_parameter_shape(&mut self, name: &str, shape: Vec<usize>) {
        self.shape_signature
            .parameters
            .insert(name.to_string(), shape);
    }

    #[must_use]
    pub fn input_shape(&self, name: &str) -> Option<&[usize]> {
        self.shape_signature
            .inputs
            .get(name)
            .map(std::vec::Vec::as_slice)
    }

    #[must_use]
    pub fn parameter_shape(&self, name: &str) -> Option<&[usize]> {
        self.shape_signature
            .parameters
            .get(name)
            .map(std::vec::Vec::as_slice)
    }

    pub fn create_block(&mut self) -> BasicBlockId {
        let id = BasicBlockId(self.blocks.len());
        self.blocks.push(BasicBlock::new(id));
        id
    }

    pub fn add_op(
        &mut self,
        block_id: BasicBlockId,
        op: Op,
    ) -> Result<(NodeId, ValueId), GraphError> {
        let node_id = NodeId(self.nodes.len());
        let value_id = ValueId(self.nodes.len());
        let node = Node::new(node_id, op, value_id);
        self.nodes.push(node);

        let block = self.blocks.get_mut(block_id.0).ok_or_else(|| GraphError {
            message: format!("Invalid BasicBlockId: {}", block_id.0),
        })?;
        block.nodes.push(node_id);

        Ok((node_id, value_id))
    }

    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0)
    }

    #[must_use]
    pub fn value_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| node.output.0)
            .max()
            .map_or(0, |max_id| max_id + 1)
    }

    #[must_use]
    pub fn last_value_id(&self) -> Option<ValueId> {
        self.nodes.last().map(|node| node.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_op_has_single_output_value() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (first_node, first_value) = graph
            .add_op(block, Op::ConstInt(7))
            .expect("op add must succeed");
        let (second_node, second_value) = graph
            .add_op(block, Op::Add(first_value, first_value))
            .expect("op add must succeed");

        let first = graph.node(first_node).expect("node must exist");
        let second = graph.node(second_node).expect("node must exist");

        assert_eq!(first.output, first_value);
        assert_eq!(second.output, second_value);
        assert_ne!(first.output, second.output);
    }

    #[test]
    fn shape_signature_bindings_are_deterministic() {
        let mut graph = Graph::new();
        graph.bind_parameter_shape("w", vec![4, 4]);
        graph.bind_input_shape("x", vec![1, 4]);

        assert_eq!(graph.input_shape("x"), Some(&[1, 4][..]));
        assert_eq!(graph.parameter_shape("w"), Some(&[4, 4][..]));
    }
}
