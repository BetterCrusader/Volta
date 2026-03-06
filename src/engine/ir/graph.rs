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

    /// Замінює ноду іншим графом (підграфом).
    /// `mapping` вказує, які `ValueId` підграфа відповідають входам оригінальної ноди.
    /// Повертає `ValueId` останньої ноди підграфа, яка замінює вихід оригінальної ноди.
    pub fn inline_subgraph(
        &mut self,
        node_id: NodeId,
        subgraph: Graph,
        input_mapping: BTreeMap<usize, ValueId>,
    ) -> Result<ValueId, GraphError> {
        let block_id = self.find_block_for_node(node_id).ok_or_else(|| GraphError {
            message: format!("Node {} not found in any block", node_id.0),
        })?;

        let mut value_remap = BTreeMap::new();
        for (sub_input_idx, outer_value) in input_mapping {
            value_remap.insert(ValueId(sub_input_idx), outer_value);
        }

        let mut last_value = None;

        // Додаємо ноди підграфа до поточного графа
        for sub_node in subgraph.nodes {
            let mut op = sub_node.op;
            op.remap_inputs(|id| *value_remap.get(&id).unwrap_or(&id));

            if let Op::Output(v) = op {
                last_value = Some(*value_remap.get(&v).unwrap_or(&v));
                continue;
            }

            let (_, new_value) = self.add_op(block_id, op)?;
            value_remap.insert(sub_node.output, new_value);
            last_value = Some(new_value);
        }

        // Позначаємо оригінальну ноду як видалену (або замінюємо її на Identity, якщо потрібно зберегти посилання)
        // Для простоти поки що просто замінимо її на Removed, але це може зламати інші ноди, що на неї посилаються.
        // Краще всього - замінити Op на Identity від останнього значення підграфа.
        if let Some(final_val) = last_value {
            if let Some(node) = self.nodes.get_mut(node_id.0) {
                node.op = Op::Identity(final_val);
            }
            Ok(final_val)
        } else {
            Err(GraphError {
                message: "Subgraph has no output".to_string(),
            })
        }
    }

    fn find_block_for_node(&self, node_id: NodeId) -> Option<BasicBlockId> {
        for block in &self.blocks {
            if block.nodes.contains(&node_id) {
                return Some(block.id);
            }
        }
        None
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
    fn inlines_subgraph_correctly() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v_in) = graph.add_op(block, Op::ConstInt(10)).unwrap();
        let (n_to_inline, _) = graph.add_op(block, Op::Identity(v_in)).unwrap();

        let mut subgraph = Graph::new();
        let sub_block = subgraph.create_block();
        // Вхід підграфа буде ValueId(0)
        let (_, v_sub_add) = subgraph.add_op(sub_block, Op::Add(ValueId(0), ValueId(0))).unwrap();
        subgraph.add_op(sub_block, Op::Output(v_sub_add)).unwrap();

        let mut mapping = BTreeMap::new();
        mapping.insert(0, v_in);

        let final_v = graph.inline_subgraph(n_to_inline, subgraph, mapping).unwrap();

        // Перевіряємо результат
        assert_eq!(graph.nodes.len(), 3); // ConstInt, Add, Identity(замість оригінальної ноди)
        if let Op::Add(l, r) = graph.nodes[2].op {
            assert_eq!(l, v_in);
            assert_eq!(r, v_in);
        } else {
            panic!("Expected Add node in inlined graph");
        }
        assert_eq!(final_v, graph.nodes[2].output);
    }
}
