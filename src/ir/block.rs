use crate::ir::node::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BasicBlockId(pub usize);

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BasicBlockId,
    pub nodes: Vec<NodeId>,
}

impl BasicBlock {
    pub fn new(id: BasicBlockId) -> Self {
        Self {
            id,
            nodes: Vec::new(),
        }
    }
}
