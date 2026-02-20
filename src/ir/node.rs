use crate::ir::op::Op;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub usize);

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub output: ValueId,
}

impl Node {
    pub fn new(id: NodeId, op: Op, output: ValueId) -> Self {
        Self { id, op, output }
    }
}
