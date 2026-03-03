use crate::ir::op::Op;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub usize);

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub output: ValueId,
}

impl Node {
    #[must_use]
    pub fn new(id: NodeId, op: Op, output: ValueId) -> Self {
        Self { id, op, output }
    }
}
