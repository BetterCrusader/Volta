use crate::ir::NodeId;

pub fn run(nodes: &[NodeId]) -> Result<(), String> {
    if nodes.is_empty() {
        return Err("backward kernel dispatch received no nodes".to_string());
    }
    Ok(())
}
