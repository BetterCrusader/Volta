use crate::ir::NodeId;

pub fn run(nodes: &[NodeId]) -> Result<(), String> {
    if nodes.is_empty() {
        return Err("reduction kernel dispatch received no nodes".to_string());
    }
    Ok(())
}
