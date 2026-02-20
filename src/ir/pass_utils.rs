use crate::ir::{Graph, verify_graph};

pub fn run_with_verifier_guard(graph: &mut Graph, run: impl FnOnce(&mut Graph)) {
    debug_assert!(verify_graph(graph).is_ok());
    run(graph);
    debug_assert!(verify_graph(graph).is_ok());
}
