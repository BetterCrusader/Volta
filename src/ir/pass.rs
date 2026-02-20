use crate::ir::Graph;

pub trait Pass {
    fn run(&mut self, graph: &mut Graph);
}
