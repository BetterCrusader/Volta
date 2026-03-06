use crate::ir::Graph;

pub trait Pass {
    fn run(&mut self, graph: &mut Graph);
    fn name(&self) -> &str;
}

/// A named group of passes that run together.
pub struct PassGroup {
    pub name: String,
    passes: Vec<Box<dyn Pass>>,
    /// If > 1, repeat the entire group up to this many times (fix-point iteration).
    pub repeat: usize,
}

impl PassGroup {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passes: Vec::new(),
            repeat: 1,
        }
    }

    pub fn with_repeat(mut self, n: usize) -> Self {
        self.repeat = n.max(1);
        self
    }

    pub fn add(mut self, pass: Box<dyn Pass>) -> Self {
        self.passes.push(pass);
        self
    }

    pub fn run(&mut self, graph: &mut Graph) {
        for _ in 0..self.repeat {
            for pass in &mut self.passes {
                pass.run(graph);
            }
        }
    }
}

/// PassManager: orchestrates groups of optimization passes over a graph.
///
/// Supports:
/// - Sequential pass groups
/// - Repeated execution (fix-point iteration) per group
/// - Optional verification between passes (with `debug-verify` feature)
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    groups: Vec<PassGroup>,
}

impl PassManager {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            groups: Vec::new(),
        }
    }

    /// Add a single pass.
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Add a named pass group.
    pub fn add_group(&mut self, group: PassGroup) {
        self.groups.push(group);
    }

    /// Build a standard optimization pipeline.
    /// Applies: CSE → ConstantFolding → AlgebraicSimplification → DCE → DeadTensorElimination.
    pub fn build_default() -> Self {
        use crate::ir::{
            AlgebraicSimplificationPass, ConstantFoldingPass, CsePass, DcePass,
            DeadTensorEliminationPass,
        };
        let mut pm = Self::new();
        pm.add_pass(Box::new(CsePass));
        pm.add_pass(Box::new(ConstantFoldingPass));
        pm.add_pass(Box::new(AlgebraicSimplificationPass));
        pm.add_pass(Box::new(DcePass));
        pm.add_pass(Box::new(DeadTensorEliminationPass));
        pm
    }

    /// Build a fusion-focused pipeline.
    pub fn build_fusion() -> Self {
        use crate::ir::{CsePass, DcePass, ElementwiseFusionPass, GradientFusionPass};
        let mut pm = Self::new();
        pm.add_pass(Box::new(CsePass));
        pm.add_pass(Box::new(ElementwiseFusionPass));
        pm.add_pass(Box::new(GradientFusionPass));
        pm.add_pass(Box::new(DcePass));
        pm
    }

    /// Run all passes (individual passes first, then groups).
    pub fn run(&mut self, graph: &mut Graph) {
        for pass in &mut self.passes {
            #[cfg(feature = "debug-verify")]
            {
                crate::ir::verifier::verify_graph(graph).unwrap_or_else(|e| {
                    panic!("Graph verify failed before {}: {:?}", pass.name(), e)
                });
            }

            pass.run(graph);

            #[cfg(feature = "debug-verify")]
            {
                crate::ir::verifier::verify_graph(graph).unwrap_or_else(|e| {
                    panic!("Graph verify failed after {}: {:?}", pass.name(), e)
                });
            }
        }

        for group in &mut self.groups {
            group.run(graph);
        }
    }

    /// Run passes and return count of passes executed.
    pub fn run_counted(&mut self, graph: &mut Graph) -> usize {
        let n = self.passes.len()
            + self
                .groups
                .iter()
                .map(|g| g.passes.len() * g.repeat)
                .sum::<usize>();
        self.run(graph);
        n
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}
