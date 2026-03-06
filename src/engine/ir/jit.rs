/// Just-In-Time (JIT) compilation for hot paths in Volta IR.
///
/// The JIT system tracks invocation counts per graph fingerprint and automatically
/// promotes graphs to compiled programs after a configurable warmup threshold.
///
/// Usage:
/// ```text
/// let mut jit = JitCache::new(JitConfig::default());
/// // On each forward pass:
/// if let Some(compiled) = jit.get_or_compile(&graph, &backend).ok() {
///     // use compiled program
/// }
/// ```
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::ir::{
    Backend, BackendError, CompiledProgram, DeterminismLevel, Graph, build_execution_plan,
    graph_fingerprint,
};

/// Configuration for the JIT compiler.
#[derive(Debug, Clone)]
pub struct JitConfig {
    /// Number of invocations before a graph is compiled ("heated up").
    /// Default: 3
    pub warmup_threshold: usize,
    /// Determinism level for compiled programs.
    pub determinism: DeterminismLevel,
    /// Maximum number of compiled programs to keep in cache.
    pub max_cache_entries: usize,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            warmup_threshold: 3,
            determinism: DeterminismLevel::Balanced,
            max_cache_entries: 64,
        }
    }
}

#[derive(Debug)]
enum JitEntry {
    /// Graph has been invoked `count` times but not yet compiled.
    Warming { count: usize },
    /// Graph has been compiled and is ready for execution.
    Compiled(CompiledProgram),
}

/// JIT cache that compiles hot graphs on demand.
pub struct JitCache {
    config: JitConfig,
    entries: HashMap<u64, JitEntry>,
}

impl JitCache {
    pub fn new(config: JitConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
        }
    }

    /// Record an invocation of the graph and return a compiled program if ready.
    ///
    /// Returns `Ok(Some(program))` when the graph has been compiled.
    /// Returns `Ok(None)` during warmup phase.
    /// Returns `Err(...)` if compilation fails.
    pub fn get_or_compile(
        &mut self,
        graph: &Graph,
        backend: &dyn Backend,
    ) -> Result<Option<CompiledProgram>, BackendError> {
        use std::collections::HashSet;

        let fp = graph_fingerprint(graph);

        match self.entries.get_mut(&fp) {
            Some(JitEntry::Compiled(program)) => {
                return Ok(Some(program.clone()));
            }
            Some(JitEntry::Warming { count }) => {
                *count += 1;
                if *count < self.config.warmup_threshold {
                    return Ok(None);
                }
                // Threshold reached — fall through to compile
            }
            None => {
                if self.config.warmup_threshold <= 1 {
                    // Compile immediately
                } else {
                    self.entries.insert(fp, JitEntry::Warming { count: 1 });
                    return Ok(None);
                }
            }
        }

        // Compile the graph
        let plan = build_execution_plan(graph, &HashSet::new()).map_err(|e| BackendError {
            message: format!("JIT: failed to build execution plan: {}", e.message),
        })?;

        let compiled = backend.compile(&plan)?;

        // Evict oldest entry if over capacity
        if self.entries.len() >= self.config.max_cache_entries {
            // Simple eviction: remove first entry
            if let Some(key) = self.entries.keys().next().copied() {
                self.entries.remove(&key);
            }
        }

        self.entries
            .insert(fp, JitEntry::Compiled(compiled.clone()));
        Ok(Some(compiled))
    }

    /// Force-compile a graph immediately (bypass warmup).
    pub fn force_compile(
        &mut self,
        graph: &Graph,
        backend: &dyn Backend,
    ) -> Result<CompiledProgram, BackendError> {
        use std::collections::HashSet;

        let fp = graph_fingerprint(graph);

        if let Some(JitEntry::Compiled(program)) = self.entries.get(&fp) {
            return Ok(program.clone());
        }

        let plan = build_execution_plan(graph, &HashSet::new()).map_err(|e| BackendError {
            message: format!(
                "JIT force_compile: failed to build execution plan: {}",
                e.message
            ),
        })?;
        let compiled = backend.compile(&plan)?;
        self.entries
            .insert(fp, JitEntry::Compiled(compiled.clone()));
        Ok(compiled)
    }

    /// Check if a graph is already compiled (warmed up).
    pub fn is_compiled(&self, graph: &Graph) -> bool {
        let fp = graph_fingerprint(graph);
        matches!(self.entries.get(&fp), Some(JitEntry::Compiled(_)))
    }

    /// Return current warmup count for a graph (0 if not seen).
    pub fn warmup_count(&self, graph: &Graph) -> usize {
        let fp = graph_fingerprint(graph);
        match self.entries.get(&fp) {
            Some(JitEntry::Warming { count }) => *count,
            Some(JitEntry::Compiled(_)) => self.config.warmup_threshold,
            None => 0,
        }
    }

    /// Remove a compiled graph from cache (forces recompilation on next use).
    pub fn invalidate(&mut self, graph: &Graph) {
        let fp = graph_fingerprint(graph);
        self.entries.remove(&fp);
    }

    /// Clear the entire JIT cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of compiled entries in cache.
    pub fn compiled_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| matches!(e, JitEntry::Compiled(_)))
            .count()
    }

    /// Number of graphs currently in warmup phase.
    pub fn warming_count(&self) -> usize {
        self.entries
            .values()
            .filter(|e| matches!(e, JitEntry::Warming { .. }))
            .count()
    }
}

/// Thread-safe shared JIT cache.
pub struct SharedJitCache(Arc<Mutex<JitCache>>);

impl SharedJitCache {
    pub fn new(config: JitConfig) -> Self {
        Self(Arc::new(Mutex::new(JitCache::new(config))))
    }

    pub fn get_or_compile(
        &self,
        graph: &Graph,
        backend: &dyn Backend,
    ) -> Result<Option<CompiledProgram>, BackendError> {
        let mut guard = self.0.lock().unwrap_or_else(|p| p.into_inner());
        guard.get_or_compile(graph, backend)
    }

    pub fn force_compile(
        &self,
        graph: &Graph,
        backend: &dyn Backend,
    ) -> Result<CompiledProgram, BackendError> {
        let mut guard = self.0.lock().unwrap_or_else(|p| p.into_inner());
        guard.force_compile(graph, backend)
    }

    pub fn is_compiled(&self, graph: &Graph) -> bool {
        let guard = self.0.lock().unwrap_or_else(|p| p.into_inner());
        guard.is_compiled(graph)
    }

    pub fn clear(&self) {
        let mut guard = self.0.lock().unwrap_or_else(|p| p.into_inner());
        guard.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{CpuBackend, Graph, Op};

    fn simple_graph() -> Graph {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph.add_op(block, Op::Input("x".to_string())).unwrap();
        graph.add_op(block, Op::Output(x)).unwrap();
        graph.bind_input_shape("x", vec![1, 4]);
        graph
    }

    #[test]
    fn jit_compiles_after_threshold() {
        let graph = simple_graph();
        let backend = CpuBackend;
        let mut jit = JitCache::new(JitConfig {
            warmup_threshold: 3,
            ..Default::default()
        });

        // First two calls: warmup
        assert!(jit.get_or_compile(&graph, &backend).unwrap().is_none());
        assert!(jit.get_or_compile(&graph, &backend).unwrap().is_none());

        // Third call: should compile
        let result = jit.get_or_compile(&graph, &backend).unwrap();
        assert!(result.is_some(), "Should have compiled after threshold");
        assert!(jit.is_compiled(&graph));
    }

    #[test]
    fn jit_threshold_one_compiles_immediately() {
        let graph = simple_graph();
        let backend = CpuBackend;
        let mut jit = JitCache::new(JitConfig {
            warmup_threshold: 1,
            ..Default::default()
        });

        let result = jit.get_or_compile(&graph, &backend).unwrap();
        assert!(
            result.is_some(),
            "Should compile immediately with threshold=1"
        );
    }

    #[test]
    fn force_compile_bypasses_warmup() {
        let graph = simple_graph();
        let backend = CpuBackend;
        let mut jit = JitCache::new(JitConfig {
            warmup_threshold: 100,
            ..Default::default()
        });

        let compiled = jit.force_compile(&graph, &backend).unwrap();
        assert!(jit.is_compiled(&graph));
        // Subsequent get_or_compile should return the compiled program
        let result = jit.get_or_compile(&graph, &backend).unwrap();
        assert_eq!(result.unwrap().fingerprint, compiled.fingerprint);
    }

    #[test]
    fn invalidate_forces_recompilation() {
        let graph = simple_graph();
        let backend = CpuBackend;
        let mut jit = JitCache::new(JitConfig {
            warmup_threshold: 1,
            ..Default::default()
        });

        jit.force_compile(&graph, &backend).unwrap();
        assert!(jit.is_compiled(&graph));

        jit.invalidate(&graph);
        assert!(!jit.is_compiled(&graph));
    }

    #[test]
    fn compiled_count_and_warming_count() {
        let graph = simple_graph();
        let backend = CpuBackend;
        let mut jit = JitCache::new(JitConfig {
            warmup_threshold: 5,
            ..Default::default()
        });

        jit.get_or_compile(&graph, &backend).unwrap();
        assert_eq!(jit.warming_count(), 1);
        assert_eq!(jit.compiled_count(), 0);

        jit.force_compile(&graph, &backend).unwrap();
        assert_eq!(jit.compiled_count(), 1);
        assert_eq!(jit.warming_count(), 0);
    }
}
