use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::ir::{
    Backend, BackendError, BackendKind, CompiledProgram, DeterminismLevel, ExecutionPlan,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PlanCacheKey {
    backend: BackendKind,
    determinism: DeterminismLevel,
    graph_fingerprint: u64,
    shape_signature_hash: u64,
}

#[derive(Debug, Default)]
struct PlanCache {
    entries: HashMap<PlanCacheKey, CompiledProgram>,
}

pub fn compile_or_get_cached(
    plan: &ExecutionPlan,
    backend: &dyn Backend,
    determinism: DeterminismLevel,
) -> Result<CompiledProgram, BackendError> {
    let key = PlanCacheKey {
        backend: backend.capabilities().backend,
        determinism,
        graph_fingerprint: plan.graph_fingerprint,
        shape_signature_hash: plan.shape_signature_hash,
    };

    {
        let guard = match cache().lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(compiled) = guard.entries.get(&key) {
            return Ok(compiled.clone());
        }
    }

    let compiled = backend.compile(plan)?;

    let mut guard = match cache().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.entries.insert(key, compiled.clone());
    Ok(compiled)
}

pub fn clear_plan_cache() {
    let mut guard = match cache().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.entries.clear();
}

fn cache() -> &'static Mutex<PlanCache> {
    static CACHE: OnceLock<Mutex<PlanCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(PlanCache::default()))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::ir::{Backend, CpuBackend, Graph, Op, build_execution_plan, compile_or_get_cached};

    #[test]
    fn cache_reuses_compiled_program_for_same_key() {
        super::clear_plan_cache();

        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add input should succeed");
        graph
            .add_op(block, Op::Output(x))
            .expect("add output should succeed");
        graph.bind_input_shape("x", vec![1, 1]);

        let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");
        let backend = CpuBackend;
        let first = compile_or_get_cached(&plan, &backend, crate::ir::DeterminismLevel::Balanced)
            .expect("compile should pass");
        let second = compile_or_get_cached(&plan, &backend, crate::ir::DeterminismLevel::Balanced)
            .expect("cache lookup should pass");

        assert_eq!(first.fingerprint, second.fingerprint);
        let direct = backend.compile(&plan).expect("direct compile should pass");
        assert_eq!(first.fingerprint, direct.fingerprint);
    }
}
