use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use volta::ir::{
    Backend, BackendCapabilities, BackendError, BackendKind, CompiledProgram, DeterminismLevel,
    ExecutionPlan, Graph, Op, build_execution_plan, clear_plan_cache, compile_or_get_cached,
};

#[derive(Debug, Clone)]
struct SpyBackend {
    compile_calls: Arc<AtomicUsize>,
}

impl SpyBackend {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                compile_calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

impl Backend for SpyBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend: BackendKind::Cuda,
            supports_inference: true,
            supports_training: true,
            supports_strict_determinism: true,
            default_determinism: DeterminismLevel::Strict,
        }
    }

    fn compile(&self, plan: &ExecutionPlan) -> Result<CompiledProgram, BackendError> {
        self.compile_calls.fetch_add(1, Ordering::SeqCst);
        Ok(CompiledProgram {
            schedule_len: plan.schedule.ordered_nodes.len(),
            peak_bytes: plan.allocation.peak_bytes,
            fingerprint: 0,
        })
    }
}

#[test]
fn infer_reuses_cached_execution_plan_for_same_backend_and_graph() {
    let _guard = cache_lock();
    clear_plan_cache();
    let (backend, calls) = SpyBackend::new();
    let (model, _dataset, _cfg, infer_input) =
        volta::model::build_tiny_transformer_fixture_for_tests();

    let first = volta::model::infer_with_backend(&model, &model.parameters, &infer_input, &backend)
        .expect("first infer should pass");
    let second =
        volta::model::infer_with_backend(&model, &model.parameters, &infer_input, &backend)
            .expect("second infer should pass");

    assert_eq!(first, second);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "second infer must reuse compiled plan from cache"
    );
}

#[test]
fn cache_invalidates_on_shape_change() {
    let _guard = cache_lock();
    clear_plan_cache();
    let (backend, calls) = SpyBackend::new();

    let first_graph = build_input_output_graph_with_shape(vec![1, 32]);
    let second_graph = build_input_output_graph_with_shape(vec![2, 16]);

    let first_plan = build_execution_plan(&first_graph, &std::collections::HashSet::new())
        .expect("first plan should build");
    let second_plan = build_execution_plan(&second_graph, &std::collections::HashSet::new())
        .expect("second plan should build");

    compile_or_get_cached(&first_plan, &backend, DeterminismLevel::Balanced)
        .expect("first compile should pass");
    compile_or_get_cached(&second_plan, &backend, DeterminismLevel::Balanced)
        .expect("shape-changed compile should pass");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "cache must invalidate when shape signature changes"
    );
}

#[test]
fn cache_invalidates_on_determinism_mode_change() {
    let _guard = cache_lock();
    clear_plan_cache();
    let (backend, calls) = SpyBackend::new();

    let graph = build_input_output_graph_with_shape(vec![1, 32]);
    let plan =
        build_execution_plan(&graph, &std::collections::HashSet::new()).expect("plan should build");

    compile_or_get_cached(&plan, &backend, DeterminismLevel::Balanced)
        .expect("balanced compile should pass");
    compile_or_get_cached(&plan, &backend, DeterminismLevel::Strict)
        .expect("strict compile should pass");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "cache must invalidate when determinism mode changes"
    );
}

fn build_input_output_graph_with_shape(shape: Vec<usize>) -> Graph {
    let mut graph = Graph::new();
    let block = graph.create_block();
    let (_, x) = graph
        .add_op(block, Op::Input("x".to_string()))
        .expect("add input should succeed");
    graph
        .add_op(block, Op::Output(x))
        .expect("add output should succeed");
    graph.bind_input_shape("x", shape);
    graph
}

fn cache_lock() -> std::sync::MutexGuard<'static, ()> {
    let lock = CACHE_LOCK.get_or_init(|| Mutex::new(()));
    match lock.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

static CACHE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
