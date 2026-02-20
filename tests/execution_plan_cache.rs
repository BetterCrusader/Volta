use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use volta::ir::{
    Backend, BackendCapabilities, BackendError, BackendKind, CompiledProgram, DeterminismLevel,
    ExecutionPlan,
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
            backend: BackendKind::Metal,
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
