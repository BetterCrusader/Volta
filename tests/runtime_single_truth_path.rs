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
            backend: BackendKind::Cpu,
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
fn infer_uses_runtime_gateway_single_path() {
    let (backend, calls) = SpyBackend::new();
    let (model, _dataset, _cfg, infer_input) =
        volta::model::build_tiny_transformer_fixture_for_tests();

    let out = volta::model::infer_with_backend(&model, &model.parameters, &infer_input, &backend)
        .expect("infer should pass through runtime gateway");

    assert_eq!(out.shape, model.output_shape.0);
    assert!(
        calls.load(Ordering::SeqCst) >= 1,
        "runtime gateway should compile plan at least once"
    );
}

#[test]
fn train_uses_runtime_gateway_single_path() {
    let (backend, calls) = SpyBackend::new();
    let (model, dataset, mut cfg, _infer_input) =
        volta::model::build_tiny_transformer_fixture_for_tests();
    cfg.epochs = 1;
    cfg.checkpoint_path = None;

    let result = volta::model::train_with_backend(&model, &dataset, &cfg, &backend)
        .expect("train should pass through runtime gateway");

    assert!(result.final_loss.is_finite());
    assert!(
        calls.load(Ordering::SeqCst) >= 1,
        "runtime gateway should compile at least one execution plan for training"
    );
}
