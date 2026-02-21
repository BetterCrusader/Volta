use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use volta::ir::cuda::{CudaKernel, lower_plan};
use volta::ir::{CudaBackend, Graph, Op, build_execution_plan};
use volta::model::{build_tiny_transformer_fixture_for_tests, train_with_backend};

#[test]
fn strict_cuda_training_replay_is_bitwise_stable() {
    if !cuda_runtime_available() {
        return;
    }
    with_determinism("strict", || {
        let (model, dataset, train_config, _infer_input) =
            build_tiny_transformer_fixture_for_tests();
        let cuda = CudaBackend;

        let first = train_with_backend(&model, &dataset, &train_config, &cuda)
            .expect("first strict cuda train run should pass");
        let second = train_with_backend(&model, &dataset, &train_config, &cuda)
            .expect("second strict cuda train run should pass");

        assert_eq!(
            first.final_parameters, second.final_parameters,
            "strict cuda replay must be bitwise stable"
        );
        assert_eq!(first.final_loss.to_bits(), second.final_loss.to_bits());
    });
}

#[test]
fn strict_cuda_reduction_topology_is_split_per_node() {
    with_determinism("strict", || {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, w) = graph
            .add_op(block, Op::Parameter("w".to_string()))
            .expect("add parameter should succeed");
        let (_, sum1) = graph
            .add_op(block, Op::Add(w, w))
            .expect("add first add should succeed");
        let (_, sum2) = graph
            .add_op(block, Op::Add(sum1, w))
            .expect("add second add should succeed");
        graph
            .add_op(block, Op::Output(sum2))
            .expect("add output should succeed");
        graph.bind_parameter_shape("w", vec![1, 1]);

        let mut gradients = HashSet::new();
        gradients.insert(sum2);
        let plan = build_execution_plan(&graph, &gradients).expect("plan should build");
        let lowered = lower_plan(&plan).expect("cuda lowering should pass");

        let reduction_nodes = lowered
            .executable_nodes
            .iter()
            .filter(|node| node.kernel == CudaKernel::Reduction)
            .collect::<Vec<_>>();

        assert!(
            !reduction_nodes.is_empty(),
            "expected at least one reduction node"
        );
        assert!(
            reduction_nodes.iter().all(|node| node.nodes.len() == 1),
            "strict mode requires fixed reduction topology with one node per reduction dispatch"
        );
    });
}

fn with_determinism(level: &str, run: impl FnOnce()) {
    let _guard = match env_lock().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let _restore = EnvVarRestore::set("VOLTA_DETERMINISM", level);
    run();
}

fn cuda_runtime_available() -> bool {
    let result = std::panic::catch_unwind(|| volta::ir::cuda::device::CudaDevice::new(0));
    match result {
        Ok(Ok(_)) => true,
        Ok(Err(_)) => false,
        Err(_) => false,
    }
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvVarRestore {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarRestore {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => unsafe {
                std::env::set_var(self.key, value);
            },
            None => unsafe {
                std::env::remove_var(self.key);
            },
        }
    }
}
