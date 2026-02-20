use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use volta::ir::{
    CpuBackend, CudaBackend, Graph, Op, OptimizerConfig, Tensor, TrainConfig, TrainSample,
    train_graph_with_backend,
};

#[test]
fn cpu_and_cuda_optimizers_match_state_in_strict_mode() {
    with_determinism("strict", || {
        let (graph, loss) = build_linear_mse_graph();
        let dataset = vec![sample(1.0, 2.0), sample(2.0, 4.0), sample(3.0, 6.0)];

        let variants = vec![
            TrainConfig {
                epochs: 30,
                optimizer: OptimizerConfig::Sgd { lr: 0.01 },
            },
            TrainConfig {
                epochs: 30,
                optimizer: OptimizerConfig::Adam {
                    lr: 0.05,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                },
            },
        ];

        for config in variants {
            let mut params_cpu = HashMap::new();
            params_cpu.insert(
                "w".to_string(),
                Tensor::new(vec![1, 1], vec![0.0]).expect("valid tensor"),
            );
            let params_cuda = params_cpu.clone();

            let cpu = CpuBackend;
            let cuda = CudaBackend;

            let cpu_result =
                train_graph_with_backend(&graph, loss, params_cpu, &dataset, &config, &cpu)
                    .expect("cpu training should pass");
            let cuda_result =
                train_graph_with_backend(&graph, loss, params_cuda, &dataset, &config, &cuda)
                    .expect("cuda training should pass");

            assert_eq!(cpu_result.final_parameters, cuda_result.final_parameters);
            assert_eq!(cpu_result.optimizer_state, cuda_result.optimizer_state);
            assert_eq!(
                cpu_result.final_loss.to_bits(),
                cuda_result.final_loss.to_bits()
            );
        }
    });
}

fn build_linear_mse_graph() -> (Graph, volta::ir::ValueId) {
    let mut graph = Graph::new();
    let block = graph.create_block();
    let (_, x) = graph
        .add_op(block, Op::Input("x".to_string()))
        .expect("add op should succeed");
    let (_, w) = graph
        .add_op(block, Op::Parameter("w".to_string()))
        .expect("add op should succeed");
    let (_, y) = graph
        .add_op(block, Op::Input("y".to_string()))
        .expect("add op should succeed");
    let (_, pred) = graph
        .add_op(block, Op::MatMul(x, w))
        .expect("add op should succeed");
    let (_, diff) = graph
        .add_op(block, Op::Sub(pred, y))
        .expect("add op should succeed");
    let (_, sq) = graph
        .add_op(block, Op::Mul(diff, diff))
        .expect("add op should succeed");
    let (_, loss) = graph
        .add_op(block, Op::Output(sq))
        .expect("add op should succeed");
    (graph, loss)
}

fn sample(x: f32, y: f32) -> TrainSample {
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::new(vec![1, 1], vec![x]).expect("valid tensor"),
    );
    inputs.insert(
        "y".to_string(),
        Tensor::new(vec![1, 1], vec![y]).expect("valid tensor"),
    );
    TrainSample { inputs }
}

fn with_determinism(level: &str, run: impl FnOnce()) {
    let _guard = match env_lock().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let _restore = EnvVarRestore::set("VOLTA_DETERMINISM", level);
    run();
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
