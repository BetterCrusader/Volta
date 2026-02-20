#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use crate::ir::{
        AlgebraicSimplificationPass, ConstantFoldingPass, CsePass, DcePass,
        DeadTensorEliminationPass, ElementwiseFusionPass, GradientFusionPass, Graph,
        LayoutAwareOptimizationPass, Op, OptimizerConfig, Tensor, TensorConstantPropagationPass,
        TrainConfig, TrainSample, build_execution_plan, build_reverse_graph, graph_fingerprint,
        run_verified_pass, verify_graph,
    };

    const PASS_COUNT: usize = 9;

    #[test]
    fn fuzz_ssa_graphs_with_verifier_guards() {
        for seed in 0_u64..200 {
            let mut graph = build_fuzz_graph(seed, 24);
            verify_graph(&graph).expect("fuzz graph must verify before passes");

            let mut order = (0..PASS_COUNT).collect::<Vec<_>>();
            shuffle_in_place(&mut order, seed ^ 0xA5A5_1024_u64);
            run_pass_sequence(&mut graph, &order);

            verify_graph(&graph).expect("fuzz graph must verify after passes");
        }
    }

    #[test]
    fn pass_order_chaos_mode_is_repeatable_per_seed() {
        let base = build_fuzz_graph(999, 30);

        for seed in 1_u64..40 {
            let mut order = (0..PASS_COUNT).collect::<Vec<_>>();
            shuffle_in_place(&mut order, seed * 31 + 7);

            let mut g1 = base.clone();
            let mut g2 = base.clone();
            run_pass_sequence(&mut g1, &order);
            run_pass_sequence(&mut g2, &order);

            let fp1 = graph_fingerprint(&g1);
            let fp2 = graph_fingerprint(&g2);
            assert_eq!(fp1, fp2);
        }
    }

    #[test]
    fn memory_pressure_plan_is_stable() {
        let start = std::time::Instant::now();
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, mut current) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![64, 64],
                    data: vec![1.0; 64 * 64],
                },
            )
            .expect("add op should succeed");

        for _ in 0..40 {
            let (_, rhs) = graph
                .add_op(
                    block,
                    Op::ConstTensor {
                        shape: vec![64, 64],
                        data: vec![0.5; 64 * 64],
                    },
                )
                .expect("add op should succeed");
            let (_, out) = graph
                .add_op(block, Op::Add(current, rhs))
                .expect("add op should succeed");
            current = out;
        }
        graph
            .add_op(block, Op::Output(current))
            .expect("add op should succeed");

        let plan_a = build_execution_plan(&graph, &HashSet::new()).expect("plan should pass");
        let plan_b = build_execution_plan(&graph, &HashSet::new()).expect("plan should pass");

        assert_eq!(plan_a.allocation.peak_bytes, plan_b.allocation.peak_bytes);
        assert_eq!(plan_a.schedule.ordered_nodes, plan_b.schedule.ordered_nodes);

        let mut allocation_signature = plan_a
            .allocation
            .buffer_map
            .iter()
            .map(|(value, buffer)| format!("{}:{}", value.0, buffer.0))
            .collect::<Vec<_>>();
        allocation_signature.sort();

        println!(
            "[freeze-heavy:memory-pressure] elapsed_ms={} schedule_hash={} peak_bytes={} allocation_signature={}",
            start.elapsed().as_millis(),
            crate::ir::schedule_hash(&plan_a.schedule),
            plan_a.allocation.peak_bytes,
            allocation_signature.join(";")
        );
    }

    #[test]
    fn deep_autograd_chain_preserves_forward_graph() {
        let mut forward = Graph::new();
        let block = forward.create_block();

        let (_, x) = forward
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, w) = forward
            .add_op(block, Op::Parameter("w".to_string()))
            .expect("add op should succeed");

        let mut current = forward
            .add_op(block, Op::MatMul(x, w))
            .expect("add op should succeed")
            .1;

        for _ in 0..220 {
            let (_, plus) = forward
                .add_op(block, Op::Add(current, current))
                .expect("add op should succeed");
            let (_, relu) = forward
                .add_op(block, Op::Relu(plus))
                .expect("add op should succeed");
            current = relu;
        }

        let (_, loss) = forward
            .add_op(block, Op::Output(current))
            .expect("add op should succeed");

        let before_nodes = forward.nodes.len();
        let backward = build_reverse_graph(&forward, loss, &[w]).expect("autograd should pass");
        assert_eq!(forward.nodes.len(), before_nodes);
        verify_graph(&backward.backward).expect("backward graph must verify");
    }

    #[test]
    fn deterministic_training_repeats_match() {
        let (graph, loss, params, dataset) = train_fixture();
        let cfg = TrainConfig {
            epochs: 8,
            optimizer: OptimizerConfig::Sgd { lr: 0.01 },
        };

        let a = crate::ir::train_graph(&graph, loss, params.clone(), &dataset, &cfg)
            .expect("train should pass");
        let b = crate::ir::train_graph(&graph, loss, params, &dataset, &cfg)
            .expect("train should pass");

        assert!((a.final_loss - b.final_loss).abs() < 1e-9);
    }

    #[test]
    #[ignore]
    fn fuzz_ssa_graphs_5000_heavy() {
        let start = std::time::Instant::now();
        let mut fingerprint_xor = 0_u64;
        let mut stable_schedule_hash = None;
        let mut stable_peak_bytes = None;

        for seed in 0_u64..5000 {
            let mut graph = build_fuzz_graph(seed, 28);
            verify_graph(&graph).expect("graph must verify");

            let mut order = (0..PASS_COUNT).collect::<Vec<_>>();
            shuffle_in_place(&mut order, seed ^ 0xDEAD_BEEF_u64);
            run_pass_sequence(&mut graph, &order);

            verify_graph(&graph).expect("graph must verify after pipeline");

            let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should pass");
            let current_schedule_hash = crate::ir::schedule_hash(&plan.schedule);
            let current_peak = plan.allocation.peak_bytes;

            if stable_schedule_hash.is_none() {
                stable_schedule_hash = Some(current_schedule_hash);
            }
            if stable_peak_bytes.is_none() {
                stable_peak_bytes = Some(current_peak);
            }

            fingerprint_xor ^= graph_fingerprint(&graph);
        }

        println!(
            "[freeze-heavy:fuzz5000] elapsed_ms={} fingerprint_xor={} schedule_hash_baseline={} peak_bytes_baseline={}",
            start.elapsed().as_millis(),
            fingerprint_xor,
            stable_schedule_hash.unwrap_or(0),
            stable_peak_bytes.unwrap_or(0)
        );
    }

    #[test]
    #[ignore]
    fn long_run_determinism_100x50_heavy() {
        let start = std::time::Instant::now();
        let (graph, loss, params, dataset) = train_fixture();
        let cfg = TrainConfig {
            epochs: 100,
            optimizer: OptimizerConfig::Sgd { lr: 0.005 },
        };

        let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should pass");
        let schedule_hash = crate::ir::schedule_hash(&plan.schedule);
        let allocation_peak = plan.allocation.peak_bytes;
        let mut allocation_signature = plan
            .allocation
            .buffer_map
            .iter()
            .map(|(value, buffer)| format!("{}:{}", value.0, buffer.0))
            .collect::<Vec<_>>();
        allocation_signature.sort();
        let allocation_signature = allocation_signature.join(";");

        let mut baseline = None;
        let mut baseline_fingerprint = None;
        for _ in 0..50 {
            let iter_plan =
                build_execution_plan(&graph, &HashSet::new()).expect("plan should pass");
            assert_eq!(schedule_hash, crate::ir::schedule_hash(&iter_plan.schedule));
            assert_eq!(allocation_peak, iter_plan.allocation.peak_bytes);
            let mut iter_sig = iter_plan
                .allocation
                .buffer_map
                .iter()
                .map(|(value, buffer)| format!("{}:{}", value.0, buffer.0))
                .collect::<Vec<_>>();
            iter_sig.sort();
            assert_eq!(allocation_signature, iter_sig.join(";"));

            let result = crate::ir::train_graph(&graph, loss, params.clone(), &dataset, &cfg)
                .expect("train should pass");
            match baseline {
                None => baseline = Some(result.final_loss),
                Some(expected) => assert!((expected - result.final_loss).abs() < 1e-6),
            }

            let mut keys = result.final_parameters.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            use std::hash::{Hash, Hasher};
            for key in keys {
                key.hash(&mut hasher);
                if let Some(tensor) = result.final_parameters.get(&key) {
                    tensor.shape.hash(&mut hasher);
                    for value in &tensor.data {
                        value.to_bits().hash(&mut hasher);
                    }
                }
            }
            let fp = hasher.finish();
            match baseline_fingerprint {
                None => baseline_fingerprint = Some(fp),
                Some(expected) => assert_eq!(expected, fp),
            }
        }

        println!(
            "[freeze-heavy:train100x50] elapsed_ms={} schedule_hash={} peak_bytes={} allocation_signature={} param_fingerprint={}",
            start.elapsed().as_millis(),
            schedule_hash,
            allocation_peak,
            allocation_signature,
            baseline_fingerprint.unwrap_or(0)
        );
    }

    fn run_pass_sequence(graph: &mut Graph, order: &[usize]) {
        for id in order {
            match *id {
                0 => {
                    let mut pass = ConstantFoldingPass::new();
                    run_verified_pass(&mut pass, graph).expect("pass should verify");
                }
                1 => {
                    let mut pass = AlgebraicSimplificationPass::new();
                    run_verified_pass(&mut pass, graph).expect("pass should verify");
                }
                2 => {
                    let mut pass = CsePass::new();
                    run_verified_pass(&mut pass, graph).expect("pass should verify");
                }
                3 => {
                    let mut pass = DcePass::new();
                    run_verified_pass(&mut pass, graph).expect("pass should verify");
                }
                4 => {
                    let mut pass = ElementwiseFusionPass::new();
                    run_verified_pass(&mut pass, graph).expect("pass should verify");
                }
                5 => {
                    let mut pass = DeadTensorEliminationPass::new();
                    run_verified_pass(&mut pass, graph).expect("pass should verify");
                }
                6 => {
                    let mut pass = TensorConstantPropagationPass::new();
                    run_verified_pass(&mut pass, graph).expect("pass should verify");
                }
                7 => {
                    let mut pass = GradientFusionPass::new();
                    run_verified_pass(&mut pass, graph).expect("pass should verify");
                }
                8 => {
                    let mut pass = LayoutAwareOptimizationPass::new();
                    run_verified_pass(&mut pass, graph).expect("pass should verify");
                }
                _ => panic!("invalid pass id"),
            }
        }
    }

    fn build_fuzz_graph(seed: u64, steps: usize) -> Graph {
        let mut rng = Lcg::new(seed);
        let mut graph = Graph::new();
        let block = graph.create_block();

        let mut scalars = Vec::new();
        let mut tensors = Vec::new();

        for _ in 0..3 {
            let value = (rng.next_u64() % 17) as i64 + 1;
            let (_, id) = graph
                .add_op(block, Op::ConstInt(value))
                .expect("add op should succeed");
            scalars.push(id);
        }

        for _ in 0..3 {
            let data = (0..4)
                .map(|_| ((rng.next_u64() % 7) as f32) - 3.0)
                .collect::<Vec<_>>();
            let (_, id) = graph
                .add_op(
                    block,
                    Op::ConstTensor {
                        shape: vec![4],
                        data,
                    },
                )
                .expect("add op should succeed");
            tensors.push(id);
        }

        for _ in 0..steps {
            let choice = (rng.next_u64() % 8) as usize;
            match choice {
                0..=3 => {
                    let a = scalars[(rng.next_u64() as usize) % scalars.len()];
                    let b = scalars[(rng.next_u64() as usize) % scalars.len()];
                    let op = match choice {
                        0 => Op::Add(a, b),
                        1 => Op::Sub(a, b),
                        2 => Op::Mul(a, b),
                        _ => Op::Div(a, b),
                    };
                    let (_, out) = graph.add_op(block, op).expect("add op should succeed");
                    scalars.push(out);
                }
                4 | 5 => {
                    let a = tensors[(rng.next_u64() as usize) % tensors.len()];
                    let b = tensors[(rng.next_u64() as usize) % tensors.len()];
                    let op = if choice == 4 {
                        Op::Add(a, b)
                    } else {
                        Op::Mul(a, b)
                    };
                    let (_, out) = graph.add_op(block, op).expect("add op should succeed");
                    tensors.push(out);
                }
                6 => {
                    let a = tensors[(rng.next_u64() as usize) % tensors.len()];
                    let (_, out) = graph
                        .add_op(block, Op::Relu(a))
                        .expect("add op should succeed");
                    tensors.push(out);
                }
                _ => {
                    let a = tensors[(rng.next_u64() as usize) % tensors.len()];
                    let (_, out) = graph
                        .add_op(block, Op::Neg(a))
                        .expect("add op should succeed");
                    tensors.push(out);
                }
            }
        }

        if rng.next_u64() & 1 == 0 {
            let target = scalars[(rng.next_u64() as usize) % scalars.len()];
            graph
                .add_op(block, Op::Output(target))
                .expect("add op should succeed");
        } else {
            let target = tensors[(rng.next_u64() as usize) % tensors.len()];
            graph
                .add_op(block, Op::Output(target))
                .expect("add op should succeed");
        }

        graph
    }

    fn train_fixture() -> (
        Graph,
        crate::ir::ValueId,
        HashMap<String, Tensor>,
        Vec<TrainSample>,
    ) {
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

        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1, 1], vec![0.0]).expect("tensor"),
        );

        let mut dataset = Vec::new();
        for (xv, yv) in [(1.0_f32, 2.0_f32), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)] {
            let mut inputs = HashMap::new();
            inputs.insert(
                "x".to_string(),
                Tensor::new(vec![1, 1], vec![xv]).expect("tensor"),
            );
            inputs.insert(
                "y".to_string(),
                Tensor::new(vec![1, 1], vec![yv]).expect("tensor"),
            );
            dataset.push(TrainSample { inputs });
        }

        (graph, loss, params, dataset)
    }

    fn shuffle_in_place(values: &mut [usize], seed: u64) {
        let mut rng = Lcg::new(seed);
        for i in (1..values.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            values.swap(i, j);
        }
    }

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed | 1 }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }
    }
}
