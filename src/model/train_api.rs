use std::collections::HashMap;

use crate::ir::{train_graph, OptimizerConfig, Tensor, TrainConfig, TrainSample};

use crate::model::{
    load_checkpoint, save_checkpoint, BatchIterator, CompiledModel, Dataset, Example,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReproducibilityMode {
    Deterministic,
    Fast,
}

#[derive(Debug, Clone)]
pub struct TrainApiConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub shuffle: bool,
    pub shuffle_seed: u64,
    pub optimizer: OptimizerConfig,
    pub reproducibility: ReproducibilityMode,
    pub checkpoint_path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TrainApiResult {
    pub final_parameters: HashMap<String, Tensor>,
    pub final_loss: f32,
}

#[derive(Debug, Clone)]
pub struct TrainApiError {
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct TinyTransformerFixtureDataset;

impl Dataset for TinyTransformerFixtureDataset {
    fn len(&self) -> usize {
        0
    }

    fn example(&self, _index: usize) -> Result<Example, TrainApiError> {
        Err(TrainApiError {
            message: "tiny transformer fixture dataset is not implemented yet".to_string(),
        })
    }
}

pub fn build_tiny_transformer_fixture_for_tests() -> (
    CompiledModel,
    TinyTransformerFixtureDataset,
    TrainApiConfig,
    HashMap<String, Tensor>,
) {
    unimplemented!("tiny transformer fixture is introduced in Task 2");
}

pub fn infer(
    _model: &CompiledModel,
    _parameters: &HashMap<String, Tensor>,
    _inputs: &HashMap<String, Tensor>,
) -> Result<Tensor, TrainApiError> {
    unimplemented!("inference API is introduced in Task 3");
}

pub fn train<D: Dataset>(
    model: &CompiledModel,
    dataset: &D,
    config: &TrainApiConfig,
) -> Result<TrainApiResult, TrainApiError> {
    let mut parameters = model.parameters.clone();
    if let Some(path) = &config.checkpoint_path {
        let loaded = load_checkpoint(path)?;
        for (name, tensor) in loaded {
            parameters.insert(name, tensor);
        }
    }

    let shuffle = match config.reproducibility {
        ReproducibilityMode::Deterministic => config.shuffle,
        ReproducibilityMode::Fast => config.shuffle,
    };

    let mut samples = Vec::<TrainSample>::new();
    for batch in BatchIterator::new(
        dataset.len(),
        config.batch_size,
        shuffle,
        config.shuffle_seed,
    ) {
        for index in batch {
            let Example { inputs } = dataset.example(index)?;
            samples.push(TrainSample { inputs });
        }
    }

    let Some(loss) = model.loss else {
        return Err(TrainApiError {
            message: "Compiled model has no loss node".to_string(),
        });
    };

    let result = train_graph(
        &model.graph,
        loss,
        parameters,
        &samples,
        &TrainConfig {
            epochs: config.epochs,
            optimizer: config.optimizer.clone(),
        },
    )
    .map_err(|err| TrainApiError {
        message: format!("Train pipeline failed: {}", err.message),
    })?;

    if let Some(path) = &config.checkpoint_path {
        save_checkpoint(path, &result.final_parameters)?;
    }

    Ok(TrainApiResult {
        final_parameters: result.final_parameters,
        final_loss: result.final_loss,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ir::{OptimizerConfig, Tensor};
    use crate::model::{
        train, CompiledModel, Dataset, Example, ReproducibilityMode, TensorShape, TrainApiConfig,
    };

    struct TinyDataset {
        rows: Vec<(f32, f32)>,
    }

    impl Dataset for TinyDataset {
        fn len(&self) -> usize {
            self.rows.len()
        }

        fn example(&self, index: usize) -> Result<Example, crate::model::TrainApiError> {
            let (x, y) = self.rows[index];
            let mut inputs = HashMap::new();
            inputs.insert(
                "x".to_string(),
                Tensor::new(vec![1, 1], vec![x]).expect("valid tensor"),
            );
            inputs.insert(
                "y".to_string(),
                Tensor::new(vec![1, 1], vec![y]).expect("valid tensor"),
            );
            Ok(Example { inputs })
        }
    }

    #[test]
    fn deterministic_mode_produces_stable_result() {
        let (model, dataset) = fixture();
        let config = TrainApiConfig {
            epochs: 20,
            batch_size: 2,
            shuffle: true,
            shuffle_seed: 7,
            optimizer: OptimizerConfig::Sgd { lr: 0.01 },
            reproducibility: ReproducibilityMode::Deterministic,
            checkpoint_path: None,
        };

        let a = train(&model, &dataset, &config).expect("train should pass");
        let b = train(&model, &dataset, &config).expect("train should pass");
        assert!((a.final_loss - b.final_loss).abs() < 1e-9);
    }

    fn fixture() -> (CompiledModel, TinyDataset) {
        let mut graph = crate::ir::Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, crate::ir::Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, w) = graph
            .add_op(block, crate::ir::Op::Parameter("w".to_string()))
            .expect("add op should succeed");
        let (_, y) = graph
            .add_op(block, crate::ir::Op::Input("y".to_string()))
            .expect("add op should succeed");
        let (_, pred) = graph
            .add_op(block, crate::ir::Op::MatMul(x, w))
            .expect("add op should succeed");
        let (_, diff) = graph
            .add_op(block, crate::ir::Op::Sub(pred, y))
            .expect("add op should succeed");
        let (_, sq) = graph
            .add_op(block, crate::ir::Op::Mul(diff, diff))
            .expect("add op should succeed");
        let (_, loss) = graph
            .add_op(block, crate::ir::Op::Output(sq))
            .expect("add op should succeed");

        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1, 1], vec![0.0]).expect("valid tensor"),
        );
        let mut param_values = HashMap::new();
        param_values.insert("w".to_string(), w);

        let model = CompiledModel {
            graph,
            output: pred,
            output_shape: TensorShape(vec![1, 1]),
            loss: Some(loss),
            parameters: params,
            parameter_values: param_values,
        };

        let dataset = TinyDataset {
            rows: vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)],
        };

        (model, dataset)
    }
}
