use std::collections::HashMap;

use crate::ir::{
    CudaBackend, Graph, Tensor, TrainConfig, TrainError, TrainResult, TrainSample, ValueId,
    train_graph_with_backend,
};

/// Trains a graph using the CUDA backend shortcut.
///
/// # Errors
/// Returns [`TrainError`] when CUDA backend execution fails, backward graph
/// creation fails, or optimizer updates cannot be applied.
pub fn train_graph_cuda(
    forward_graph: &Graph,
    loss_value: ValueId,
    initial_parameters: HashMap<String, Tensor>,
    dataset: &[TrainSample],
    config: &TrainConfig,
) -> Result<TrainResult, TrainError> {
    let backend = CudaBackend;
    train_graph_with_backend(
        forward_graph,
        loss_value,
        initial_parameters,
        dataset,
        config,
        &backend,
    )
}
