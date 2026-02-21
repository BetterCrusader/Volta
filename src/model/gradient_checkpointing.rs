use std::collections::{HashMap, HashSet};

use crate::ir::{Op, ValueId, plan_memory};
use crate::model::CompiledModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientCheckpointingConfig {
    pub interval_nodes: usize,
    pub min_tensor_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecomputeSegment {
    pub start_node: usize,
    pub end_node: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradientCheckpointPlan {
    pub checkpoint_values: Vec<ValueId>,
    pub recompute_segments: Vec<RecomputeSegment>,
    pub estimated_saved_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct GradientCheckpointingError {
    pub message: String,
}

pub fn plan_gradient_checkpointing(
    model: &CompiledModel,
    config: &GradientCheckpointingConfig,
) -> Result<GradientCheckpointPlan, GradientCheckpointingError> {
    if config.interval_nodes == 0 {
        return Err(GradientCheckpointingError {
            message: "interval_nodes must be greater than zero".to_string(),
        });
    }

    let memory_plan = plan_memory(&model.graph).map_err(|err| GradientCheckpointingError {
        message: format!("memory planning failed: {}", err.message),
    })?;

    let bytes_by_value = memory_plan
        .values
        .iter()
        .map(|value| (value.value, value.estimated_bytes))
        .collect::<HashMap<_, _>>();

    let mut checkpoint_values = Vec::new();
    for (node_index, node) in model.graph.nodes.iter().enumerate() {
        if node_index % config.interval_nodes != 0 {
            continue;
        }
        if !is_checkpoint_candidate(&node.op) {
            continue;
        }

        let bytes = bytes_by_value.get(&node.output).copied().unwrap_or(0);
        if bytes < config.min_tensor_bytes {
            continue;
        }

        checkpoint_values.push(node.output);
    }

    let checkpoint_set = checkpoint_values.iter().copied().collect::<HashSet<_>>();

    let mut estimated_activation_bytes = 0usize;
    for node in &model.graph.nodes {
        if !is_checkpoint_candidate(&node.op) {
            continue;
        }
        estimated_activation_bytes = estimated_activation_bytes
            .saturating_add(bytes_by_value.get(&node.output).copied().unwrap_or(0));
    }

    let checkpoint_bytes = checkpoint_values
        .iter()
        .map(|value| bytes_by_value.get(value).copied().unwrap_or(0))
        .sum::<usize>();

    let estimated_saved_bytes = if checkpoint_set.is_empty() {
        0
    } else {
        estimated_activation_bytes.saturating_sub(checkpoint_bytes)
    };

    let checkpoint_nodes = checkpoint_values
        .iter()
        .map(|value| value.0)
        .collect::<Vec<_>>();
    let mut recompute_segments = Vec::new();
    for pair in checkpoint_nodes.windows(2) {
        let start = pair[0].saturating_add(1);
        let end = pair[1];
        if start <= end {
            recompute_segments.push(RecomputeSegment {
                start_node: start,
                end_node: end,
            });
        }
    }

    Ok(GradientCheckpointPlan {
        checkpoint_values,
        recompute_segments,
        estimated_saved_bytes,
    })
}

fn is_checkpoint_candidate(op: &Op) -> bool {
    !matches!(
        op,
        Op::Input(_) | Op::Parameter(_) | Op::Output(_) | Op::Removed
    )
}

#[cfg(test)]
mod tests {
    use crate::model::{
        GradientCheckpointingConfig, build_tiny_transformer_fixture_for_tests,
        plan_gradient_checkpointing,
    };

    #[test]
    fn rejects_zero_interval() {
        let (model, _dataset, _cfg, _infer_input) = build_tiny_transformer_fixture_for_tests();
        let err = plan_gradient_checkpointing(
            &model,
            &GradientCheckpointingConfig {
                interval_nodes: 0,
                min_tensor_bytes: 0,
            },
        )
        .expect_err("zero interval must fail");

        assert!(err.message.contains("interval_nodes"));
    }
}
