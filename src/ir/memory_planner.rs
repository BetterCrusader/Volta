use std::collections::HashMap;

use crate::ir::{Graph, ShapeFact, ValueId, infer_shapes};

const F32_BYTES: usize = std::mem::size_of::<f32>();
const MIN_BUFFER_ALIGNMENT_BYTES: usize = 16;

#[derive(Debug, Clone)]
pub struct MemoryPlanError {
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ValueLiveness {
    pub value: ValueId,
    pub start_node: usize,
    pub end_node: usize,
    pub estimated_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryPlan {
    pub values: Vec<ValueLiveness>,
    pub peak_live_values: usize,
    pub peak_live_bytes: usize,
}

pub fn render_lifetime_heatmap(plan: &MemoryPlan, node_count: usize) -> String {
    let mut lines = Vec::new();
    for value in &plan.values {
        let mut row = String::new();
        for point in 0..node_count {
            if value.start_node <= point && point <= value.end_node {
                row.push('#');
            } else {
                row.push('.');
            }
        }
        lines.push(format!("v{} {}", value.value.0, row));
    }
    lines.join("\n")
}

pub fn plan_memory(graph: &Graph) -> Result<MemoryPlan, MemoryPlanError> {
    let shapes = infer_shapes(graph).map_err(|err| MemoryPlanError {
        message: err.message,
    })?;

    let mut intervals: HashMap<ValueId, (usize, usize)> = HashMap::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, crate::ir::Op::Removed) {
            continue;
        }
        intervals.insert(node.output, (index, index));
    }

    for (index, node) in graph.nodes.iter().enumerate() {
        for input in node.op.input_values() {
            if let Some((_, end)) = intervals.get_mut(&input)
                && *end < index
            {
                *end = index;
            }
        }
    }

    if !graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, crate::ir::Op::Output(_)))
        && let Some(last) = graph.last_value_id()
        && let Some((_, end)) = intervals.get_mut(&last)
    {
        *end = graph.nodes.len().saturating_sub(1);
    }

    let mut values = intervals
        .into_iter()
        .map(|(value, (start_node, end_node))| ValueLiveness {
            value,
            start_node,
            end_node,
            estimated_bytes: estimate_bytes(shapes.get(&value)),
        })
        .collect::<Vec<_>>();
    values.sort_by_key(|entry| entry.value.0);

    let (peak_live_values, peak_live_bytes) = compute_peak_usage(&values, graph.nodes.len());
    Ok(MemoryPlan {
        values,
        peak_live_values,
        peak_live_bytes,
    })
}

fn estimate_bytes(shape: Option<&ShapeFact>) -> usize {
    match shape {
        Some(ShapeFact::Tensor(dims)) => {
            if dims.contains(&0) {
                return 0;
            }

            let mut count = 1usize;
            for dim in dims {
                let Some(next) = count.checked_mul(*dim) else {
                    return 0;
                };
                count = next;
            }

            let Some(raw_bytes) = count.checked_mul(F32_BYTES) else {
                return 0;
            };
            align_bytes(raw_bytes)
        }
        _ => 0,
    }
}

fn align_bytes(raw_bytes: usize) -> usize {
    if raw_bytes == 0 {
        return 0;
    }

    raw_bytes
        .div_ceil(MIN_BUFFER_ALIGNMENT_BYTES)
        .saturating_mul(MIN_BUFFER_ALIGNMENT_BYTES)
}

fn compute_peak_usage(values: &[ValueLiveness], node_count: usize) -> (usize, usize) {
    let mut peak_values = 0usize;
    let mut peak_bytes = 0usize;

    for point in 0..node_count {
        let mut live_values = 0usize;
        let mut live_bytes = 0usize;
        for value in values {
            if value.start_node <= point && point <= value.end_node {
                live_values += 1;
                live_bytes = live_bytes.saturating_add(value.estimated_bytes);
            }
        }
        peak_values = peak_values.max(live_values);
        peak_bytes = peak_bytes.max(live_bytes);
    }

    (peak_values, peak_bytes)
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, plan_memory, render_lifetime_heatmap};

    #[test]
    fn plans_liveness_and_peak_usage() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, a) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2],
                    data: vec![1.0, 2.0],
                },
            )
            .expect("add op should succeed");
        let (_, b) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2],
                    data: vec![3.0, 4.0],
                },
            )
            .expect("add op should succeed");
        let (_, c) = graph
            .add_op(block, Op::Add(a, b))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Output(c))
            .expect("add op should succeed");

        let plan = plan_memory(&graph).expect("planning should pass");
        assert!(!plan.values.is_empty());
        assert!(plan.peak_live_values >= 2);
        assert!(plan.peak_live_bytes >= 8);

        let heatmap = render_lifetime_heatmap(&plan, graph.nodes.len());
        assert!(heatmap.contains("v"));
    }
}
