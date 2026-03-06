use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ir::{
    Graph, MemoryPlanError, Op, Schedule, ShapeFact, ValueId, infer_shapes, plan_memory,
    verify_schedule,
};

const F32_BYTES: usize = std::mem::size_of::<f32>();
const MIN_BUFFER_ALIGNMENT_BYTES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    Input,
    Parameter,
    Temporary,
    Output,
    Gradient,
}

#[derive(Debug, Clone)]
pub struct AllocationPlan {
    pub buffer_map: HashMap<ValueId, BufferId>,
    pub storage_class: HashMap<ValueId, StorageClass>,
    pub reuse_edges: Vec<(ValueId, ValueId, BufferId)>,
    pub conflict_edges: Vec<(ValueId, ValueId)>,
    pub peak_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct AllocationError {
    pub message: String,
}

/// Plans the memory allocation for all IR values in the given `graph`.
///
/// # Errors
///
/// Returns `Err(AllocationError)` if the schedule is invalid, shape
/// inference fails, or the graph contains cycles that prevent liveness analysis.
#[allow(clippy::implicit_hasher)]
#[must_use = "the allocation plan must be used or errored"]
pub fn plan_allocation(
    graph: &Graph,
    schedule: &Schedule,
    gradient_values: &HashSet<ValueId>,
) -> Result<AllocationPlan, AllocationError> {
    verify_schedule(graph, schedule).map_err(|err| AllocationError {
        message: err.message,
    })?;

    let shapes = infer_shapes(graph).map_err(|err| AllocationError {
        message: err.message,
    })?;
    let liveness =
        plan_memory(graph).map_err(|MemoryPlanError { message }| AllocationError { message })?;

    let mut storage_class = HashMap::new();
    for node in &graph.nodes {
        if matches!(node.op, Op::Removed) {
            continue;
        }
        let class = classify_storage(&node.op, node.output, gradient_values);
        storage_class.insert(node.output, class);
    }

    let mut intervals = liveness.values.clone();
    intervals.sort_by_key(|entry| (entry.start_node, entry.end_node));

    let mut buffer_map = HashMap::<ValueId, BufferId>::new();
    let mut last_value_for_buffer = HashMap::<BufferId, ValueId>::new();
    let mut reuse_edges = Vec::new();
    let mut next_buffer_id = 0usize;
    let mut reusable = Vec::<(BufferId, usize, usize)>::new();

    for value in &intervals {
        let class = storage_class
            .get(&value.value)
            .copied()
            .unwrap_or(StorageClass::Temporary);
        let bytes = estimate_bytes(shapes.get(&value.value));

        let buffer = match class {
            StorageClass::Temporary | StorageClass::Gradient => {
                if let Some((index, (buffer, _, _))) =
                    reusable
                        .iter()
                        .enumerate()
                        .find(|(_, (_, free_after, size))| {
                            *free_after < value.start_node && *size >= bytes
                        })
                {
                    let chosen = *buffer;
                    reusable.remove(index);
                    chosen
                } else {
                    let id = BufferId(next_buffer_id);
                    next_buffer_id += 1;
                    id
                }
            }
            StorageClass::Input | StorageClass::Parameter | StorageClass::Output => {
                let id = BufferId(next_buffer_id);
                next_buffer_id += 1;
                id
            }
        };

        buffer_map.insert(value.value, buffer);

        if let Some(previous) = last_value_for_buffer.insert(buffer, value.value)
            && previous != value.value
        {
            reuse_edges.push((previous, value.value, buffer));
        }

        if matches!(class, StorageClass::Temporary | StorageClass::Gradient) {
            reusable.push((buffer, value.end_node, bytes));
        }
    }

    let peak_bytes = compute_peak_bytes(&liveness.values, &buffer_map, &shapes);

    let plan = AllocationPlan {
        buffer_map,
        storage_class,
        reuse_edges,
        conflict_edges: Vec::new(),
        peak_bytes,
    };

    verify_allocation(graph, schedule, &plan).map_err(|err| AllocationError {
        message: err.message,
    })?;

    Ok(plan)
}

/// Verifies an existing allocation plan against the current graph schedule.
///
/// # Errors
///
/// Returns `Err(AllocationError)` if the schedule is invalid or the plan
/// is inconsistent with the graph's current memory requirements.
pub fn verify_allocation(
    graph: &Graph,
    schedule: &Schedule,
    plan: &AllocationPlan,
) -> Result<(), AllocationError> {
    verify_schedule(graph, schedule).map_err(|err| AllocationError {
        message: err.message,
    })?;
    let memory =
        plan_memory(graph).map_err(|MemoryPlanError { message }| AllocationError { message })?;

    let mut intervals_by_buffer: BTreeMap<usize, Vec<(usize, usize, StorageClass, ValueId)>> =
        BTreeMap::new();
    let mut conflict_edges = Vec::new();
    for interval in &memory.values {
        let Some(buffer) = plan.buffer_map.get(&interval.value).copied() else {
            return Err(AllocationError {
                message: format!("Allocation missing buffer for ValueId {}", interval.value.0),
            });
        };
        let class = plan
            .storage_class
            .get(&interval.value)
            .copied()
            .unwrap_or(StorageClass::Temporary);
        intervals_by_buffer.entry(buffer.0).or_default().push((
            interval.start_node,
            interval.end_node,
            class,
            interval.value,
        ));
    }

    for (_buffer, intervals) in intervals_by_buffer {
        for i in 0..intervals.len() {
            for j in (i + 1)..intervals.len() {
                let (start_a, end_a, class_a, value_a) = intervals[i];
                let (start_b, end_b, class_b, value_b) = intervals[j];

                let overlap = start_a <= end_b && start_b <= end_a;
                if !overlap {
                    continue;
                }

                if matches!(class_a, StorageClass::Input | StorageClass::Parameter)
                    || matches!(class_b, StorageClass::Input | StorageClass::Parameter)
                {
                    conflict_edges.push((value_a, value_b));
                    return Err(AllocationError {
                        message: format!(
                            "Illegal aliasing: {value_a:?} and {value_b:?} overlap on immutable storage"
                        ),
                    });
                }
            }
        }
    }

    let _ = conflict_edges;

    Ok(())
}

fn classify_storage(op: &Op, value: ValueId, gradients: &HashSet<ValueId>) -> StorageClass {
    if gradients.contains(&value) {
        return StorageClass::Gradient;
    }

    match op {
        Op::Input(_) => StorageClass::Input,
        Op::Parameter(_) => StorageClass::Parameter,
        Op::Output(_) => StorageClass::Output,
        _ => StorageClass::Temporary,
    }
}

fn estimate_bytes(shape: Option<&ShapeFact>) -> usize {
    match shape {
        Some(ShapeFact::Tensor(dims)) => {
            if dims.contains(&0) {
                return 0;
            }

            let Some(raw_bytes) = dims
                .iter()
                .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
                .and_then(|elements| elements.checked_mul(F32_BYTES))
            else {
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

fn compute_peak_bytes(
    intervals: &[crate::ir::ValueLiveness],
    buffer_map: &HashMap<ValueId, BufferId>,
    shapes: &HashMap<ValueId, ShapeFact>,
) -> usize {
    let mut peak = 0usize;
    let max_point = intervals
        .iter()
        .map(|entry| entry.end_node)
        .max()
        .unwrap_or(0);

    for point in 0..=max_point {
        let mut live_buffers = BTreeMap::<usize, usize>::new();
        for interval in intervals {
            if interval.start_node <= point
                && point <= interval.end_node
                && let Some(buffer) = buffer_map.get(&interval.value)
            {
                let bytes = estimate_bytes(shapes.get(&interval.value));
                live_buffers.entry(buffer.0).or_insert(bytes);
            }
        }
        let total = live_buffers.values().copied().sum::<usize>();
        peak = peak.max(total);
    }

    peak
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::ir::{Graph, Op, StorageClass, build_schedule, plan_allocation, verify_allocation};

    #[test]
    fn plans_allocation_with_reuse_for_temporaries() {
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

        let schedule = build_schedule(&graph).expect("schedule should pass");
        let plan = plan_allocation(&graph, &schedule, &HashSet::new()).expect("plan should pass");
        verify_allocation(&graph, &schedule, &plan).expect("allocation should verify");

        assert!(plan.peak_bytes >= 8);
        assert_eq!(plan.storage_class.get(&c), Some(&StorageClass::Temporary));
    }
}
