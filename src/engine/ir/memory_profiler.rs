/// Memory profiler for tracking tensor allocation and peak usage in Volta IR.
use std::collections::HashMap;
use crate::ir::{Graph, NodeId, op::Op, node::ValueId};
use crate::ir::interpreter::{ExecutionContext, RuntimeValue};
use crate::ir::scheduler::build_schedule;

/// Per-node memory usage record.
#[derive(Debug, Clone)]
pub struct NodeMemoryProfile {
    pub node_id: NodeId,
    pub op_name: String,
    pub output_bytes: usize,
    pub live_bytes_after: usize,
}

/// Result of a memory profiling run.
#[derive(Debug, Clone)]
pub struct MemoryProfileReport {
    pub nodes: Vec<NodeMemoryProfile>,
    /// Peak live memory across all steps (bytes).
    pub peak_bytes: usize,
    /// Total tensor memory allocated (counting all outputs, not just peak).
    pub total_allocated: usize,
    /// Estimated parameter memory.
    pub parameter_bytes: usize,
}

impl MemoryProfileReport {
    /// Print formatted memory report.
    pub fn print(&self) {
        println!("\n=== Volta Memory Profiler Report ===");
        println!("Peak live memory:    {:.2} MB ({} bytes)", self.peak_bytes as f64 / 1024.0 / 1024.0, self.peak_bytes);
        println!("Total allocated:     {:.2} MB ({} bytes)", self.total_allocated as f64 / 1024.0 / 1024.0, self.total_allocated);
        println!("Parameter memory:    {:.2} MB ({} bytes)", self.parameter_bytes as f64 / 1024.0 / 1024.0, self.parameter_bytes);
        println!("\nTop memory ops:");
        let mut sorted = self.nodes.clone();
        sorted.sort_by(|a, b| b.output_bytes.cmp(&a.output_bytes));
        println!("{:<8} {:<30} {:>12} {:>12}", "NodeId", "Op", "Output (KB)", "Live (KB)");
        println!("{}", "-".repeat(68));
        for n in sorted.iter().take(20) {
            println!("{:<8} {:<30} {:>12.1} {:>12.1}",
                n.node_id.0,
                &n.op_name[..n.op_name.len().min(29)],
                n.output_bytes as f64 / 1024.0,
                n.live_bytes_after as f64 / 1024.0,
            );
        }
    }
}

#[derive(Debug)]
pub struct MemoryProfilerError {
    pub message: String,
}

/// Profile memory usage of a graph execution.
/// Simulates execution tracking which tensors are live at each step.
pub fn profile_memory(
    graph: &Graph,
    context: &ExecutionContext,
) -> Result<MemoryProfileReport, MemoryProfilerError> {
    let schedule = build_schedule(graph).map_err(|e| MemoryProfilerError {
        message: format!("Schedule error: {:?}", e),
    })?;

    let mut nodes_profile = Vec::new();
    let mut live_tensors: HashMap<ValueId, usize> = HashMap::new();
    let mut peak_bytes = 0usize;
    let mut total_allocated = 0usize;

    // Estimate parameter memory from context
    let parameter_bytes: usize = context.parameters.values()
        .map(|v| estimate_value_bytes(v))
        .sum();

    // Build a map of which nodes use each value (for liveness tracking)
    let mut last_use: HashMap<ValueId, usize> = HashMap::new();
    for (step, &node_id) in schedule.ordered_nodes.iter().enumerate() {
        if let Some(node) = graph.node(node_id) {
            for input_val in node.op.input_values() {
                last_use.insert(input_val, step);
            }
        }
    }

    for (step, &node_id) in schedule.ordered_nodes.iter().enumerate() {
        let node = graph.node(node_id).ok_or_else(|| MemoryProfilerError {
            message: format!("Invalid node {:?}", node_id),
        })?;

        // Estimate output size
        let output_bytes = estimate_op_output_bytes(&node.op, graph, &live_tensors);
        live_tensors.insert(node.output, output_bytes);
        total_allocated += output_bytes;

        // Free tensors whose last use was before this step
        live_tensors.retain(|val, _| last_use.get(val).copied().unwrap_or(usize::MAX) >= step);

        let live_bytes: usize = live_tensors.values().sum();
        if live_bytes > peak_bytes { peak_bytes = live_bytes; }

        nodes_profile.push(NodeMemoryProfile {
            node_id,
            op_name: crate::ir::printer::op_name(&node.op).to_string(),
            output_bytes,
            live_bytes_after: live_bytes,
        });
    }

    Ok(MemoryProfileReport {
        nodes: nodes_profile,
        peak_bytes,
        total_allocated,
        parameter_bytes,
    })
}

fn estimate_value_bytes(value: &RuntimeValue) -> usize {
    match value {
        RuntimeValue::Tensor(t) => t.data.len() * 4,
        RuntimeValue::Int(_) => 8,
        RuntimeValue::Float(_) => 8,
    }
}

fn estimate_op_output_bytes(
    op: &Op,
    _graph: &Graph,
    live: &HashMap<ValueId, usize>,
) -> usize {
    // Rough estimate based on op type and input sizes
    let first_input_bytes = op.input_values().first()
        .and_then(|v| live.get(v))
        .copied()
        .unwrap_or(0);

    match op {
        Op::ConstTensor { shape, .. } => shape.iter().product::<usize>() * 4,
        Op::ConstInt(_) | Op::ConstFloat(_) => 4,
        Op::Parameter(_) | Op::Input(_) => first_input_bytes,
        // These preserve input size
        Op::Relu(..) | Op::Sigmoid(..) | Op::Gelu(..) | Op::GeluExact(..)
        | Op::Softmax(..) | Op::Neg(..) | Op::Exp(..) | Op::Log(..)
        | Op::Dropout { .. } | Op::Identity(..)
        | Op::LayerNorm { .. } | Op::GroupNorm { .. } | Op::InstanceNorm { .. }
        | Op::SinusoidalPE { .. } | Op::RoPE { .. } | Op::RoPEBackward { .. } => first_input_bytes,
        Op::Add(..) | Op::Sub(..) | Op::Mul(..) | Op::Div(..) => first_input_bytes,
        // These change sizes
        Op::MatMul(..) | Op::Gemm { .. } => first_input_bytes, // rough
        Op::Flatten { .. } => first_input_bytes,
        Op::GlobalAveragePool { .. } => {
            // Reduces spatial dims: rough estimate is input/spatial_reduction
            (first_input_bytes / 16).max(4)
        }
        Op::Embedding { weight, .. } => live.get(weight).copied().unwrap_or(0),
        _ => first_input_bytes,
    }
}
