use crate::ir::InterpreterError;
use crate::ir::interpreter::{ExecutionContext, RuntimeValue};
use crate::ir::scheduler::build_schedule;
use crate::ir::{Graph, NodeId};
/// Node-level execution profiler for Volta IR graphs.
///
/// Records timing for each op in a graph execution pass,
/// then reports per-op statistics sorted by total time.
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Per-node timing record.
#[derive(Debug, Clone)]
pub struct NodeProfile {
    pub node_id: NodeId,
    pub op_name: String,
    pub calls: usize,
    pub total_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl NodeProfile {
    pub fn avg_time(&self) -> Duration {
        if self.calls == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.calls as u32
        }
    }
}

/// Profiler: runs a graph repeatedly and collects per-node timing.
pub struct Profiler {
    records: HashMap<NodeId, NodeProfile>,
    pub warmup_runs: usize,
    pub measure_runs: usize,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            warmup_runs: 3,
            measure_runs: 10,
        }
    }

    pub fn with_runs(mut self, warmup: usize, measure: usize) -> Self {
        self.warmup_runs = warmup;
        self.measure_runs = measure;
        self
    }

    /// Profile a graph execution.
    pub fn profile(
        &mut self,
        graph: &Graph,
        context: &ExecutionContext,
    ) -> Result<(), ProfilerError> {
        let schedule = build_schedule(graph).map_err(|e| ProfilerError {
            message: format!("Schedule error: {:?}", e),
        })?;

        // Warmup runs
        for _ in 0..self.warmup_runs {
            run_graph_once(graph, context, &schedule.ordered_nodes)
                .map_err(|e| ProfilerError { message: e.message })?;
        }

        // Measurement runs
        for _ in 0..self.measure_runs {
            self.run_and_record(graph, context, &schedule.ordered_nodes)?;
        }

        Ok(())
    }

    fn run_and_record(
        &mut self,
        graph: &Graph,
        context: &ExecutionContext,
        ordered_nodes: &[NodeId],
    ) -> Result<(), ProfilerError> {
        let mut values: Vec<Option<RuntimeValue>> = vec![None; graph.value_count()];

        for &node_id in ordered_nodes {
            let node = graph.node(node_id).ok_or_else(|| ProfilerError {
                message: format!("Invalid NodeId: {:?}", node_id),
            })?;

            let t0 = Instant::now();
            let result =
                crate::ir::interpreter::evaluate_op_public(&node.op, &values, node_id, context)
                    .map_err(|e| ProfilerError { message: e.message })?;
            let elapsed = t0.elapsed();

            let output_idx = node.output.0;
            if output_idx < values.len() {
                values[output_idx] = Some(result);
            }

            let op_name = crate::ir::printer::op_name(&node.op);
            let entry = self.records.entry(node_id).or_insert_with(|| NodeProfile {
                node_id,
                op_name: op_name.to_string(),
                calls: 0,
                total_time: Duration::ZERO,
                min_time: Duration::MAX,
                max_time: Duration::ZERO,
            });
            entry.calls += 1;
            entry.total_time += elapsed;
            if elapsed < entry.min_time {
                entry.min_time = elapsed;
            }
            if elapsed > entry.max_time {
                entry.max_time = elapsed;
            }
        }

        Ok(())
    }

    /// Returns profiles sorted by total time descending.
    pub fn sorted_profiles(&self) -> Vec<&NodeProfile> {
        let mut v: Vec<&NodeProfile> = self.records.values().collect();
        v.sort_by(|a, b| b.total_time.cmp(&a.total_time));
        v
    }

    /// Print a formatted report.
    pub fn print_report(&self) {
        let profiles = self.sorted_profiles();
        if profiles.is_empty() {
            println!("No profiling data collected.");
            return;
        }
        let total: Duration = profiles.iter().map(|p| p.total_time).sum();
        println!("\n=== Volta Profiler Report ===");
        println!(
            "{:<8} {:<30} {:>8} {:>12} {:>12} {:>12} {:>7}",
            "NodeId", "Op", "Calls", "Total (µs)", "Avg (µs)", "Max (µs)", "%"
        );
        println!("{}", "-".repeat(95));
        for p in &profiles {
            let pct = if total.is_zero() {
                0.0
            } else {
                p.total_time.as_secs_f64() / total.as_secs_f64() * 100.0
            };
            println!(
                "{:<8} {:<30} {:>8} {:>12.1} {:>12.1} {:>12.1} {:>6.1}%",
                p.node_id.0,
                &p.op_name[..p.op_name.len().min(29)],
                p.calls,
                p.total_time.as_micros() as f64,
                p.avg_time().as_micros() as f64,
                p.max_time.as_micros() as f64,
                pct,
            );
        }
        println!("{}", "-".repeat(95));
        println!(
            "Total: {:.1} µs across {} nodes",
            total.as_micros() as f64,
            profiles.len()
        );
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct ProfilerError {
    pub message: String,
}

fn run_graph_once(
    graph: &Graph,
    context: &ExecutionContext,
    ordered_nodes: &[NodeId],
) -> Result<(), InterpreterError> {
    let mut values: Vec<Option<RuntimeValue>> = vec![None; graph.value_count()];
    for &node_id in ordered_nodes {
        let node = graph.node(node_id).ok_or_else(|| InterpreterError {
            message: format!("Invalid NodeId: {:?}", node_id),
            node: None,
        })?;
        let result =
            crate::ir::interpreter::evaluate_op_public(&node.op, &values, node_id, context)?;
        let output_idx = node.output.0;
        if output_idx < values.len() {
            values[output_idx] = Some(result);
        }
    }
    Ok(())
}
