use crate::ir::{Graph, Schedule, verify_schedule};

#[derive(Debug, Clone)]
pub struct ScheduleOptimizationResult {
    pub schedule: Schedule,
    pub fused_transitions: usize,
}

#[derive(Debug, Clone)]
pub struct ScheduleOptimizationError {
    pub message: String,
}

pub fn optimize_schedule(
    graph: &Graph,
    schedule: &Schedule,
) -> Result<ScheduleOptimizationResult, ScheduleOptimizationError> {
    verify_schedule(graph, schedule).map_err(|err| ScheduleOptimizationError {
        message: format!("input schedule verification failed: {}", err.message),
    })?;

    Ok(ScheduleOptimizationResult {
        schedule: schedule.clone(),
        fused_transitions: 0,
    })
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, build_schedule, optimize_schedule, verify_schedule};

    #[test]
    fn optimized_schedule_remains_valid() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add input should succeed");
        graph
            .add_op(block, Op::Output(x))
            .expect("add output should succeed");
        graph.bind_input_shape("x", vec![1, 1]);

        let base = build_schedule(&graph).expect("schedule should build");
        let optimized = optimize_schedule(&graph, &base).expect("optimization should pass");
        verify_schedule(&graph, &optimized.schedule).expect("optimized schedule must verify");
    }
}
