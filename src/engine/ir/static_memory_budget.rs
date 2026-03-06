use crate::ir::{Graph, plan_memory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticMemoryBudget {
    pub max_peak_live_bytes: usize,
    pub max_peak_live_values: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticMemoryBudgetReport {
    pub peak_live_bytes: usize,
    pub peak_live_values: usize,
    pub max_peak_live_bytes: usize,
    pub max_peak_live_values: usize,
    pub within_budget: bool,
}

#[derive(Debug, Clone)]
pub struct StaticMemoryBudgetError {
    pub message: String,
}

pub fn evaluate_static_memory_budget(
    graph: &Graph,
    budget: &StaticMemoryBudget,
) -> Result<StaticMemoryBudgetReport, StaticMemoryBudgetError> {
    let plan = plan_memory(graph).map_err(|err| StaticMemoryBudgetError {
        message: format!("memory planning failed: {}", err.message),
    })?;

    if plan.peak_live_values > budget.max_peak_live_values {
        return Err(StaticMemoryBudgetError {
            message: format!(
                "peak_live_values exceeds budget: current={} max={}",
                plan.peak_live_values, budget.max_peak_live_values
            ),
        });
    }
    if plan.peak_live_bytes > budget.max_peak_live_bytes {
        return Err(StaticMemoryBudgetError {
            message: format!(
                "peak_live_bytes exceeds budget: current={} max={}",
                plan.peak_live_bytes, budget.max_peak_live_bytes
            ),
        });
    }

    Ok(StaticMemoryBudgetReport {
        peak_live_bytes: plan.peak_live_bytes,
        peak_live_values: plan.peak_live_values,
        max_peak_live_bytes: budget.max_peak_live_bytes,
        max_peak_live_values: budget.max_peak_live_values,
        within_budget: true,
    })
}

#[cfg(test)]
mod tests {
    use crate::ir::{Graph, Op, StaticMemoryBudget, evaluate_static_memory_budget};

    #[test]
    fn rejects_plan_when_peak_bytes_exceed_budget() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add input should succeed");
        graph
            .add_op(block, Op::Output(input))
            .expect("add output should succeed");
        graph.bind_input_shape("x", vec![1, 8]);

        let err = evaluate_static_memory_budget(
            &graph,
            &StaticMemoryBudget {
                max_peak_live_bytes: 1,
                max_peak_live_values: 32,
            },
        )
        .expect_err("budget must reject low byte limit");

        assert!(err.message.contains("peak_live_bytes"));
    }
}
