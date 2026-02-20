use std::collections::HashSet;

use crate::ir::{
    AllocationPlan, Graph, KernelGroup, Schedule, ValueId, build_schedule, group_kernels,
    plan_allocation, verify_allocation, verify_graph, verify_schedule,
};

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub schedule: Schedule,
    pub allocation: AllocationPlan,
    pub kernel_groups: Vec<KernelGroup>,
}

#[derive(Debug, Clone)]
pub struct ExecutionPlanError {
    pub message: String,
}

pub fn build_execution_plan(
    graph: &Graph,
    gradient_values: &HashSet<ValueId>,
) -> Result<ExecutionPlan, ExecutionPlanError> {
    verify_graph(graph).map_err(|err| ExecutionPlanError {
        message: err.message,
    })?;

    let schedule = build_schedule(graph).map_err(|err| ExecutionPlanError {
        message: err.message,
    })?;
    verify_schedule(graph, &schedule).map_err(|err| ExecutionPlanError {
        message: err.message,
    })?;

    let allocation =
        plan_allocation(graph, &schedule, gradient_values).map_err(|err| ExecutionPlanError {
            message: err.message,
        })?;
    verify_allocation(graph, &schedule, &allocation).map_err(|err| ExecutionPlanError {
        message: err.message,
    })?;

    let kernel_groups = group_kernels(graph, &schedule).map_err(|err| ExecutionPlanError {
        message: err.message,
    })?;

    Ok(ExecutionPlan {
        schedule,
        allocation,
        kernel_groups,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::ir::{Graph, Op, build_execution_plan};

    #[test]
    fn builds_schedule_and_allocation_together() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, x) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        let (_, w) = graph
            .add_op(block, Op::Parameter("w".to_string()))
            .expect("add op should succeed");
        let (_, y) = graph
            .add_op(block, Op::MatMul(x, w))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Output(y))
            .expect("add op should succeed");

        let plan = build_execution_plan(&graph, &HashSet::new()).expect("plan should build");
        assert_eq!(plan.schedule.ordered_nodes.len(), 4);
        assert!(!plan.allocation.buffer_map.is_empty());
        assert!(!plan.kernel_groups.is_empty());
    }
}
