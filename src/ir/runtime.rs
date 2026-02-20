use crate::ir::{
    Backend, CompilerFlags, DeterminismLevel, ExecutionContext, ExecutionPlan, Graph, NodeId,
    RuntimeValue, ValueId, execute_value_with_schedule_context, execute_with_schedule_context,
};

#[derive(Debug, Clone)]
pub struct RuntimeGatewayError {
    pub message: String,
}

pub fn execute_terminal_with_backend(
    graph: &Graph,
    plan: &ExecutionPlan,
    ordered_nodes: &[NodeId],
    backend: &dyn Backend,
    context: &ExecutionContext,
) -> Result<Option<RuntimeValue>, RuntimeGatewayError> {
    compile_plan_for_backend(plan, backend)?;

    execute_with_schedule_context(graph, ordered_nodes, context).map_err(|err| {
        RuntimeGatewayError {
            message: format!("Runtime execute failed: {}", err.message),
        }
    })
}

pub fn execute_value_with_backend(
    graph: &Graph,
    plan: &ExecutionPlan,
    target: ValueId,
    ordered_nodes: &[NodeId],
    backend: &dyn Backend,
    context: &ExecutionContext,
) -> Result<RuntimeValue, RuntimeGatewayError> {
    compile_plan_for_backend(plan, backend)?;

    execute_value_with_schedule_context(graph, target, ordered_nodes, context).map_err(|err| {
        RuntimeGatewayError {
            message: format!("Runtime execute-value failed: {}", err.message),
        }
    })
}

fn compile_plan_for_backend(
    plan: &ExecutionPlan,
    backend: &dyn Backend,
) -> Result<(), RuntimeGatewayError> {
    let flags = CompilerFlags::from_env();
    let caps = backend.capabilities();
    if flags.determinism == DeterminismLevel::Strict && !caps.supports_strict_determinism {
        return Err(RuntimeGatewayError {
            message: format!(
                "Backend {:?} does not support strict determinism",
                caps.backend
            ),
        });
    }

    backend
        .compile(plan)
        .map(|_| ())
        .map_err(|err| RuntimeGatewayError {
            message: format!("Backend compile failed: {}", err.message),
        })
}
