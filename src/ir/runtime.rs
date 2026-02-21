use crate::ir::{
    Backend, CompilerFlags, DeterminismLevel, ExecutionContext, ExecutionPlan, Graph, NodeId,
    RuntimeValue, ValueId, compile_or_get_cached,
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
    let determinism = compile_plan_for_backend(plan, backend)?;

    backend
        .execute_terminal(graph, plan, ordered_nodes, context, determinism)
        .map_err(|err| RuntimeGatewayError {
            message: format!("Runtime execute failed: {}", err.message),
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
    let determinism = compile_plan_for_backend(plan, backend)?;

    backend
        .execute_value(graph, plan, target, ordered_nodes, context, determinism)
        .map_err(|err| RuntimeGatewayError {
            message: format!("Runtime execute-value failed: {}", err.message),
        })
}

fn compile_plan_for_backend(
    plan: &ExecutionPlan,
    backend: &dyn Backend,
) -> Result<DeterminismLevel, RuntimeGatewayError> {
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

    compile_or_get_cached(plan, backend, flags.determinism)
        .map(|_| flags.determinism)
        .map_err(|err| RuntimeGatewayError {
            message: format!("Backend compile failed: {}", err.message),
        })
}
