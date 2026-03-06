use std::collections::HashMap;
use crate::ir::{
    Backend, CompilerFlags, DeterminismLevel, ExecutionContext, ExecutionPlan, Graph, NodeId,
    RuntimeValue, ValueId, compile_or_get_cached, verify_allocation,
    verify_no_undecomposed_backward_ops, verify_schedule,
};
use crate::ir::interpreter::{
    execute_multiple_values_with_schedule_context,
    execute_multiple_values_with_buffer,
    execute_terminal_with_buffer,
    execute_terminal_and_save_all,
    execute_multiple_values_with_saved_activations,
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
    let determinism = compile_plan_for_backend(graph, plan, backend)?;

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
    let determinism = compile_plan_for_backend(graph, plan, backend)?;

    backend
        .execute_value(graph, plan, target, ordered_nodes, context, determinism)
        .map_err(|err| RuntimeGatewayError {
            message: format!("Runtime execute-value failed: {}", err.message),
        })
}

pub fn execute_multiple_values_with_backend(
    graph: &Graph,
    plan: &ExecutionPlan,
    targets: &[ValueId],
    ordered_nodes: &[NodeId],
    backend: &dyn Backend,
    context: &ExecutionContext,
) -> Result<HashMap<ValueId, RuntimeValue>, RuntimeGatewayError> {
    let _ = compile_plan_for_backend(graph, plan, backend)?;
    execute_multiple_values_with_schedule_context(graph, ordered_nodes, targets, context)
        .map_err(|err| RuntimeGatewayError {
            message: format!("Runtime execute-multiple failed: {}", err.message),
        })
}

/// Execute the terminal value reusing a pre-allocated buffer (avoids per-call heap allocation).
/// The buffer is automatically grown and reset; pass `&mut Vec::new()` on first call.
pub fn execute_terminal_with_backend_buffered(
    graph: &Graph,
    plan: &ExecutionPlan,
    ordered_nodes: &[NodeId],
    backend: &dyn Backend,
    context: &ExecutionContext,
    buf: &mut Vec<Option<RuntimeValue>>,
) -> Result<Option<RuntimeValue>, RuntimeGatewayError> {
    let _ = compile_plan_for_backend(graph, plan, backend)?;
    execute_terminal_with_buffer(graph, ordered_nodes, context, buf)
        .map_err(|err| RuntimeGatewayError {
            message: format!("Runtime execute-terminal-buffered failed: {}", err.message),
        })
}

/// Execute multiple values reusing a pre-allocated buffer.
pub fn execute_multiple_values_with_backend_buffered(
    graph: &Graph,
    plan: &ExecutionPlan,
    targets: &[ValueId],
    ordered_nodes: &[NodeId],
    backend: &dyn Backend,
    context: &ExecutionContext,
    buf: &mut Vec<Option<RuntimeValue>>,
) -> Result<HashMap<ValueId, RuntimeValue>, RuntimeGatewayError> {
    let _ = compile_plan_for_backend(graph, plan, backend)?;
    execute_multiple_values_with_buffer(graph, ordered_nodes, targets, context, buf)
        .map_err(|err| RuntimeGatewayError {
            message: format!("Runtime execute-multiple-buffered failed: {}", err.message),
        })
}

/// Forward pass that saves all intermediate activations into `fwd_buf`.
/// Pass the filled `fwd_buf` to `execute_backward_with_saved_activations` to
/// skip re-running the forward sub-graph in the backward pass.
pub fn execute_forward_and_save(
    graph: &Graph,
    plan: &ExecutionPlan,
    ordered_nodes: &[NodeId],
    backend: &dyn Backend,
    context: &ExecutionContext,
    fwd_buf: &mut Vec<Option<RuntimeValue>>,
) -> Result<Option<RuntimeValue>, RuntimeGatewayError> {
    let _ = compile_plan_for_backend(graph, plan, backend)?;
    execute_terminal_and_save_all(graph, ordered_nodes, context, fwd_buf)
        .map_err(|err| RuntimeGatewayError {
            message: format!("Runtime forward-and-save failed: {}", err.message),
        })
}

/// Backward pass reusing saved forward activations (avoids re-running the
/// forward sub-graph that was cloned into the backward graph).
pub fn execute_backward_with_saved_activations(
    graph: &Graph,
    plan: &ExecutionPlan,
    targets: &[ValueId],
    ordered_nodes: &[NodeId],
    backend: &dyn Backend,
    context: &ExecutionContext,
    fwd_saved: &[Option<RuntimeValue>],
    bwd_buf: &mut Vec<Option<RuntimeValue>>,
) -> Result<HashMap<ValueId, RuntimeValue>, RuntimeGatewayError> {
    let _ = compile_plan_for_backend(graph, plan, backend)?;
    execute_multiple_values_with_saved_activations(
        graph, ordered_nodes, targets, context, fwd_saved, bwd_buf,
    )
    .map_err(|err| RuntimeGatewayError {
        message: format!("Runtime backward-with-activations failed: {}", err.message),
    })
}

fn compile_plan_for_backend(
    graph: &Graph,
    plan: &ExecutionPlan,
    backend: &dyn Backend,
) -> Result<DeterminismLevel, RuntimeGatewayError> {
    // Skip the three graph-traversal verify passes when the plan was already
    // fully verified at build time (the common hot-path during training).
    if !plan.verified {
        verify_no_undecomposed_backward_ops(graph).map_err(|err| RuntimeGatewayError {
            message: format!("Runtime preflight failed: {}", err.message),
        })?;
        verify_schedule(graph, &plan.schedule).map_err(|err| RuntimeGatewayError {
            message: format!("Runtime schedule preflight failed: {}", err.message),
        })?;
        verify_allocation(graph, &plan.schedule, &plan.allocation).map_err(|err| {
            RuntimeGatewayError {
                message: format!("Runtime allocation preflight failed: {}", err.message),
            }
        })?;
    }

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
