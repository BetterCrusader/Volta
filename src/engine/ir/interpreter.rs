use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::ir::kernels::{activations, arithmetic, conv, math, norm, reduce};
use crate::ir::tensor::Tensor;
use crate::ir::{ExecutionPhase, Graph, NodeId, Op, ValueId, build_schedule};

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeValue {
    Int(i64),
    Float(f64),
    Tensor(Arc<Tensor>),
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionContext {
    pub inputs: HashMap<String, RuntimeValue>,
    pub parameters: HashMap<String, RuntimeValue>,
    pub phase: ExecutionPhase,
}

#[derive(Debug, Clone)]
pub struct InterpreterError {
    pub message: String,
    pub node: Option<NodeId>,
}

pub fn execute(graph: &Graph) -> Result<Option<RuntimeValue>, InterpreterError> {
    execute_with_context(graph, &ExecutionContext::default())
}

pub fn execute_with_context(
    graph: &Graph,
    context: &ExecutionContext,
) -> Result<Option<RuntimeValue>, InterpreterError> {
    let schedule = build_schedule(graph).map_err(|err| InterpreterError {
        message: format!("Failed to build schedule: {}", err.message),
        node: None,
    })?;

    if let Some(target) = terminal_value_id(graph) {
        execute_value_with_context_scheduled(graph, target, context, schedule.nodes()).map(Some)
    } else {
        Ok(None)
    }
}

pub fn execute_value(graph: &Graph, target: ValueId) -> Result<RuntimeValue, InterpreterError> {
    execute_value_with_context(graph, target, &ExecutionContext::default())
}

pub fn execute_value_with_context(
    graph: &Graph,
    target: ValueId,
    context: &ExecutionContext,
) -> Result<RuntimeValue, InterpreterError> {
    let schedule = build_schedule(graph).map_err(|err| InterpreterError {
        message: format!("Failed to build schedule: {}", err.message),
        node: None,
    })?;
    execute_value_with_context_scheduled(graph, target, context, schedule.nodes())
}

pub fn execute_with_schedule_context(
    graph: &Graph,
    ordered_nodes: &[NodeId],
    context: &ExecutionContext,
) -> Result<Option<RuntimeValue>, InterpreterError> {
    if let Some(target) = terminal_value_id(graph) {
        execute_value_with_context_scheduled(graph, target, context, ordered_nodes).map(Some)
    } else {
        Ok(None)
    }
}

pub fn execute_value_with_schedule_context(
    graph: &Graph,
    ordered_nodes: &[NodeId],
    target: ValueId,
    context: &ExecutionContext,
) -> Result<RuntimeValue, InterpreterError> {
    execute_value_with_context_scheduled(graph, target, context, ordered_nodes)
}

/// Execute the graph once and return all requested ValueIds in a single pass.
/// Much faster than calling execute_value_with_schedule_context N times.
pub fn execute_multiple_values_with_schedule_context(
    graph: &Graph,
    ordered_nodes: &[NodeId],
    targets: &[ValueId],
    context: &ExecutionContext,
) -> Result<HashMap<ValueId, RuntimeValue>, InterpreterError> {
    let mut values = vec![None::<RuntimeValue>; graph.value_count()];
    run_nodes_into_buffer(graph, ordered_nodes, context, &mut values)?;

    let mut result = HashMap::with_capacity(targets.len());
    for &target in targets {
        let value = read_value(&values, target, None)?;
        result.insert(target, value);
    }
    Ok(result)
}

/// Same as `execute_multiple_values_with_schedule_context` but reuses a
/// caller-provided buffer to avoid repeated heap allocation in hot loops.
/// The buffer is automatically grown and reset on each call.
pub fn execute_multiple_values_with_buffer(
    graph: &Graph,
    ordered_nodes: &[NodeId],
    targets: &[ValueId],
    context: &ExecutionContext,
    buf: &mut Vec<Option<RuntimeValue>>,
) -> Result<HashMap<ValueId, RuntimeValue>, InterpreterError> {
    let needed = graph.value_count();
    if buf.len() < needed {
        buf.resize(needed, None);
    } else {
        buf[..needed].fill(None);
    }
    run_nodes_into_buffer(graph, ordered_nodes, context, &mut buf[..needed])?;

    let mut result = HashMap::with_capacity(targets.len());
    for &target in targets {
        let value = read_value(&buf[..needed], target, None)?;
        result.insert(target, value);
    }
    Ok(result)
}

/// Execute and return the terminal value, reusing a caller-provided buffer.
pub fn execute_terminal_with_buffer(
    graph: &Graph,
    ordered_nodes: &[NodeId],
    context: &ExecutionContext,
    buf: &mut Vec<Option<RuntimeValue>>,
) -> Result<Option<RuntimeValue>, InterpreterError> {
    let needed = graph.value_count();
    if buf.len() < needed {
        buf.resize(needed, None);
    } else {
        buf[..needed].fill(None);
    }
    run_nodes_into_buffer(graph, ordered_nodes, context, &mut buf[..needed])?;

    if let Some(target) = terminal_value_id(graph) {
        read_value(&buf[..needed], target, None).map(Some)
    } else {
        Ok(None)
    }
}

/// Execute the forward pass, capture ALL intermediate values into `fwd_buf`,
/// and return the terminal value.  The filled `fwd_buf` can then be passed to
/// `execute_multiple_values_with_saved_activations` for the backward pass to
/// skip re-running the forward portion of the backward graph.
pub fn execute_terminal_and_save_all(
    graph: &Graph,
    ordered_nodes: &[NodeId],
    context: &ExecutionContext,
    fwd_buf: &mut Vec<Option<RuntimeValue>>,
) -> Result<Option<RuntimeValue>, InterpreterError> {
    let needed = graph.value_count();
    if fwd_buf.len() < needed {
        fwd_buf.resize(needed, None);
    } else {
        fwd_buf[..needed].fill(None);
    }
    run_nodes_into_buffer(graph, ordered_nodes, context, &mut fwd_buf[..needed])?;

    if let Some(target) = terminal_value_id(graph) {
        read_value(&fwd_buf[..needed], target, None).map(Some)
    } else {
        Ok(None)
    }
}

/// Execute the backward graph, reusing pre-computed forward activations.
/// `fwd_saved` contains the values from the forward pass at matching ValueId
/// positions (the backward graph clones forward nodes at the same ValueId
/// positions, so they can be skipped).
/// The result buffer `bwd_buf` is also reused across calls.
pub fn execute_multiple_values_with_saved_activations(
    graph: &Graph,
    ordered_nodes: &[NodeId],
    targets: &[ValueId],
    context: &ExecutionContext,
    fwd_saved: &[Option<RuntimeValue>],
    bwd_buf: &mut Vec<Option<RuntimeValue>>,
) -> Result<HashMap<ValueId, RuntimeValue>, InterpreterError> {
    let needed = graph.value_count();
    // Size the backward buffer and pre-fill with saved forward activations.
    if bwd_buf.len() < needed {
        bwd_buf.resize(needed, None);
    }
    // Copy saved forward activations into the backward buffer.
    // Only copy slots that exist in both buffers.
    let copy_len = fwd_saved.len().min(needed);
    bwd_buf[..copy_len].clone_from_slice(&fwd_saved[..copy_len]);
    // Zero out any extra slots beyond the forward activations.
    bwd_buf[copy_len..needed].fill(None);

    // Run the backward-specific nodes, skipping pre-filled forward slots.
    run_nodes_into_buffer_with_prefill(graph, ordered_nodes, context, &mut bwd_buf[..needed])?;

    let mut result = HashMap::with_capacity(targets.len());
    for &target in targets {
        let value = read_value(&bwd_buf[..needed], target, None)?;
        result.insert(target, value);
    }
    Ok(result)
}

fn run_nodes_into_buffer(
    graph: &Graph,
    ordered_nodes: &[NodeId],
    context: &ExecutionContext,
    values: &mut [Option<RuntimeValue>],
) -> Result<(), InterpreterError> {
    run_nodes_into_buffer_inner(graph, ordered_nodes, context, values, false)
}

/// Like `run_nodes_into_buffer` but skips nodes whose output slot is already
/// filled (pre-populated with saved activations from a prior forward pass).
fn run_nodes_into_buffer_with_prefill(
    graph: &Graph,
    ordered_nodes: &[NodeId],
    context: &ExecutionContext,
    values: &mut [Option<RuntimeValue>],
) -> Result<(), InterpreterError> {
    run_nodes_into_buffer_inner(graph, ordered_nodes, context, values, true)
}

fn run_nodes_into_buffer_inner(
    graph: &Graph,
    ordered_nodes: &[NodeId],
    context: &ExecutionContext,
    values: &mut [Option<RuntimeValue>],
    skip_prefilled: bool,
) -> Result<(), InterpreterError> {
    for node_id in ordered_nodes {
        let node = graph
            .node(*node_id)
            .ok_or_else(|| error(format!("Invalid NodeId in schedule: {node_id:?}"), None))?;
        let output_index = node.output.0;
        if output_index >= values.len() {
            return Err(error(
                format!("Output ValueId out of range: {output_index}"),
                Some(node.id),
            ));
        }
        if values[output_index].is_some() {
            if skip_prefilled {
                // Pre-filled with a saved activation — skip recomputation.
                continue;
            }
            return Err(error(
                format!("SSA violation: ValueId {output_index} assigned more than once"),
                Some(node.id),
            ));
        }
        if matches!(node.op, Op::Removed) {
            continue;
        }
        let computed = evaluate_op(&node.op, values, node.id, context)?;
        values[output_index] = Some(computed);
    }
    Ok(())
}

fn execute_value_with_context_scheduled(
    graph: &Graph,
    target: ValueId,
    context: &ExecutionContext,
    ordered_nodes: &[NodeId],
) -> Result<RuntimeValue, InterpreterError> {
    let mut values = vec![None; graph.value_count()];
    run_nodes_into_buffer(graph, ordered_nodes, context, &mut values)?;
    read_value(&values, target, None)
}

fn terminal_value_id(graph: &Graph) -> Option<ValueId> {
    for node in graph.nodes.iter().rev() {
        if let Op::Output(value) = node.op {
            return Some(value);
        }
    }
    graph.last_value_id()
}

/// Exposed for the profiler: evaluate a single op given pre-computed values buffer.
pub fn evaluate_op_public(
    op: &Op,
    values: &[Option<RuntimeValue>],
    node_id: NodeId,
    context: &ExecutionContext,
) -> Result<RuntimeValue, InterpreterError> {
    evaluate_op(op, values, node_id, context)
}

fn evaluate_op(
    op: &Op,
    values: &[Option<RuntimeValue>],
    node_id: NodeId,
    context: &ExecutionContext,
) -> Result<RuntimeValue, InterpreterError> {
    match op {
        Op::ConstInt(value) => Ok(RuntimeValue::Int(*value)),
        Op::ConstFloat(value) => Ok(RuntimeValue::Float(*value)),
        Op::ConstTensor { shape, data } => {
            let tensor = Tensor::new(shape.clone(), data.clone())
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(tensor))
        }
        Op::Parameter(name) => context
            .parameters
            .get(name)
            .cloned()
            .ok_or_else(|| error(format!("Missing parameter: '{name}'"), Some(node_id))),
        Op::Input(name) => context
            .inputs
            .get(name)
            .cloned()
            .ok_or_else(|| error(format!("Missing input: '{name}'"), Some(node_id))),
        Op::Output(value) => read_value(values, *value, Some(node_id)),
        Op::Add(left, right) => {
            let left_value = read_value(values, *left, Some(node_id))?;
            let right_value = read_value(values, *right, Some(node_id))?;
            arithmetic::add_values(left_value, right_value, node_id)
        }
        Op::Sub(left, right) => {
            let left_value = read_value(values, *left, Some(node_id))?;
            let right_value = read_value(values, *right, Some(node_id))?;
            arithmetic::sub_values(left_value, right_value, node_id)
        }
        Op::Mul(left, right) => {
            let left_value = read_value(values, *left, Some(node_id))?;
            let right_value = read_value(values, *right, Some(node_id))?;
            arithmetic::mul_values(left_value, right_value, node_id)
        }
        Op::Div(left, right) => {
            let left_value = read_value(values, *left, Some(node_id))?;
            let right_value = read_value(values, *right, Some(node_id))?;
            arithmetic::div_values(left_value, right_value, node_id)
        }
        Op::Neg(value) => {
            let value = read_value(values, *value, Some(node_id))?;
            arithmetic::neg_value(value, node_id)
        }
        Op::ElementwiseChain { input, ops } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let out = activations::apply_elementwise_chain(&input_tensor, ops)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Reshape { input, shape } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let out = input_tensor
                .reshape(shape.clone())
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Concat { inputs, axis } => {
            if inputs.len() < 2 {
                return Err(error(
                    "concat requires at least two inputs".to_string(),
                    Some(node_id),
                ));
            }
            let mut tensors = Vec::with_capacity(inputs.len());
            for input in inputs {
                tensors.push((*read_tensor(values, *input, node_id)?).clone());
            }
            let out =
                math::concat(&tensors, *axis).map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Gather {
            input,
            indices,
            axis,
        } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let out = math::gather(&input_tensor, indices, *axis)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::ConvTranspose2D {
            input,
            weight,
            stride,
            padding,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let weight_t = read_tensor(values, *weight, node_id)?;
            let out = crate::engine::ir::kernels::conv::conv_transpose2d_nchw(
                &input_t,
                &weight_t,
                (stride[0], stride[1]),
                (padding[0], padding[1]),
            )
            .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Upsample2D {
            input,
            scale_h,
            scale_w,
            mode,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let out = if *mode == 0 {
                crate::engine::ir::kernels::conv::upsample_nearest2d(
                    &input_t,
                    scale_h.round() as usize,
                    scale_w.round() as usize,
                )
                .map_err(|e| error(e.message, Some(node_id)))?
            } else {
                crate::engine::ir::kernels::conv::upsample_bilinear2d(&input_t, *scale_h, *scale_w)
                    .map_err(|e| error(e.message, Some(node_id)))?
            };
            Ok(runtime_from_tensor(out))
        }
        Op::Upsample2DBackward {
            upstream,
            orig_h,
            orig_w,
            scale_h,
            scale_w,
        } => {
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let out = crate::engine::ir::kernels::conv::upsample_nearest2d_backward(
                &upstream_t,
                *orig_h,
                *orig_w,
                *scale_h,
                *scale_w,
            )
            .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::MultiHeadAttention {
            q_input,
            k_input,
            v_input,
            w_q,
            w_k,
            w_v,
            w_o,
            bias_q,
            bias_k,
            bias_v,
            bias_o,
            num_heads,
            causal,
            output_idx,
        } => {
            let qt = read_tensor(values, *q_input, node_id)?;
            let kt = read_tensor(values, *k_input, node_id)?;
            let vt = read_tensor(values, *v_input, node_id)?;
            let wqt = read_tensor(values, *w_q, node_id)?;
            let wkt = read_tensor(values, *w_k, node_id)?;
            let wvt = read_tensor(values, *w_v, node_id)?;
            let wot = read_tensor(values, *w_o, node_id)?;
            let bqt = read_tensor(values, *bias_q, node_id)?;
            let bkt = read_tensor(values, *bias_k, node_id)?;
            let bvt = read_tensor(values, *bias_v, node_id)?;
            let bot = read_tensor(values, *bias_o, node_id)?;
            let out = crate::engine::ir::kernels::attention::multi_head_attention(
                &qt, &kt, &vt, &wqt, &wkt, &wvt, &wot, &bqt, &bkt, &bvt, &bot, *num_heads, *causal,
            )
            .map_err(|e| error(e.message, Some(node_id)))?;
            let tensor = match output_idx {
                0 => out.output,
                1 => out.attn_weights,
                2 => out.q_proj,
                3 => out.k_proj,
                4 => out.v_proj,
                5 => out.context,
                _ => {
                    return Err(error(
                        "MultiHeadAttention invalid output_idx".to_string(),
                        Some(node_id),
                    ));
                }
            };
            Ok(runtime_from_tensor(tensor))
        }
        Op::SinusoidalPE { input } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let out = crate::engine::ir::kernels::positional::sinusoidal_pe(&input_t)
                .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::RoPE { input, offset } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let out = crate::engine::ir::kernels::positional::rope(&input_t, *offset)
                .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::RoPEBackward { upstream, offset } => {
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let out = crate::engine::ir::kernels::positional::rope_backward(&upstream_t, *offset)
                .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Embedding { weight, indices } => {
            let weight_t = read_tensor(values, *weight, node_id)?;
            let indices_t = read_tensor(values, *indices, node_id)?;
            let out = math::embedding(&weight_t, &indices_t)
                .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::EmbeddingBackward {
            weight,
            indices,
            upstream,
        } => {
            let weight_t = read_tensor(values, *weight, node_id)?;
            let indices_t = read_tensor(values, *indices, node_id)?;
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let out = math::embedding_backward(&weight_t, &indices_t, &upstream_t)
                .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::LstmCell {
            x,
            h_prev,
            c_prev,
            weight_ih,
            weight_hh,
            bias,
            output_idx,
        } => {
            let x_t = read_tensor(values, *x, node_id)?;
            let h_t = read_tensor(values, *h_prev, node_id)?;
            let c_t = read_tensor(values, *c_prev, node_id)?;
            let wih = read_tensor(values, *weight_ih, node_id)?;
            let whh = read_tensor(values, *weight_hh, node_id)?;
            let b = read_tensor(values, *bias, node_id)?;
            let out = crate::engine::ir::kernels::rnn::lstm_cell(&x_t, &h_t, &c_t, &wih, &whh, &b)
                .map_err(|e| error(e.message, Some(node_id)))?;
            let tensor = match output_idx {
                0 => out.h_next,
                1 => out.c_next,
                2 => out.gates_raw,
                3 => out.tanh_c_next,
                _ => {
                    return Err(error(
                        "LstmCell invalid output_idx".to_string(),
                        Some(node_id),
                    ));
                }
            };
            Ok(runtime_from_tensor(tensor))
        }
        Op::LstmCellBackward {
            x,
            h_prev,
            c_prev,
            weight_ih,
            weight_hh,
            gates_raw,
            tanh_c_next,
            dh_next,
            dc_next,
            grad_target,
        } => {
            let x_t = read_tensor(values, *x, node_id)?;
            let h_t = read_tensor(values, *h_prev, node_id)?;
            let c_t = read_tensor(values, *c_prev, node_id)?;
            let wih = read_tensor(values, *weight_ih, node_id)?;
            let whh = read_tensor(values, *weight_hh, node_id)?;
            let gr = read_tensor(values, *gates_raw, node_id)?;
            let tc = read_tensor(values, *tanh_c_next, node_id)?;
            let dhn = read_tensor(values, *dh_next, node_id)?;
            let dcn = read_tensor(values, *dc_next, node_id)?;
            let grads = crate::engine::ir::kernels::rnn::lstm_cell_backward(
                &x_t, &h_t, &c_t, &wih, &whh, &gr, &tc, &dhn, &dcn,
            )
            .map_err(|e| error(e.message, Some(node_id)))?;
            let tensor = match grad_target {
                0 => grads.dx,
                1 => grads.dh_prev,
                2 => grads.dc_prev,
                3 => grads.dweight_ih,
                4 => grads.dweight_hh,
                5 => grads.dbias,
                _ => {
                    return Err(error(
                        "LstmCellBackward invalid grad_target".to_string(),
                        Some(node_id),
                    ));
                }
            };
            Ok(runtime_from_tensor(tensor))
        }
        Op::GruCell {
            x,
            h_prev,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            output_idx,
        } => {
            let x_t = read_tensor(values, *x, node_id)?;
            let h_t = read_tensor(values, *h_prev, node_id)?;
            let wih = read_tensor(values, *weight_ih, node_id)?;
            let whh = read_tensor(values, *weight_hh, node_id)?;
            let bih = read_tensor(values, *bias_ih, node_id)?;
            let bhh = read_tensor(values, *bias_hh, node_id)?;
            let out = crate::engine::ir::kernels::rnn::gru_cell(&x_t, &h_t, &wih, &whh, &bih, &bhh)
                .map_err(|e| error(e.message, Some(node_id)))?;
            let tensor = match output_idx {
                0 => out.h_next,
                1 => out.z_gate,
                2 => out.r_gate,
                3 => out.n_gate,
                _ => {
                    return Err(error(
                        "GruCell invalid output_idx".to_string(),
                        Some(node_id),
                    ));
                }
            };
            Ok(runtime_from_tensor(tensor))
        }
        Op::GruCellBackward {
            x,
            h_prev,
            weight_ih,
            weight_hh,
            z_gate,
            r_gate,
            n_gate,
            dh_next,
            grad_target,
        } => {
            let x_t = read_tensor(values, *x, node_id)?;
            let h_t = read_tensor(values, *h_prev, node_id)?;
            let wih = read_tensor(values, *weight_ih, node_id)?;
            let whh = read_tensor(values, *weight_hh, node_id)?;
            let z = read_tensor(values, *z_gate, node_id)?;
            let r = read_tensor(values, *r_gate, node_id)?;
            let n = read_tensor(values, *n_gate, node_id)?;
            let dhn = read_tensor(values, *dh_next, node_id)?;
            let grads = crate::engine::ir::kernels::rnn::gru_cell_backward(
                &x_t, &h_t, &wih, &whh, &z, &r, &n, &dhn,
            )
            .map_err(|e| error(e.message, Some(node_id)))?;
            let tensor = match grad_target {
                0 => grads.dx,
                1 => grads.dh_prev,
                2 => grads.dweight_ih,
                3 => grads.dweight_hh,
                4 => grads.dbias_ih,
                5 => grads.dbias_hh,
                _ => {
                    return Err(error(
                        "GruCellBackward invalid grad_target".to_string(),
                        Some(node_id),
                    ));
                }
            };
            Ok(runtime_from_tensor(tensor))
        }
        Op::Slice {
            input,
            starts,
            ends,
            axes,
        } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let out = input_tensor
                .slice(starts, ends, axes)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Transpose(value) => {
            let value = read_tensor(values, *value, node_id)?;
            let transposed = value
                .transpose_2d()
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(transposed))
        }
        Op::MatMul(left, right) => {
            let lhs = read_tensor(values, *left, node_id)?;
            let rhs = read_tensor(values, *right, node_id)?;
            let out = math::matmul(&lhs, &rhs).map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Plugin { operator, inputs } => {
            let mut input_values = Vec::with_capacity(inputs.len());
            for input in inputs {
                input_values.push(read_value(values, *input, Some(node_id))?);
            }
            operator.execute(&input_values, context)
        }
        Op::Relu(value) => {
            let tensor = read_tensor(values, *value, node_id)?;
            let out =
                activations::relu(&tensor).map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::ReluBackward(input, grad) => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let grad_tensor = read_tensor(values, *grad, node_id)?;
            let out = activations::relu_backward(&input_tensor, &grad_tensor)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Softmax(value) => {
            let tensor = read_tensor(values, *value, node_id)?;
            let out = reduce::softmax(&tensor).map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Log(value) => {
            let tensor = read_tensor(values, *value, node_id)?;
            let out =
                math::log_elementwise(&tensor).map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Exp(value) => {
            let tensor = read_tensor(values, *value, node_id)?;
            let out =
                math::exp_elementwise(&tensor).map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Sigmoid(value) => {
            let tensor = read_tensor(values, *value, node_id)?;
            let out =
                activations::sigmoid(&tensor).map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Gelu(value) => {
            let tensor = read_tensor(values, *value, node_id)?;
            let out =
                activations::gelu(&tensor).map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::GeluExact(value) => {
            let tensor = read_tensor(values, *value, node_id)?;
            let out = activations::gelu_exact(&tensor)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::SigmoidBackward(input, grad) => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let grad_tensor = read_tensor(values, *grad, node_id)?;
            let out = activations::sigmoid_backward(&input_tensor, &grad_tensor)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::GeluBackward(input, grad) | Op::GeluExactBackward(input, grad) => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let grad_tensor = read_tensor(values, *grad, node_id)?;
            let out = activations::gelu_backward(&input_tensor, &grad_tensor)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::SoftmaxCrossEntropyLossFromLogits { logits, targets } => {
            let logits_t = read_tensor(values, *logits, node_id)?;
            let targets_t = read_tensor(values, *targets, node_id)?;

            if logits_t.shape != targets_t.shape {
                return Err(error(
                    format!(
                        "Shape mismatch in SoftmaxCrossEntropyLossFromLogits: logits {:?}, targets {:?}",
                        logits_t.shape, targets_t.shape
                    ),
                    Some(node_id),
                ));
            }

            let batch_size = logits_t.shape[0];
            let num_classes = logits_t.shape[1];

            let mut loss_sum = 0.0_f32;

            for i in 0..batch_size {
                let start = i * num_classes;
                let end = start + num_classes;

                let logits_slice = &logits_t.data[start..end];
                let targets_slice = &targets_t.data[start..end];

                // 1. Stable log_softmax
                let max_logit = logits_slice
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0_f32;
                for &l in logits_slice {
                    sum_exp += (l - max_logit).exp();
                }
                let log_sum_exp = max_logit + sum_exp.ln();

                // 2. Compute cross entropy for this sample
                let mut sample_loss = 0.0_f32;
                for j in 0..num_classes {
                    let log_prob = logits_slice[j] - log_sum_exp;
                    sample_loss -= targets_slice[j] * log_prob;
                }

                loss_sum += sample_loss;
            }

            // Mean over batch
            let mean_loss = loss_sum / (batch_size as f32);

            Ok(RuntimeValue::Tensor(Arc::new(
                Tensor::new(vec![], vec![mean_loss]).unwrap(),
            )))
        }
        Op::ReduceSum {
            input,
            axis,
            keepdims,
        } => {
            let tensor = read_tensor(values, *input, node_id)?;
            let out = if *keepdims {
                tensor
                    .reduce_sum_keepdims(*axis)
                    .map_err(|err| error(err.message, Some(node_id)))?
            } else {
                reduce::reduce_sum(&tensor, *axis)
                    .map_err(|err| error(err.message, Some(node_id)))?
            };
            Ok(runtime_from_tensor(out))
        }
        Op::ReduceMax {
            input,
            axis,
            keepdims,
        } => {
            let tensor = read_tensor(values, *input, node_id)?;
            let out = if *keepdims {
                tensor
                    .reduce_max_keepdims(*axis)
                    .map_err(|err| error(err.message, Some(node_id)))?
            } else {
                reduce::reduce_max(&tensor, *axis)
                    .map_err(|err| error(err.message, Some(node_id)))?
            };
            Ok(runtime_from_tensor(out))
        }
        Op::ReduceMaxBackward {
            input,
            output_max,
            upstream,
            axis,
            keepdims: _,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let output_max_t = read_tensor(values, *output_max, node_id)?;
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let out = reduce::reduce_max_backward(&input_t, &output_max_t, &upstream_t, *axis)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::ReduceMean {
            input,
            axis,
            keepdims,
        } => {
            let tensor = read_tensor(values, *input, node_id)?;
            let out = if *keepdims {
                tensor
                    .reduce_mean_keepdims(*axis)
                    .map_err(|err| error(err.message, Some(node_id)))?
            } else {
                reduce::reduce_mean(&tensor, *axis)
                    .map_err(|err| error(err.message, Some(node_id)))?
            };
            Ok(runtime_from_tensor(out))
        }
        Op::Conv2D(input, weight) => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let weight_tensor = read_tensor(values, *weight, node_id)?;
            let out = conv::conv2d(&input_tensor, &weight_tensor)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Conv2DBackwardInput(input, weight, grad) => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let weight_tensor = read_tensor(values, *weight, node_id)?;
            let grad_tensor = read_tensor(values, *grad, node_id)?;
            let out = conv::conv2d_backward_input(&input_tensor, &weight_tensor, &grad_tensor)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Conv2DBackwardWeight(input, weight, grad) => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let weight_tensor = read_tensor(values, *weight, node_id)?;
            let grad_tensor = read_tensor(values, *grad, node_id)?;
            let out = conv::conv2d_backward_weight(
                &input_tensor,
                weight_tensor.shape.clone(),
                &grad_tensor,
            )
            .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::MaxPool {
            input,
            kernel_shape,
            strides,
            pads,
        } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let out = conv::pool2d_nchw(&input_tensor, kernel_shape, strides, pads, true)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::AvgPool {
            input,
            kernel_shape,
            strides,
            pads,
        } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let out = conv::pool2d_nchw(&input_tensor, kernel_shape, strides, pads, false)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::MaxPoolBackward {
            input,
            upstream,
            kernel_shape,
            strides,
            pads,
        } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let upstream_tensor = read_tensor(values, *upstream, node_id)?;
            let out = conv::max_pool2d_backward_nchw(
                &input_tensor,
                &upstream_tensor,
                kernel_shape,
                strides,
                pads,
            )
            .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::AvgPoolBackward {
            input,
            upstream,
            kernel_shape,
            strides,
            pads,
        } => {
            let input_tensor = read_tensor(values, *input, node_id)?;
            let upstream_tensor = read_tensor(values, *upstream, node_id)?;
            let out = conv::avg_pool2d_backward_nchw(
                &input_tensor,
                &upstream_tensor,
                kernel_shape,
                strides,
                pads,
            )
            .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::BatchNorm {
            input,
            weight,
            bias,
            mean,
            var,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let weight_t = read_tensor(values, *weight, node_id)?;
            let bias_t = read_tensor(values, *bias, node_id)?;
            let mean_t = read_tensor(values, *mean, node_id)?;
            let var_t = read_tensor(values, *var, node_id)?;
            let out = norm::batch_norm_nchw(&input_t, &weight_t, &bias_t, &mean_t, &var_t, 1e-5)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::BatchNormBackwardInput {
            input,
            upstream,
            weight,
            var,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let weight_t = read_tensor(values, *weight, node_id)?;
            let var_t = read_tensor(values, *var, node_id)?;
            let out = norm::batch_norm_backward_input_nchw(
                &input_t,
                &upstream_t,
                &weight_t,
                &var_t,
                1e-5,
            )
            .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::BatchNormBackwardWeight {
            input,
            upstream,
            mean,
            var,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let mean_t = read_tensor(values, *mean, node_id)?;
            let var_t = read_tensor(values, *var, node_id)?;
            let out =
                norm::batch_norm_backward_weight_nchw(&input_t, &upstream_t, &mean_t, &var_t, 1e-5)
                    .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::BatchNormBackwardBias { upstream } => {
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let out = norm::batch_norm_backward_bias_nchw(&upstream_t)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::LayerNorm {
            input,
            weight,
            bias,
            epsilon,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let weight_t = read_tensor(values, *weight, node_id)?;
            let bias_t = read_tensor(values, *bias, node_id)?;
            let out = norm::layer_norm(&input_t, &weight_t, &bias_t, *epsilon)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::LayerNormBackwardInput {
            input,
            upstream,
            weight,
            epsilon,
        } => {
            let x = read_tensor(values, *input, node_id)?;
            let dy = read_tensor(values, *upstream, node_id)?;
            let w = read_tensor(values, *weight, node_id)?;
            let out = norm::layer_norm_backward_input(&x, &dy, &w, *epsilon)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::LayerNormBackwardWeight {
            input,
            upstream,
            epsilon,
        } => {
            let x = read_tensor(values, *input, node_id)?;
            let dy = read_tensor(values, *upstream, node_id)?;
            let out = norm::layer_norm_backward_weight(&x, &dy, *epsilon)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::LayerNormBackwardBias { upstream } => {
            let dy = read_tensor(values, *upstream, node_id)?;
            let out = norm::layer_norm_backward_bias(&dy)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Flatten { input, axis } => {
            let tensor = read_tensor(values, *input, node_id)?;
            let left = tensor.shape[..*axis].iter().product::<usize>().max(1);
            let right = tensor.shape[*axis..].iter().product::<usize>().max(1);
            let out = tensor
                .reshape(vec![left, right])
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::GlobalAveragePool { input } => {
            let tensor = read_tensor(values, *input, node_id)?;
            let contig = tensor
                .make_contiguous()
                .map_err(|e| error(e.message, Some(node_id)))?;
            if contig.shape.len() < 2 {
                return Err(error(
                    "GlobalAveragePool expects at least rank-2 input".to_string(),
                    Some(node_id),
                ));
            }
            let n = contig.shape[0];
            let c = contig.shape[1];
            let spatial: usize = contig.shape[2..].iter().product::<usize>().max(1);
            let mut out_data = vec![0.0_f32; n * c];
            let scale = 1.0 / spatial as f32;
            for i in 0..n {
                for j in 0..c {
                    let base = (i * c + j) * spatial;
                    let s: f32 = contig.data[base..base + spatial].iter().sum();
                    out_data[i * c + j] = s * scale;
                }
            }
            // Output shape: [N, C] (flatten spatial dims)
            let out =
                Tensor::new(vec![n, c], out_data).map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::GlobalAveragePoolBackward { input, upstream } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let contig = input_t
                .make_contiguous()
                .map_err(|e| error(e.message, Some(node_id)))?;
            let up_c = upstream_t
                .make_contiguous()
                .map_err(|e| error(e.message, Some(node_id)))?;
            if contig.shape.len() < 2 {
                return Err(error(
                    "GlobalAveragePoolBackward expects at least rank-2 input".to_string(),
                    Some(node_id),
                ));
            }
            let n = contig.shape[0];
            let c = contig.shape[1];
            let spatial: usize = contig.shape[2..].iter().product::<usize>().max(1);
            let scale = 1.0 / spatial as f32;
            let mut grad = vec![0.0_f32; contig.data.len()];
            for i in 0..n {
                for j in 0..c {
                    let g = up_c.data[i * c + j] * scale;
                    let base = (i * c + j) * spatial;
                    for k in 0..spatial {
                        grad[base + k] = g;
                    }
                }
            }
            let out = Tensor::new(contig.shape.clone(), grad)
                .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::GroupNorm {
            input,
            weight,
            bias,
            num_groups,
            epsilon,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let weight_t = read_tensor(values, *weight, node_id)?;
            let bias_t = read_tensor(values, *bias, node_id)?;
            let out = crate::engine::ir::kernels::norm::group_norm(
                &input_t,
                &weight_t,
                &bias_t,
                *num_groups,
                *epsilon,
            )
            .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::GroupNormBackwardInput {
            input,
            upstream,
            weight,
            num_groups,
            epsilon,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let weight_t = read_tensor(values, *weight, node_id)?;
            let out = crate::engine::ir::kernels::norm::group_norm_backward_input(
                &input_t,
                &upstream_t,
                &weight_t,
                *num_groups,
                *epsilon,
            )
            .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::GroupNormBackwardWeight {
            input,
            upstream,
            num_groups,
            epsilon,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let out = crate::engine::ir::kernels::norm::group_norm_backward_weight(
                &input_t,
                &upstream_t,
                *num_groups,
                *epsilon,
            )
            .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::GroupNormBackwardBias { upstream } => {
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let out = crate::engine::ir::kernels::norm::group_norm_backward_bias(&upstream_t)
                .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::InstanceNorm {
            input,
            weight,
            bias,
            epsilon,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let weight_t = read_tensor(values, *weight, node_id)?;
            let bias_t = read_tensor(values, *bias, node_id)?;
            let out = crate::engine::ir::kernels::norm::instance_norm(
                &input_t, &weight_t, &bias_t, *epsilon,
            )
            .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::InstanceNormBackwardInput {
            input,
            upstream,
            weight,
            epsilon,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let weight_t = read_tensor(values, *weight, node_id)?;
            let out = crate::engine::ir::kernels::norm::instance_norm_backward_input(
                &input_t,
                &upstream_t,
                &weight_t,
                *epsilon,
            )
            .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::InstanceNormBackwardWeight {
            input,
            upstream,
            epsilon,
        } => {
            let input_t = read_tensor(values, *input, node_id)?;
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let out = crate::engine::ir::kernels::norm::instance_norm_backward_weight(
                &input_t,
                &upstream_t,
                *epsilon,
            )
            .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::InstanceNormBackwardBias { upstream } => {
            let upstream_t = read_tensor(values, *upstream, node_id)?;
            let out = crate::engine::ir::kernels::norm::instance_norm_backward_bias(&upstream_t)
                .map_err(|e| error(e.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::Gemm {
            lhs,
            rhs,
            bias,
            alpha,
            beta,
        } => {
            let lhs_t = read_tensor(values, *lhs, node_id)?;
            let rhs_t = read_tensor(values, *rhs, node_id)?;
            let bias_t = bias.map(|b| read_tensor(values, b, node_id)).transpose()?;
            let bias_ref = bias_t.as_deref();
            let out = lhs_t
                .gemm(&rhs_t, bias_ref, *alpha, *beta)
                .map_err(|err| error(err.message, Some(node_id)))?;
            Ok(runtime_from_tensor(out))
        }
        Op::GemmBackward { .. } => Err(error(
            "GemmBackward should be decomposed into MatMul in autograd".to_string(),
            Some(node_id),
        )),
        Op::Phi(_) => Err(error(
            "Unsupported op in phase 1: phi".to_string(),
            Some(node_id),
        )),
        Op::Removed => Err(error(
            "Removed node cannot be executed".to_string(),
            Some(node_id),
        )),
        Op::CustomCall { target, inputs, .. } => {
            // CustomCall is a backend-specific operation; the interpreter cannot execute it.
            // Return an error indicating the op requires a backend.
            Err(error(
                format!(
                    "CustomCall '{target}' with {} inputs cannot be executed by the CPU interpreter; requires a backend-specific dispatch",
                    inputs.len()
                ),
                Some(node_id),
            ))
        }
        Op::QuantizeLinear {
            input,
            scale,
            zero_point,
            bits,
        } => {
            let inp = read_tensor(values, *input, node_id)?;
            let clamp_min = -(1i32 << (*bits - 1)) as f32;
            let clamp_max = ((1i32 << (*bits - 1)) - 1) as f32;
            let data: Vec<f32> = inp
                .data
                .iter()
                .map(|&v| {
                    let q = (v / scale).round() + *zero_point as f32;
                    let q_clamped = q.max(clamp_min).min(clamp_max);
                    // Dequantize back: store as f32
                    (q_clamped - *zero_point as f32) * scale
                })
                .collect();
            Ok(runtime_from_tensor(
                Tensor::new(inp.shape.clone(), data)
                    .map_err(|e| error(e.message, Some(node_id)))?,
            ))
        }
        Op::DequantizeLinear {
            input,
            scale,
            zero_point,
        } => {
            let inp = read_tensor(values, *input, node_id)?;
            let data: Vec<f32> = inp
                .data
                .iter()
                .map(|&v| {
                    // Input is already dequantized float; just scale for correctness
                    v - *zero_point as f32 * scale
                })
                .collect();
            Ok(runtime_from_tensor(
                Tensor::new(inp.shape.clone(), data)
                    .map_err(|e| error(e.message, Some(node_id)))?,
            ))
        }
        Op::DepthwiseSeparableConv {
            input,
            dw_weight,
            pw_weight,
            stride,
            padding,
        } => {
            let inp = read_tensor(values, *input, node_id)?;
            let dw_w = read_tensor(values, *dw_weight, node_id)?;
            let pw_w = read_tensor(values, *pw_weight, node_id)?;
            let result = crate::engine::ir::kernels::conv::depthwise_separable_conv_nchw(
                &inp, &dw_w, &pw_w, *stride, *padding,
            )
            .map_err(|e| {
                error(
                    format!("DepthwiseSeparableConv: {}", e.message),
                    Some(node_id),
                )
            })?;
            Ok(runtime_from_tensor(result))
        }
        Op::Identity(value) => read_value(values, *value, Some(node_id)),
        Op::Dropout { input, .. } => read_value(values, *input, Some(node_id)),
        Op::MultiHeadAttentionBackward { .. } => Err(error(
            "MultiHeadAttentionBackward is not supported by the interpreter".to_string(),
            Some(node_id),
        )),
    }
}

fn read_tensor(
    values: &[Option<RuntimeValue>],
    id: ValueId,
    node: NodeId,
) -> Result<Arc<Tensor>, InterpreterError> {
    match read_value(values, id, Some(node))? {
        RuntimeValue::Tensor(tensor) => Ok(tensor),
        _ => Err(error(
            "Type mismatch: expected tensor input".to_string(),
            Some(node),
        )),
    }
}

fn read_value(
    values: &[Option<RuntimeValue>],
    id: ValueId,
    node: Option<NodeId>,
) -> Result<RuntimeValue, InterpreterError> {
    values
        .get(id.0)
        .ok_or_else(|| error(format!("ValueId out of range: {}", id.0), node))?
        .clone()
        .ok_or_else(|| error(format!("ValueId {} not computed", id.0), node))
}

fn runtime_from_tensor(tensor: Tensor) -> RuntimeValue {
    RuntimeValue::Tensor(Arc::new(tensor))
}

fn error(message: String, node: Option<NodeId>) -> InterpreterError {
    InterpreterError { message, node }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        Graph, Op, RuntimeValue, ValueId, execute, execute_value, execute_with_context,
    };

    #[test]
    fn executes_linear_integer_arithmetic() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        let (_, v2) = graph
            .add_op(block, Op::Add(v0, v1))
            .expect("add op should succeed");
        let (_, v3) = graph
            .add_op(block, Op::ConstInt(3))
            .expect("add op should succeed");
        let (_, v4) = graph
            .add_op(block, Op::Mul(v2, v3))
            .expect("add op should succeed");
        let (_, v5) = graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Div(v4, v5))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(result, Some(RuntimeValue::Int(4)));
    }

    #[test]
    fn executes_float_arithmetic() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstFloat(1.5))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstFloat(2.5))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(v0, v1))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(result, Some(RuntimeValue::Float(4.0)));
    }

    #[test]
    fn rejects_mixed_numeric_types() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstFloat(2.0))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Add(v0, v1))
            .expect("add op should succeed");

        let err = execute(&graph).expect_err("execute must fail");
        assert!(err.message.contains("Type mismatch"));
    }

    #[test]
    fn rejects_division_by_zero() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(8))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstInt(0))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Div(v0, v1))
            .expect("add op should succeed");

        let err = execute(&graph).expect_err("execute must fail");
        assert!(err.message.contains("Division by zero"));
    }

    #[test]
    fn fails_on_uninitialized_input_value() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        graph
            .add_op(block, Op::Add(ValueId(99), ValueId(1)))
            .expect("add op should succeed");

        let err = execute(&graph).expect_err("execute must fail");
        assert!(
            err.message.contains("not computed")
                || err.message.contains("out of range")
                || err.message.contains("missing producer")
        );
    }

    #[test]
    fn fails_on_double_assignment_to_same_value_id() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        graph
            .add_op(block, Op::ConstInt(1))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::ConstInt(2))
            .expect("add op should succeed");

        let duplicate_output = graph.nodes[0].output;
        graph.nodes[1].output = duplicate_output;

        let err = execute(&graph).expect_err("execute must fail");
        assert!(err.message.contains("assigned more than once"));
    }

    #[test]
    fn execute_value_can_read_non_terminal_result() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, v0) = graph
            .add_op(block, Op::ConstInt(5))
            .expect("add op should succeed");
        let (_, v1) = graph
            .add_op(block, Op::ConstInt(7))
            .expect("add op should succeed");
        let (_, add_value) = graph
            .add_op(block, Op::Add(v0, v1))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Mul(add_value, v1))
            .expect("add op should succeed");

        let result = execute_value(&graph, add_value).expect("execute_value should succeed");
        assert_eq!(result, RuntimeValue::Int(12));
    }

    #[test]
    fn executes_small_matmul_2x2() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, left) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2, 2],
                    data: vec![1.0, 2.0, 3.0, 4.0],
                },
            )
            .expect("add op should succeed");
        let (_, right) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2, 2],
                    data: vec![5.0, 6.0, 7.0, 8.0],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::MatMul(left, right))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(
            result,
            Some(RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![2, 2], vec![19.0, 22.0, 43.0, 50.0]).unwrap()
            )))
        );
    }

    #[test]
    fn relu_clamps_negative_values() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![4],
                    data: vec![-2.0, -0.5, 1.0, 3.5],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Relu(input))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(
            result,
            Some(RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![4], vec![0.0, 0.0, 1.0, 3.5]).unwrap()
            )))
        );
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![3],
                    data: vec![1.0, 2.0, 3.0],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Softmax(input))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        let RuntimeValue::Tensor(__tensor) = result.expect("output exists") else {
            panic!("expected tensor");
        };
        let data = &__tensor.data;

        let sum = data.iter().copied().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn conv2d_minimal_example() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![3, 3],
                    data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                },
            )
            .expect("add op should succeed");
        let (_, kernel) = graph
            .add_op(
                block,
                Op::ConstTensor {
                    shape: vec![2, 2],
                    data: vec![1.0, 0.0, 0.0, -1.0],
                },
            )
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Conv2D(input, kernel))
            .expect("add op should succeed");

        let result = execute(&graph).expect("execute should succeed");
        assert_eq!(
            result,
            Some(RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![2, 2], vec![-4.0, -4.0, -4.0, -4.0]).unwrap()
            )))
        );
    }

    #[test]
    fn input_and_output_ops_use_execution_context() {
        let mut graph = Graph::new();
        let block = graph.create_block();
        let (_, input) = graph
            .add_op(block, Op::Input("x".to_string()))
            .expect("add op should succeed");
        graph
            .add_op(block, Op::Output(input))
            .expect("add op should succeed");

        let mut context = ExecutionContext::default();
        context.inputs.insert(
            "x".to_string(),
            RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1], vec![42.0]).unwrap(),
            )),
        );

        let result = execute_with_context(&graph, &context).expect("execute should pass");
        assert_eq!(
            result,
            Some(RuntimeValue::Tensor(std::sync::Arc::new(
                crate::ir::tensor::Tensor::new(vec![1], vec![42.0]).unwrap()
            )))
        );
    }
}
