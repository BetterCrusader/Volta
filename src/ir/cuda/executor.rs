use crate::ir::cuda::LoweredCudaPlan;
use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::kernels::{
    add::{add_f32, div_f32, mul_f32, neg_f32, sub_f32},
    backward::transpose_2d_f32,
    execute_node,
    matmul::matmul_f32,
    relu::{relu_backward_f32, relu_f32},
    softmax::softmax_f32,
};
use crate::ir::{
    DeterminismLevel, ElementwiseUnaryOp, ExecutionContext, ExecutionPlan, Graph, NodeId, Op,
    RuntimeValue, ValueId,
};

#[derive(Debug, Clone)]
pub struct CudaExecutionError {
    pub message: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CudaExecutor;

impl CudaExecutor {
    pub fn execute(&self, plan: &LoweredCudaPlan) -> Result<(), CudaExecutionError> {
        for node in &plan.executable_nodes {
            execute_node(node).map_err(|message| CudaExecutionError { message })?;
        }
        Ok(())
    }
}

pub fn execute_terminal_cuda(
    graph: &Graph,
    plan: &ExecutionPlan,
    ordered_nodes: &[NodeId],
    context: &ExecutionContext,
    determinism: DeterminismLevel,
) -> Result<Option<RuntimeValue>, CudaExecutionError> {
    let Some(target) = terminal_value_id(graph) else {
        return Ok(None);
    };
    execute_value_cuda(graph, plan, target, ordered_nodes, context, determinism).map(Some)
}

pub fn execute_value_cuda(
    graph: &Graph,
    _plan: &ExecutionPlan,
    target: ValueId,
    ordered_nodes: &[NodeId],
    context: &ExecutionContext,
    determinism: DeterminismLevel,
) -> Result<RuntimeValue, CudaExecutionError> {
    let device = CudaDevice::default_shared().map_err(|err| CudaExecutionError {
        message: format!("CUDA device init failed: {}", err.message),
    })?;

    let mut values = vec![None; graph.value_count()];
    for node_id in ordered_nodes {
        let node = graph
            .nodes
            .get(node_id.0)
            .ok_or_else(|| CudaExecutionError {
                message: format!("Schedule node out of range: {}", node_id.0),
            })?;

        let output_index = node.output.0;
        if output_index >= values.len() {
            return Err(CudaExecutionError {
                message: format!("Output ValueId out of range: {output_index}"),
            });
        }
        if values[output_index].is_some() {
            return Err(CudaExecutionError {
                message: format!(
                    "SSA violation: ValueId {} assigned more than once",
                    output_index
                ),
            });
        }

        let computed = evaluate_node_cuda(&device, &node.op, &values, context, determinism)?;
        values[output_index] = Some(computed);
    }

    read_value(&values, target)
}

fn evaluate_node_cuda(
    device: &CudaDevice,
    op: &Op,
    values: &[Option<RuntimeValue>],
    context: &ExecutionContext,
    determinism: DeterminismLevel,
) -> Result<RuntimeValue, CudaExecutionError> {
    match op {
        Op::ConstInt(value) => Ok(RuntimeValue::Int(*value)),
        Op::ConstFloat(value) => Ok(RuntimeValue::Float(*value)),
        Op::ConstTensor { shape, data } => Ok(RuntimeValue::Tensor {
            shape: shape.clone(),
            data: data.clone(),
        }),
        Op::Parameter(name) => {
            context
                .parameters
                .get(name)
                .cloned()
                .ok_or_else(|| CudaExecutionError {
                    message: format!("Missing parameter: '{name}'"),
                })
        }
        Op::Input(name) => context
            .inputs
            .get(name)
            .cloned()
            .ok_or_else(|| CudaExecutionError {
                message: format!("Missing input: '{name}'"),
            }),
        Op::Output(value) => read_value(values, *value),
        Op::Add(left, right) => {
            binary_op_cuda(device, values, *left, *right, determinism, BinaryOp::Add)
        }
        Op::Sub(left, right) => {
            binary_op_cuda(device, values, *left, *right, determinism, BinaryOp::Sub)
        }
        Op::Mul(left, right) => {
            binary_op_cuda(device, values, *left, *right, determinism, BinaryOp::Mul)
        }
        Op::Div(left, right) => div_op_cuda(device, values, *left, *right, determinism),
        Op::Neg(value) => {
            let value = read_value(values, *value)?;
            match value {
                RuntimeValue::Int(v) => {
                    let negated = v.checked_neg().ok_or_else(|| CudaExecutionError {
                        message: "Integer overflow in neg".to_string(),
                    })?;
                    Ok(RuntimeValue::Int(negated))
                }
                RuntimeValue::Float(v) => {
                    let negated = -v;
                    if !negated.is_finite() {
                        return Err(CudaExecutionError {
                            message: "Float overflow in neg".to_string(),
                        });
                    }
                    Ok(RuntimeValue::Float(negated))
                }
                RuntimeValue::Tensor { shape, data } => {
                    let out = neg_f32(device, &data, determinism).map_err(|message| {
                        CudaExecutionError {
                            message: format!("CUDA neg kernel failed: {message}"),
                        }
                    })?;
                    Ok(RuntimeValue::Tensor { shape, data: out })
                }
            }
        }
        Op::ElementwiseChain { input, ops } => {
            let (mut shape, mut data) = read_tensor(values, *input)?;
            for op in ops {
                match op {
                    ElementwiseUnaryOp::Neg => {
                        data = neg_f32(device, &data, determinism).map_err(|message| {
                            CudaExecutionError {
                                message: format!("CUDA neg kernel failed: {message}"),
                            }
                        })?;
                    }
                    ElementwiseUnaryOp::Relu => {
                        data = relu_f32(device, &data, determinism).map_err(|message| {
                            CudaExecutionError {
                                message: format!("CUDA relu kernel failed: {message}"),
                            }
                        })?;
                    }
                }
            }
            Ok(RuntimeValue::Tensor {
                shape: std::mem::take(&mut shape),
                data,
            })
        }
        Op::Transpose(value) => {
            let (shape, data) = read_tensor(values, *value)?;
            if shape.len() != 2 {
                return Err(CudaExecutionError {
                    message: format!("Transpose expects 2D tensor, got shape {:?}", shape),
                });
            }
            let rows = shape[0];
            let cols = shape[1];
            let out =
                transpose_2d_f32(device, &data, rows, cols, determinism).map_err(|message| {
                    CudaExecutionError {
                        message: format!("CUDA transpose kernel failed: {message}"),
                    }
                })?;
            Ok(RuntimeValue::Tensor {
                shape: vec![cols, rows],
                data: out,
            })
        }
        Op::MatMul(left, right) => {
            let (lhs_shape, lhs_data) = read_tensor(values, *left)?;
            let (rhs_shape, rhs_data) = read_tensor(values, *right)?;
            if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
                return Err(CudaExecutionError {
                    message: format!(
                        "MatMul expects 2D tensors, got lhs={:?} rhs={:?}",
                        lhs_shape, rhs_shape
                    ),
                });
            }
            if lhs_shape[1] != rhs_shape[0] {
                return Err(CudaExecutionError {
                    message: format!(
                        "Shape mismatch in MatMul: lhs {:?} rhs {:?}",
                        lhs_shape, rhs_shape
                    ),
                });
            }
            let m = lhs_shape[0];
            let k = lhs_shape[1];
            let n = rhs_shape[1];

            let out = matmul_f32(device, &lhs_data, &rhs_data, m, n, k, determinism).map_err(
                |message| CudaExecutionError {
                    message: format!("CUDA matmul kernel failed: {message}"),
                },
            )?;
            Ok(RuntimeValue::Tensor {
                shape: vec![m, n],
                data: out,
            })
        }
        Op::Relu(value) => {
            let (shape, data) = read_tensor(values, *value)?;
            let out =
                relu_f32(device, &data, determinism).map_err(|message| CudaExecutionError {
                    message: format!("CUDA relu kernel failed: {message}"),
                })?;
            Ok(RuntimeValue::Tensor { shape, data: out })
        }
        Op::ReluBackward(input, grad) => {
            let (input_shape, input_data) = read_tensor(values, *input)?;
            let (grad_shape, grad_data) = read_tensor(values, *grad)?;
            if input_shape != grad_shape {
                return Err(CudaExecutionError {
                    message: format!(
                        "ReluBackward shape mismatch: input {:?} grad {:?}",
                        input_shape, grad_shape
                    ),
                });
            }
            let out = relu_backward_f32(device, &input_data, &grad_data, determinism).map_err(
                |message| CudaExecutionError {
                    message: format!("CUDA relu backward kernel failed: {message}"),
                },
            )?;
            Ok(RuntimeValue::Tensor {
                shape: input_shape,
                data: out,
            })
        }
        Op::Softmax(value) => {
            let (shape, data) = read_tensor(values, *value)?;
            if shape.len() != 1 {
                return Err(CudaExecutionError {
                    message: format!("Softmax expects 1D tensor, got shape {:?}", shape),
                });
            }
            let out =
                softmax_f32(device, &data, determinism).map_err(|message| CudaExecutionError {
                    message: format!("CUDA softmax kernel failed: {message}"),
                })?;
            Ok(RuntimeValue::Tensor { shape, data: out })
        }
        Op::Log(value) => {
            let (shape, data) = read_tensor(values, *value)?;
            let mut out = Vec::with_capacity(data.len());
            for v in &data {
                if *v <= 0.0 {
                    return Err(CudaExecutionError {
                        message: format!("log of non-positive value: {}", v),
                    });
                }
                out.push(v.ln());
            }
            Ok(RuntimeValue::Tensor { shape, data: out })
        }
        Op::Exp(value) => {
            let (shape, data) = read_tensor(values, *value)?;
            let mut out = Vec::with_capacity(data.len());
            for v in &data {
                let e = v.exp();
                if !e.is_finite() {
                    return Err(CudaExecutionError {
                        message: format!("exp overflow for value: {}", v),
                    });
                }
                out.push(e);
            }
            Ok(RuntimeValue::Tensor { shape, data: out })
        }
        Op::ReduceSum { input, axis } => {
            let (shape, data) = read_tensor(values, *input)?;
            if data.is_empty() {
                return Err(CudaExecutionError {
                    message: "reduce_sum expects non-empty tensor".to_string(),
                });
            }
            match axis {
                None => {
                    let sum: f32 = data.iter().copied().sum();
                    Ok(RuntimeValue::Tensor {
                        shape: vec![1],
                        data: vec![sum],
                    })
                }
                Some(a) => {
                    let rank = shape.len();
                    if *a >= rank {
                        return Err(CudaExecutionError {
                            message: format!(
                                "reduce_sum axis {} out of bounds for rank {} tensor",
                                a, rank
                            ),
                        });
                    }
                    let outer: usize = shape[..*a].iter().product();
                    let axis_dim = shape[*a];
                    let inner: usize = shape[*a + 1..].iter().product();
                    let mut out_shape = shape;
                    out_shape.remove(*a);
                    if out_shape.is_empty() {
                        out_shape.push(1);
                    }
                    let mut out = vec![0.0_f32; outer * inner];
                    for o in 0..outer {
                        for i in 0..inner {
                            let mut acc = 0.0_f32;
                            for k in 0..axis_dim {
                                acc += data[o * axis_dim * inner + k * inner + i];
                            }
                            out[o * inner + i] = acc;
                        }
                    }
                    Ok(RuntimeValue::Tensor {
                        shape: out_shape,
                        data: out,
                    })
                }
            }
        }
        Op::Reshape { input, shape } => {
            let (_, data) = read_tensor(values, *input)?;
            let element_count: usize = shape.iter().product();
            if element_count != data.len() {
                return Err(CudaExecutionError {
                    message: format!(
                        "Reshape element mismatch: {} vs {}",
                        data.len(),
                        element_count
                    ),
                });
            }
            Ok(RuntimeValue::Tensor {
                shape: shape.clone(),
                data,
            })
        }
        Op::Concat { inputs, axis } => {
            if inputs.len() < 2 {
                return Err(CudaExecutionError {
                    message: "Concat requires at least 2 inputs".to_string(),
                });
            }
            let mut tensors: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(inputs.len());
            for id in inputs {
                tensors.push(read_tensor(values, *id)?);
            }
            let rank = tensors[0].0.len();
            if *axis >= rank {
                return Err(CudaExecutionError {
                    message: format!("Concat axis {} out of bounds for rank {}", axis, rank),
                });
            }
            let mut out_shape = tensors[0].0.clone();
            let mut axis_sum = 0usize;
            for (shape, _) in &tensors {
                if shape.len() != rank {
                    return Err(CudaExecutionError {
                        message: "Concat rank mismatch".to_string(),
                    });
                }
                axis_sum += shape[*axis];
            }
            out_shape[*axis] = axis_sum;
            let mut out = Vec::new();
            let outer: usize = tensors[0].0[..*axis].iter().product();
            let inner: usize = tensors[0].0[*axis + 1..].iter().product();
            for o in 0..outer {
                for (shape, data) in &tensors {
                    let axis_dim = shape[*axis];
                    let start = o * axis_dim * inner;
                    let end = start + axis_dim * inner;
                    out.extend_from_slice(&data[start..end]);
                }
            }
            Ok(RuntimeValue::Tensor {
                shape: out_shape,
                data: out,
            })
        }
        Op::Gather { .. } => Err(CudaExecutionError {
            message: "CUDA Gather execution is not implemented".to_string(),
        }),
        Op::Slice { input, starts, ends, axes } => {
            let (shape, data) = read_tensor(values, *input)?;
            let rank = shape.len();
            let mut out_shape = shape.clone();
            for idx in 0..axes.len() {
                let axis = axes[idx];
                let start = starts[idx];
                let end = ends[idx];
                if axis >= rank || start >= end || end > shape[axis] {
                    return Err(CudaExecutionError {
                        message: format!("Slice bounds error at axis {}", axis),
                    });
                }
                out_shape[axis] = end - start;
            }
            let out_count: usize = out_shape.iter().product();
            let mut out = vec![0.0_f32; out_count];
            for (linear, out_val) in out.iter_mut().enumerate().take(out_count) {
                let mut rem = linear;
                let mut in_offset = 0usize;
                let mut stride = 1usize;
                for dim_idx in (0..rank).rev() {
                    let coord = rem % out_shape[dim_idx];
                    rem /= out_shape[dim_idx];
                    let start_shift = axes
                        .iter()
                        .position(|a| *a == dim_idx)
                        .map(|i| starts[i])
                        .unwrap_or(0);
                    in_offset += (coord + start_shift) * stride;
                    stride *= shape[dim_idx];
                }
                *out_val = data[in_offset];
            }
            Ok(RuntimeValue::Tensor {
                shape: out_shape,
                data: out,
            })
        }
        Op::Conv2D(_, _) => Err(CudaExecutionError {
            message: "CUDA Conv2D execution is not implemented".to_string(),
        }),
        Op::Phi(_) => Err(CudaExecutionError {
            message: "CUDA phi execution is unsupported".to_string(),
        }),
        Op::Removed => Err(CudaExecutionError {
            message: "Removed node cannot be executed".to_string(),
        }),
    }
}

#[derive(Debug, Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
}

type TensorBinaryKernel =
    fn(&CudaDevice, &[f32], &[f32], DeterminismLevel) -> Result<Vec<f32>, String>;

fn binary_op_cuda(
    device: &CudaDevice,
    values: &[Option<RuntimeValue>],
    left: ValueId,
    right: ValueId,
    determinism: DeterminismLevel,
    op: BinaryOp,
) -> Result<RuntimeValue, CudaExecutionError> {
    let left = read_value(values, left)?;
    let right = read_value(values, right)?;
    match (left, right) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => {
            let value = apply_int_binary(op, a, b)?;
            Ok(RuntimeValue::Int(value))
        }
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
            let value = apply_float_binary(op, a, b)?;
            Ok(RuntimeValue::Float(value))
        }
        (
            RuntimeValue::Tensor {
                shape: left_shape,
                data: left_data,
            },
            RuntimeValue::Tensor {
                shape: right_shape,
                data: right_data,
            },
        ) => {
            if left_shape != right_shape {
                return Err(CudaExecutionError {
                    message: format!(
                        "Tensor shape mismatch: left {:?} right {:?}",
                        left_shape, right_shape
                    ),
                });
            }
            let kernel = tensor_kernel_for(op);
            let out = kernel(device, &left_data, &right_data, determinism).map_err(|message| {
                CudaExecutionError {
                    message: format!("CUDA {} kernel failed: {message}", op_name(op)),
                }
            })?;
            Ok(RuntimeValue::Tensor {
                shape: left_shape,
                data: out,
            })
        }
        _ => Err(CudaExecutionError {
            message: "Type mismatch in binary operation".to_string(),
        }),
    }
}

fn div_op_cuda(
    device: &CudaDevice,
    values: &[Option<RuntimeValue>],
    left: ValueId,
    right: ValueId,
    determinism: DeterminismLevel,
) -> Result<RuntimeValue, CudaExecutionError> {
    let left = read_value(values, left)?;
    let right = read_value(values, right)?;
    match (left, right) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => {
            if b == 0 {
                return Err(CudaExecutionError {
                    message: "Division by zero".to_string(),
                });
            }
            let value = a.checked_div(b).ok_or_else(|| CudaExecutionError {
                message: "Integer overflow in div".to_string(),
            })?;
            Ok(RuntimeValue::Int(value))
        }
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
            if b == 0.0 {
                return Err(CudaExecutionError {
                    message: "Division by zero".to_string(),
                });
            }
            let value = a / b;
            if !value.is_finite() {
                return Err(CudaExecutionError {
                    message: "Float overflow in div".to_string(),
                });
            }
            Ok(RuntimeValue::Float(value))
        }
        (
            RuntimeValue::Tensor {
                shape: left_shape,
                data: left_data,
            },
            RuntimeValue::Tensor {
                shape: right_shape,
                data: right_data,
            },
        ) => {
            if left_shape != right_shape {
                return Err(CudaExecutionError {
                    message: format!(
                        "Tensor shape mismatch in div: left {:?} right {:?}",
                        left_shape, right_shape
                    ),
                });
            }
            if tensor_has_zero_divisor(&right_data) {
                return Err(CudaExecutionError {
                    message: "Division by zero".to_string(),
                });
            }
            let out = div_f32(device, &left_data, &right_data, determinism).map_err(|message| {
                CudaExecutionError {
                    message: format!("CUDA div kernel failed: {message}"),
                }
            })?;
            Ok(RuntimeValue::Tensor {
                shape: left_shape,
                data: out,
            })
        }
        _ => Err(CudaExecutionError {
            message: "Type mismatch in div".to_string(),
        }),
    }
}

fn apply_int_binary(op: BinaryOp, left: i64, right: i64) -> Result<i64, CudaExecutionError> {
    match op {
        BinaryOp::Add => left.checked_add(right).ok_or_else(|| CudaExecutionError {
            message: "Integer overflow in add".to_string(),
        }),
        BinaryOp::Sub => left.checked_sub(right).ok_or_else(|| CudaExecutionError {
            message: "Integer overflow in sub".to_string(),
        }),
        BinaryOp::Mul => left.checked_mul(right).ok_or_else(|| CudaExecutionError {
            message: "Integer overflow in mul".to_string(),
        }),
    }
}

fn apply_float_binary(op: BinaryOp, left: f64, right: f64) -> Result<f64, CudaExecutionError> {
    let value = match op {
        BinaryOp::Add => left + right,
        BinaryOp::Sub => left - right,
        BinaryOp::Mul => left * right,
    };
    if !value.is_finite() {
        return Err(CudaExecutionError {
            message: format!("Float overflow in {}", op_name(op)),
        });
    }
    Ok(value)
}

fn tensor_kernel_for(op: BinaryOp) -> TensorBinaryKernel {
    match op {
        BinaryOp::Add => add_f32,
        BinaryOp::Sub => sub_f32,
        BinaryOp::Mul => mul_f32,
    }
}

fn op_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
    }
}

fn tensor_has_zero_divisor(values: &[f32]) -> bool {
    values.contains(&0.0)
}

fn terminal_value_id(graph: &Graph) -> Option<ValueId> {
    for node in graph.nodes.iter().rev() {
        if let Op::Output(value) = node.op {
            return Some(value);
        }
    }
    graph.last_value_id()
}

fn read_value(
    values: &[Option<RuntimeValue>],
    id: ValueId,
) -> Result<RuntimeValue, CudaExecutionError> {
    values
        .get(id.0)
        .ok_or_else(|| CudaExecutionError {
            message: format!("ValueId out of range: {}", id.0),
        })?
        .clone()
        .ok_or_else(|| CudaExecutionError {
            message: format!("ValueId {} not computed", id.0),
        })
}

fn read_tensor(
    values: &[Option<RuntimeValue>],
    id: ValueId,
) -> Result<(Vec<usize>, Vec<f32>), CudaExecutionError> {
    match read_value(values, id)? {
        RuntimeValue::Tensor { shape, data } => Ok((shape, data)),
        _ => Err(CudaExecutionError {
            message: format!("Expected tensor runtime value for ValueId {}", id.0),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryOp, apply_float_binary, apply_int_binary, tensor_has_zero_divisor};

    #[test]
    fn int_binary_reports_overflow() {
        let err = apply_int_binary(BinaryOp::Add, i64::MAX, 1)
            .expect_err("i64 add overflow should be reported");
        assert!(err.message.contains("Integer overflow in add"));
    }

    #[test]
    fn float_binary_reports_overflow() {
        let err = apply_float_binary(BinaryOp::Mul, f64::MAX, 2.0)
            .expect_err("f64 mul overflow should be reported");
        assert!(err.message.contains("Float overflow in mul"));
    }

    #[test]
    fn zero_divisor_guard_handles_negative_zero() {
        assert!(tensor_has_zero_divisor(&[1.0, -0.0, 3.0]));
        assert!(!tensor_has_zero_divisor(&[1.0, 2.0, 3.0]));
    }
}
