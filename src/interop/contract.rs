use std::collections::{HashMap, HashSet};

use crate::interop::InteropError;
use crate::ir::{Graph, Op, Tensor, ValueId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrContractVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrDataType {
    F32,
    I64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IrTensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: IrDataType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IrTensorLiteral {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IrOpContract {
    Input {
        name: String,
    },
    Parameter {
        name: String,
    },
    ConstTensor(IrTensorLiteral),
    Add {
        lhs: String,
        rhs: String,
    },
    Sub {
        lhs: String,
        rhs: String,
    },
    Mul {
        lhs: String,
        rhs: String,
    },
    Div {
        lhs: String,
        rhs: String,
    },
    Neg {
        input: String,
    },
    MatMul {
        lhs: String,
        rhs: String,
    },
    Transpose {
        input: String,
    },
    Relu {
        input: String,
    },
    Softmax {
        input: String,
    },
    Log {
        input: String,
    },
    Exp {
        input: String,
    },
    Sigmoid {
        input: String,
    },
    Gelu {
        input: String,
    },
    GeluExact {
        input: String,
    },
    ReduceSum {
        input: String,
        axis: Option<usize>,
        keepdims: bool,
    },
    ReduceMax {
        input: String,
        axis: Option<usize>,
        keepdims: bool,
    },
    ReduceMean {
        input: String,
        axis: Option<usize>,
        keepdims: bool,
    },
    /// Generalised matrix multiply: `alpha * (lhs × rhs) + beta * bias`.
    ///
    /// Matches the ONNX `Gemm` operator and the standard BLAS SGEMM interface.
    /// `bias` is optional — when `None` the operation is a scaled MatMul.
    Gemm {
        lhs: String,
        rhs: String,
        bias: Option<String>,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    },
    Reshape {
        input: String,
        shape: Vec<usize>,
    },
    Concat {
        inputs: Vec<String>,
        axis: usize,
    },
    Gather {
        input: String,
        indices: Vec<usize>,
        axis: usize,
    },
    Slice {
        input: String,
        starts: Vec<usize>,
        ends: Vec<usize>,
        axes: Vec<usize>,
    },
    Output {
        value: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct IrNodeContract {
    pub id: String,
    pub op: IrOpContract,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IrGraphContract {
    pub version: IrContractVersion,
    pub name: String,
    pub inputs: Vec<IrTensorSpec>,
    pub parameters: Vec<IrTensorSpec>,
    pub nodes: Vec<IrNodeContract>,
}

#[derive(Debug, Clone)]
pub struct ImportedProgram {
    pub graph: Graph,
    pub output: ValueId,
    pub parameter_defaults: HashMap<String, Tensor>,
}

impl IrGraphContract {
    /// Validates the IR graph contract for structural correctness.
    ///
    /// # Errors
    ///
    /// Returns `Err(InteropError)` if:
    /// - the node list is empty
    /// - any node ID is duplicated
    /// - an edge references an undefined node ID
    pub fn validate(&self) -> Result<(), InteropError> {
        if self.nodes.is_empty() {
            return Err(InteropError::new(
                "IR contract must contain at least one node",
            ));
        }

        let mut seen = HashSet::<&str>::new();
        for node in &self.nodes {
            if !seen.insert(node.id.as_str()) {
                return Err(InteropError::new(format!(
                    "duplicate IR node id '{}'",
                    node.id
                )));
            }
        }

        let mut id_set = HashSet::<&str>::new();
        for node in &self.nodes {
            id_set.insert(node.id.as_str());
        }

        for node in &self.nodes {
            for input in op_inputs(&node.op) {
                if !id_set.contains(input) {
                    return Err(InteropError::new(format!(
                        "node '{}' references unknown input '{}'",
                        node.id, input
                    )));
                }
            }
        }

        let output_nodes = self
            .nodes
            .iter()
            .filter(|node| matches!(node.op, IrOpContract::Output { .. }))
            .count();
        if output_nodes != 1 {
            return Err(InteropError::new(format!(
                "IR contract must contain exactly one Output node, got {output_nodes}"
            )));
        }

        for input in &self.inputs {
            if input.shape.is_empty() {
                return Err(InteropError::new(format!(
                    "input '{}' must have non-empty shape",
                    input.name
                )));
            }
        }
        for parameter in &self.parameters {
            if parameter.shape.is_empty() {
                return Err(InteropError::new(format!(
                    "parameter '{}' must have non-empty shape",
                    parameter.name
                )));
            }
        }

        for node in &self.nodes {
            match &node.op {
                IrOpContract::Reshape { shape, .. } => {
                    if shape.is_empty() {
                        return Err(InteropError::new(format!(
                            "node '{}' reshape target must be non-empty",
                            node.id
                        )));
                    }
                    if shape.contains(&0) {
                        return Err(InteropError::new(format!(
                            "node '{}' reshape target cannot contain zero dimension",
                            node.id
                        )));
                    }
                }
                IrOpContract::Concat { inputs, .. } => {
                    if inputs.len() < 2 {
                        return Err(InteropError::new(format!(
                            "node '{}' concat requires at least 2 inputs",
                            node.id
                        )));
                    }
                }
                IrOpContract::Gather { indices, .. } => {
                    if indices.is_empty() {
                        return Err(InteropError::new(format!(
                            "node '{}' gather requires at least one index",
                            node.id
                        )));
                    }
                }
                IrOpContract::Slice {
                    starts, ends, axes, ..
                } => {
                    if starts.is_empty() || ends.is_empty() || axes.is_empty() {
                        return Err(InteropError::new(format!(
                            "node '{}' slice starts/ends/axes must be non-empty",
                            node.id
                        )));
                    }
                    if starts.len() != ends.len() || starts.len() != axes.len() {
                        return Err(InteropError::new(format!(
                            "node '{}' slice starts/ends/axes lengths must match",
                            node.id
                        )));
                    }
                    if starts
                        .iter()
                        .zip(ends.iter())
                        .any(|(start, end)| start >= end)
                    {
                        return Err(InteropError::new(format!(
                            "node '{}' slice requires start < end for every axis",
                            node.id
                        )));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Compiles the validated IR graph into an [`ImportedProgram`].
    ///
    /// Calls [`validate`][Self::validate] internally. On success returns\\    
    /// the compiled program ready for execution.
    ///
    /// # Errors
    ///
    /// Returns `Err(InteropError)` if validation fails or any node
    /// cannot be lowered into the IR graph representation.
    pub fn compile(&self) -> Result<ImportedProgram, InteropError> {
        self.validate()?;

        let mut graph = Graph::new();
        let block = graph.create_block();
        let mut ids = HashMap::<String, ValueId>::new();
        let mut parameter_defaults = HashMap::<String, Tensor>::new();
        let mut output = None::<ValueId>;

        let input_shapes = self
            .inputs
            .iter()
            .map(|spec| (spec.name.as_str(), spec.shape.clone()))
            .collect::<HashMap<_, _>>();
        let parameter_shapes = self
            .parameters
            .iter()
            .map(|spec| (spec.name.as_str(), spec.shape.clone()))
            .collect::<HashMap<_, _>>();

        for node in &self.nodes {
            let value = match &node.op {
                IrOpContract::Input { name } => {
                    let shape = input_shapes.get(name.as_str()).ok_or_else(|| {
                        InteropError::new(format!("missing input spec for '{name}'"))
                    })?;
                    let (_, value) = graph
                        .add_op(block, Op::Input(name.clone()))
                        .map_err(|err| InteropError::new(err.message))?;
                    graph.bind_input_shape(name, shape.clone());
                    value
                }
                IrOpContract::Parameter { name } => {
                    let shape = parameter_shapes.get(name.as_str()).ok_or_else(|| {
                        InteropError::new(format!("missing parameter spec for '{name}'"))
                    })?;
                    let (_, value) = graph
                        .add_op(block, Op::Parameter(name.clone()))
                        .map_err(|err| InteropError::new(err.message))?;
                    graph.bind_parameter_shape(name, shape.clone());
                    value
                }
                IrOpContract::ConstTensor(tensor) => {
                    let (_, value) = graph
                        .add_op(
                            block,
                            Op::ConstTensor {
                                shape: tensor.shape.clone(),
                                data: tensor.data.clone(),
                            },
                        )
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Add { lhs, rhs } => {
                    let lhs = resolve_value(&ids, lhs)?;
                    let rhs = resolve_value(&ids, rhs)?;
                    let (_, value) = graph
                        .add_op(block, Op::Add(lhs, rhs))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Sub { lhs, rhs } => {
                    let lhs = resolve_value(&ids, lhs)?;
                    let rhs = resolve_value(&ids, rhs)?;
                    let (_, value) = graph
                        .add_op(block, Op::Sub(lhs, rhs))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Mul { lhs, rhs } => {
                    let lhs = resolve_value(&ids, lhs)?;
                    let rhs = resolve_value(&ids, rhs)?;
                    let (_, value) = graph
                        .add_op(block, Op::Mul(lhs, rhs))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Div { lhs, rhs } => {
                    let lhs = resolve_value(&ids, lhs)?;
                    let rhs = resolve_value(&ids, rhs)?;
                    let (_, value) = graph
                        .add_op(block, Op::Div(lhs, rhs))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Neg { input } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(block, Op::Neg(input))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::MatMul { lhs, rhs } => {
                    let lhs = resolve_value(&ids, lhs)?;
                    let rhs = resolve_value(&ids, rhs)?;
                    let (_, value) = graph
                        .add_op(block, Op::MatMul(lhs, rhs))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Transpose { input } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(block, Op::Transpose(input))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Relu { input } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(block, Op::Relu(input))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Softmax { input } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(block, Op::Softmax(input))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Log { input } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(block, Op::Log(input))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Exp { input } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(block, Op::Exp(input))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Sigmoid { input } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(block, Op::Sigmoid(input))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Gelu { input } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(block, Op::Gelu(input))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::GeluExact { input } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(block, Op::GeluExact(input))
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::ReduceSum {
                    input,
                    axis,
                    keepdims,
                } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(
                            block,
                            Op::ReduceSum {
                                input,
                                axis: *axis,
                                keepdims: *keepdims,
                            },
                        )
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::ReduceMax {
                    input,
                    axis,
                    keepdims,
                } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(
                            block,
                            Op::ReduceMax {
                                input,
                                axis: *axis,
                                keepdims: *keepdims,
                            },
                        )
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::ReduceMean {
                    input,
                    axis,
                    keepdims,
                } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(
                            block,
                            Op::ReduceMean {
                                input,
                                axis: *axis,
                                keepdims: *keepdims,
                            },
                        )
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Reshape { input, shape } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(
                            block,
                            Op::Reshape {
                                input,
                                shape: shape.clone(),
                            },
                        )
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Concat { inputs, axis } => {
                    let mut resolved = Vec::with_capacity(inputs.len());
                    for input in inputs {
                        resolved.push(resolve_value(&ids, input)?);
                    }
                    let (_, value) = graph
                        .add_op(
                            block,
                            Op::Concat {
                                inputs: resolved,
                                axis: *axis,
                            },
                        )
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Gather {
                    input,
                    indices,
                    axis,
                } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(
                            block,
                            Op::Gather {
                                input,
                                indices: indices.clone(),
                                axis: *axis,
                            },
                        )
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Slice {
                    input,
                    starts,
                    ends,
                    axes,
                } => {
                    let input = resolve_value(&ids, input)?;
                    let (_, value) = graph
                        .add_op(
                            block,
                            Op::Slice {
                                input,
                                starts: starts.clone(),
                                ends: ends.clone(),
                                axes: axes.clone(),
                            },
                        )
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
                IrOpContract::Output { value } => {
                    let value_id = resolve_value(&ids, value)?;
                    let (_, output_id) = graph
                        .add_op(block, Op::Output(value_id))
                        .map_err(|err| InteropError::new(err.message))?;
                    output = Some(output_id);
                    output_id
                }
                IrOpContract::Gemm {
                    lhs,
                    rhs,
                    bias,
                    alpha,
                    beta,
                    trans_a,
                    trans_b,
                } => {
                    let mut lhs_id = resolve_value(&ids, lhs)?;
                    let mut rhs_id = resolve_value(&ids, rhs)?;

                    if *trans_a {
                        let (_, transposed) = graph
                            .add_op(block, Op::Transpose(lhs_id))
                            .map_err(|err| InteropError::new(err.message))?;
                        lhs_id = transposed;
                    }
                    if *trans_b {
                        let (_, transposed) = graph
                            .add_op(block, Op::Transpose(rhs_id))
                            .map_err(|err| InteropError::new(err.message))?;
                        rhs_id = transposed;
                    }

                    let bias_id = bias
                        .as_deref()
                        .map(|b| resolve_value(&ids, b))
                        .transpose()?;
                    let (_, value) = graph
                        .add_op(
                            block,
                            Op::Gemm {
                                lhs: lhs_id,
                                rhs: rhs_id,
                                bias: bias_id,
                                alpha: *alpha,
                                beta: *beta,
                            },
                        )
                        .map_err(|err| InteropError::new(err.message))?;
                    value
                }
            };
            ids.insert(node.id.clone(), value);
        }

        crate::ir::shape_inference::infer_shapes(&graph).map_err(|err| {
            InteropError::new(format!(
                "import-time shape inference failed for graph '{}': {}",
                self.name, err.message
            ))
        })?;

        for spec in &self.parameters {
            let element_count = spec.shape.iter().copied().product::<usize>();
            if element_count == 0 {
                return Err(InteropError::new(format!(
                    "parameter '{}' has invalid zero-sized shape",
                    spec.name
                )));
            }
            parameter_defaults.insert(
                spec.name.clone(),
                Tensor::new(spec.shape.clone(), vec![0.0; element_count])
                    .map_err(|err| InteropError::new(err.message))?,
            );
        }

        Ok(ImportedProgram {
            graph,
            output: output.ok_or_else(|| InteropError::new("missing output node"))?,
            parameter_defaults,
        })
    }
}

fn resolve_value(ids: &HashMap<String, ValueId>, id: &str) -> Result<ValueId, InteropError> {
    ids.get(id)
        .copied()
        .ok_or_else(|| InteropError::new(format!("unknown value id '{id}'")))
}

#[must_use]
fn op_inputs(op: &IrOpContract) -> Vec<&str> {
    match op {
        IrOpContract::Input { .. }
        | IrOpContract::Parameter { .. }
        | IrOpContract::ConstTensor(_) => Vec::new(),
        IrOpContract::Add { lhs, rhs }
        | IrOpContract::Sub { lhs, rhs }
        | IrOpContract::Mul { lhs, rhs }
        | IrOpContract::Div { lhs, rhs }
        | IrOpContract::MatMul { lhs, rhs } => vec![lhs.as_str(), rhs.as_str()],
        IrOpContract::Neg { input }
        | IrOpContract::Transpose { input }
        | IrOpContract::Relu { input }
        | IrOpContract::Softmax { input }
        | IrOpContract::Log { input }
        | IrOpContract::Exp { input }
        | IrOpContract::Sigmoid { input }
        | IrOpContract::Gelu { input }
        | IrOpContract::GeluExact { input }
        | IrOpContract::ReduceSum { input, .. }
        | IrOpContract::ReduceMax { input, .. }
        | IrOpContract::ReduceMean { input, .. }
        | IrOpContract::Reshape { input, .. }
        | IrOpContract::Gather { input, .. }
        | IrOpContract::Slice { input, .. }
        | IrOpContract::Output { value: input } => vec![input.as_str()],
        IrOpContract::Concat { inputs, .. } => inputs.iter().map(String::as_str).collect(),
        IrOpContract::Gemm { lhs, rhs, bias, .. } => {
            let mut out = vec![lhs.as_str(), rhs.as_str()];
            if let Some(b) = bias {
                out.push(b.as_str());
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_compile_rejects_unknown_input_id() {
        let contract = IrGraphContract {
            version: IrContractVersion::V1,
            name: "bad".to_string(),
            inputs: vec![IrTensorSpec {
                name: "x".to_string(),
                shape: vec![1, 2],
                dtype: IrDataType::F32,
            }],
            parameters: vec![],
            nodes: vec![
                IrNodeContract {
                    id: "x".to_string(),
                    op: IrOpContract::Input {
                        name: "x".to_string(),
                    },
                },
                IrNodeContract {
                    id: "sum".to_string(),
                    op: IrOpContract::Add {
                        lhs: "x".to_string(),
                        rhs: "missing".to_string(),
                    },
                },
                IrNodeContract {
                    id: "out".to_string(),
                    op: IrOpContract::Output {
                        value: "sum".to_string(),
                    },
                },
            ],
        };

        let err = contract.compile().expect_err("contract must fail");
        assert!(err.message.contains("unknown input"));
    }
}
