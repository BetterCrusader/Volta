use crate::interop::contract::{
    ImportedProgram, IrContractVersion, IrDataType, IrGraphContract, IrNodeContract, IrOpContract,
    IrTensorLiteral, IrTensorSpec,
};
use crate::interop::{InteropError, PluginRegistry};
use prost::Message;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tract_onnx::pb;

#[derive(Debug, Clone, PartialEq)]
pub enum OnnxOpStub {
    Input {
        name: String,
        shape: Vec<usize>,
    },
    Parameter {
        name: String,
        shape: Vec<usize>,
    },
    ConstTensor {
        shape: Vec<usize>,
        data: Vec<f32>,
    },
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
    /// Matches ONNX `Gemm` op (opset 11+).
    Gemm {
        lhs: String,
        rhs: String,
        bias: Option<String>,
        alpha: f32,
        beta: f32,
    },
    Output {
        value: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnnxNodeStub {
    pub id: String,
    pub op: OnnxOpStub,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnnxGraphStub {
    pub name: String,
    pub nodes: Vec<OnnxNodeStub>,
}

#[derive(Default, Clone)]
pub struct OnnxImporter {
    plugins: PluginRegistry,
}

impl OnnxImporter {
    pub fn new(plugins: PluginRegistry) -> Self {
        Self { plugins }
    }

    pub fn import_graph(&self, source: &OnnxGraphStub) -> Result<ImportedProgram, InteropError> {
        let mut input_specs = Vec::<IrTensorSpec>::new();
        let mut parameter_specs = Vec::<IrTensorSpec>::new();
        let mut nodes = Vec::<IrNodeContract>::new();

        for node in &source.nodes {
            let ir_op = match &node.op {
                OnnxOpStub::Input { name, shape } => {
                    input_specs.push(IrTensorSpec {
                        name: name.clone(),
                        shape: shape.clone(),
                        dtype: IrDataType::F32,
                    });
                    IrOpContract::Input { name: name.clone() }
                }
                OnnxOpStub::Parameter { name, shape } => {
                    parameter_specs.push(IrTensorSpec {
                        name: name.clone(),
                        shape: shape.clone(),
                        dtype: IrDataType::F32,
                    });
                    IrOpContract::Parameter { name: name.clone() }
                }
                OnnxOpStub::ConstTensor { shape, data } => {
                    IrOpContract::ConstTensor(IrTensorLiteral {
                        shape: shape.clone(),
                        data: data.clone(),
                    })
                }
                OnnxOpStub::Add { lhs, rhs } => IrOpContract::Add {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                },
                OnnxOpStub::Sub { lhs, rhs } => IrOpContract::Sub {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                },
                OnnxOpStub::Mul { lhs, rhs } => IrOpContract::Mul {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                },
                OnnxOpStub::Div { lhs, rhs } => IrOpContract::Div {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                },
                OnnxOpStub::Neg { input } => IrOpContract::Neg {
                    input: input.clone(),
                },
                OnnxOpStub::MatMul { lhs, rhs } => IrOpContract::MatMul {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                },
                OnnxOpStub::Transpose { input } => IrOpContract::Transpose {
                    input: input.clone(),
                },
                OnnxOpStub::Relu { input } => IrOpContract::Relu {
                    input: input.clone(),
                },
                OnnxOpStub::Softmax { input } => IrOpContract::Softmax {
                    input: input.clone(),
                },
                OnnxOpStub::Reshape { input, shape } => IrOpContract::Reshape {
                    input: input.clone(),
                    shape: shape.clone(),
                },
                OnnxOpStub::Concat { inputs, axis } => IrOpContract::Concat {
                    inputs: inputs.clone(),
                    axis: *axis,
                },
                OnnxOpStub::Gather {
                    input,
                    indices,
                    axis,
                } => IrOpContract::Gather {
                    input: input.clone(),
                    indices: indices.clone(),
                    axis: *axis,
                },
                OnnxOpStub::Slice {
                    input,
                    starts,
                    ends,
                    axes,
                } => IrOpContract::Slice {
                    input: input.clone(),
                    starts: starts.clone(),
                    ends: ends.clone(),
                    axes: axes.clone(),
                },
                OnnxOpStub::Log { input } => IrOpContract::Log {
                    input: input.clone(),
                },
                OnnxOpStub::Exp { input } => IrOpContract::Exp {
                    input: input.clone(),
                },
                OnnxOpStub::Sigmoid { input } => IrOpContract::Sigmoid {
                    input: input.clone(),
                },
                OnnxOpStub::Gelu { input } => IrOpContract::Gelu {
                    input: input.clone(),
                },
                OnnxOpStub::ReduceSum {
                    input,
                    axis,
                    keepdims,
                } => IrOpContract::ReduceSum {
                    input: input.clone(),
                    axis: *axis,
                    keepdims: *keepdims,
                },
                OnnxOpStub::ReduceMax {
                    input,
                    axis,
                    keepdims,
                } => IrOpContract::ReduceMax {
                    input: input.clone(),
                    axis: *axis,
                    keepdims: *keepdims,
                },
                OnnxOpStub::ReduceMean {
                    input,
                    axis,
                    keepdims,
                } => IrOpContract::ReduceMean {
                    input: input.clone(),
                    axis: *axis,
                    keepdims: *keepdims,
                },
                OnnxOpStub::Gemm {
                    lhs,
                    rhs,
                    bias,
                    alpha,
                    beta,
                } => IrOpContract::Gemm {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    bias: bias.clone(),
                    alpha: *alpha,
                    beta: *beta,
                    trans_a: false,
                    trans_b: false,
                },
                OnnxOpStub::Output { value } => IrOpContract::Output {
                    value: value.clone(),
                },
            };

            let contract_node = IrNodeContract {
                id: node.id.clone(),
                op: ir_op,
            };
            if let Some(plugin) = self.plugins.find(op_kind(&node.op)) {
                plugin.validate_node(&contract_node)?;
            }
            nodes.push(contract_node);
        }

        IrGraphContract {
            version: IrContractVersion::V1,
            name: source.name.clone(),
            inputs: input_specs,
            parameters: parameter_specs,
            nodes,
        }
        .compile()
    }

    pub fn import_model_proto(
        &self,
        model: &pb::ModelProto,
    ) -> Result<ImportedProgram, InteropError> {
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| InteropError::new("ONNX model is missing graph"))?;
        let contract = self.contract_from_proto_graph(graph)?;
        contract.compile()
    }

    fn contract_from_proto_graph(
        &self,
        graph: &pb::GraphProto,
    ) -> Result<IrGraphContract, InteropError> {
        let mut nodes = Vec::<IrNodeContract>::new();
        let mut input_specs = Vec::<IrTensorSpec>::new();
        let mut parameter_specs = Vec::<IrTensorSpec>::new();
        let mut value_ranks = HashMap::<String, usize>::new();

        let initializer_names: HashSet<&str> = graph
            .initializer
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        let initializer_lookup = graph
            .initializer
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect::<HashMap<_, _>>();

        for value in &graph.input {
            if initializer_names.contains(value.name.as_str()) {
                continue;
            }
            let mut spec = tensor_spec_from_value_info(value)?;
            spec.name = value.name.clone();
            value_ranks.insert(value.name.clone(), spec.shape.len());
            input_specs.push(spec.clone());
            nodes.push(IrNodeContract {
                id: value.name.clone(),
                op: IrOpContract::Input {
                    name: value.name.clone(),
                },
            });
        }

        for tensor in &graph.initializer {
            let spec = tensor_spec_from_initializer(tensor)?;
            value_ranks.insert(tensor.name.clone(), spec.shape.len());
            parameter_specs.push(spec.clone());
            nodes.push(IrNodeContract {
                id: tensor.name.clone(),
                op: IrOpContract::ConstTensor(parse_initializer_literal(tensor)?),
            });
        }

        for proto_node in &graph.node {
            let id = first_non_empty(&proto_node.output).ok_or_else(|| {
                InteropError::new(format!(
                    "ONNX node '{}' has no non-empty outputs",
                    proto_node.name
                ))
            })?;

            let op = map_proto_op(proto_node, &initializer_lookup, &value_ranks)?;
            let node = IrNodeContract {
                id: id.to_string(),
                op,
            };
            if let Some(plugin) = self.plugins.find(op_kind_from_contract(&node.op)) {
                plugin.validate_node(&node)?;
            }
            if let Some(rank) = inferred_output_rank(&node.op, &value_ranks) {
                value_ranks.insert(node.id.clone(), rank);
            }
            nodes.push(node);
        }

        let output_name = graph
            .output
            .first()
            .map(|value| value.name.clone())
            .filter(|name| !name.is_empty())
            .ok_or_else(|| InteropError::new("ONNX graph must have one non-empty output"))?;

        nodes.push(IrNodeContract {
            id: "__volta_output".to_string(),
            op: IrOpContract::Output { value: output_name },
        });

        Ok(IrGraphContract {
            version: IrContractVersion::V1,
            name: graph.name.clone(),
            inputs: input_specs,
            parameters: parameter_specs,
            nodes,
        })
    }
}

pub fn import_onnx_stub_graph(source: &OnnxGraphStub) -> Result<ImportedProgram, InteropError> {
    OnnxImporter::default().import_graph(source)
}

/// Imports a serialized ONNX protobuf model from raw bytes.
///
/// # Errors
/// Returns [`InteropError`] when protobuf decoding fails or when the importer
/// encounters unsupported operators, attributes, dtypes, or shape constraints.
pub fn import_onnx_bytes(bytes: &[u8]) -> Result<ImportedProgram, InteropError> {
    let model = pb::ModelProto::decode(bytes)
        .map_err(|err| InteropError::new(format!("failed to decode ONNX protobuf: {err}")))?;
    OnnxImporter::default().import_model_proto(&model)
}

/// Imports an ONNX model from a filesystem path.
///
/// # Errors
/// Returns [`InteropError`] if the file cannot be read, protobuf decoding
/// fails, or graph import validation fails.
pub fn import_onnx_file(path: impl AsRef<Path>) -> Result<ImportedProgram, InteropError> {
    let data = std::fs::read(path.as_ref()).map_err(|err| {
        InteropError::new(format!(
            "failed to read ONNX file '{}': {err}",
            path.as_ref().display()
        ))
    })?;
    import_onnx_bytes(&data)
}

fn first_non_empty(items: &[String]) -> Option<&str> {
    items
        .iter()
        .find(|item| !item.is_empty())
        .map(|item| item.as_str())
}

fn require_input(node: &pb::NodeProto, index: usize) -> Result<&str, InteropError> {
    node.input
        .get(index)
        .map(String::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            InteropError::new(format!(
                "node '{}' missing required input #{}",
                node.op_type, index
            ))
        })
}

fn require_initializer_input<'a>(
    node: &pb::NodeProto,
    index: usize,
    initializers: &HashMap<&'a str, &'a pb::TensorProto>,
) -> Result<&'a pb::TensorProto, InteropError> {
    let name = require_input(node, index)?;
    initializers.get(name).copied().ok_or_else(|| {
        InteropError::new(format!(
            "node '{}' input '{}' must be initializer for Wave 2 static import",
            node.op_type, name
        ))
    })
}

fn attribute_i64(node: &pb::NodeProto, key: &str) -> Option<i64> {
    node.attribute
        .iter()
        .find(|attr| attr.name == key)
        .map(|attr| attr.i)
}

/// Reads a float (`FLOAT`) attribute by name from an ONNX node.
///
/// Returns `None` if the attribute is absent or has a different type.
fn attribute_f32(node: &pb::NodeProto, key: &str) -> Option<f32> {
    node.attribute
        .iter()
        .find(|attr| attr.name == key)
        .map(|attr| attr.f)
}

fn attribute_string(node: &pb::NodeProto, key: &str) -> Option<String> {
    node.attribute
        .iter()
        .find(|attr| attr.name == key)
        .and_then(|attr| String::from_utf8(attr.s.clone()).ok())
}

fn attribute_i64s(node: &pb::NodeProto, key: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|attr| attr.name == key)
        .map(|attr| attr.ints.clone())
}

fn parse_reduce_axis(
    node: &pb::NodeProto,
    initializers: &HashMap<&str, &pb::TensorProto>,
    op_name: &str,
    input_rank: Option<usize>,
) -> Result<Option<usize>, InteropError> {
    if let Some(axis_name) = node.input.get(1).filter(|name| !name.is_empty()) {
        let axis_tensor = initializers
            .get(axis_name.as_str())
            .copied()
            .ok_or_else(|| {
                InteropError::new(format!(
                    "node '{}' {} axes input '{}' must be initializer for static import",
                    node.name, op_name, axis_name
                ))
            })?;
        let axes = parse_reduce_axes(axis_tensor, node, op_name, input_rank)?;
        return single_axis_from_list(&axes, node, op_name);
    }

    if let Some(axes) = attribute_i64s(node, "axes") {
        let axes = normalize_reduce_axes(&axes, node, op_name, input_rank)?;
        return single_axis_from_list(&axes, node, op_name);
    }

    if let Some(axis) = attribute_i64(node, "axis") {
        let axes = normalize_reduce_axes(&[axis], node, op_name, input_rank)?;
        return single_axis_from_list(&axes, node, op_name);
    }

    Ok(None)
}

fn parse_reduce_axes(
    tensor: &pb::TensorProto,
    node: &pb::NodeProto,
    op_name: &str,
    input_rank: Option<usize>,
) -> Result<Vec<usize>, InteropError> {
    let axes = parse_tensor_data_as_i64(tensor)?;
    if axes.is_empty() {
        return Ok(Vec::new());
    }
    normalize_reduce_axes(&axes, node, op_name, input_rank)
}

fn normalize_reduce_axes(
    axes: &[i64],
    node: &pb::NodeProto,
    op_name: &str,
    input_rank: Option<usize>,
) -> Result<Vec<usize>, InteropError> {
    axes.iter()
        .map(|&axis| normalize_single_axis(axis, node, op_name, input_rank))
        .collect()
}

fn normalize_single_axis(
    axis: i64,
    node: &pb::NodeProto,
    op_name: &str,
    input_rank: Option<usize>,
) -> Result<usize, InteropError> {
    if axis >= 0 {
        return Ok(axis as usize);
    }

    let Some(rank) = input_rank else {
        return Err(InteropError::new(format!(
            "node '{}' {} negative axis {} requires known input rank",
            node.name, op_name, axis
        )));
    };
    let rank_i64 = rank as i64;
    let normalized = rank_i64 + axis;
    if normalized < 0 || normalized >= rank_i64 {
        return Err(InteropError::new(format!(
            "node '{}' {} axis {} is out of bounds for rank {}",
            node.name, op_name, axis, rank
        )));
    }
    Ok(normalized as usize)
}

fn single_axis_from_list(
    axes: &[usize],
    node: &pb::NodeProto,
    op_name: &str,
) -> Result<Option<usize>, InteropError> {
    if axes.is_empty() {
        Ok(None)
    } else if axes.len() == 1 {
        Ok(Some(axes[0]))
    } else {
        Err(InteropError::new(format!(
            "node '{}' {} supports only a single axis in Wave 2 static import",
            node.name, op_name
        )))
    }
}

fn parse_usize_list(tensor: &pb::TensorProto, label: &str) -> Result<Vec<usize>, InteropError> {
    let dims = parse_tensor_data_as_i64(tensor)?;
    if dims.is_empty() {
        return Err(InteropError::new(format!(
            "initializer '{}' for {} must be non-empty",
            tensor.name, label
        )));
    }
    dims.into_iter()
        .map(|dim| {
            if dim < 0 {
                Err(InteropError::new(format!(
                    "initializer '{}' for {} contains negative dimension/index {}",
                    tensor.name, label, dim
                )))
            } else {
                Ok(dim as usize)
            }
        })
        .collect()
}

fn map_proto_op(
    node: &pb::NodeProto,
    initializers: &HashMap<&str, &pb::TensorProto>,
    value_ranks: &HashMap<String, usize>,
) -> Result<IrOpContract, InteropError> {
    let op = node.op_type.as_str();
    Ok(match op {
        "Add" => IrOpContract::Add {
            lhs: require_input(node, 0)?.to_string(),
            rhs: require_input(node, 1)?.to_string(),
        },
        "Sub" => IrOpContract::Sub {
            lhs: require_input(node, 0)?.to_string(),
            rhs: require_input(node, 1)?.to_string(),
        },
        "Mul" => IrOpContract::Mul {
            lhs: require_input(node, 0)?.to_string(),
            rhs: require_input(node, 1)?.to_string(),
        },
        "Div" => IrOpContract::Div {
            lhs: require_input(node, 0)?.to_string(),
            rhs: require_input(node, 1)?.to_string(),
        },
        "Neg" => IrOpContract::Neg {
            input: require_input(node, 0)?.to_string(),
        },
        "MatMul" => IrOpContract::MatMul {
            lhs: require_input(node, 0)?.to_string(),
            rhs: require_input(node, 1)?.to_string(),
        },
        "Transpose" => IrOpContract::Transpose {
            input: require_input(node, 0)?.to_string(),
        },
        "Relu" => IrOpContract::Relu {
            input: require_input(node, 0)?.to_string(),
        },
        "Sigmoid" => IrOpContract::Sigmoid {
            input: require_input(node, 0)?.to_string(),
        },
        "Gelu" => {
            let approximation = attribute_string(node, "approximate")
                .or_else(|| attribute_string(node, "approximation"));
            if let Some(mode) = approximation {
                if mode == "tanh" {
                    IrOpContract::Gelu {
                        input: require_input(node, 0)?.to_string(),
                    }
                } else if mode == "none" {
                    IrOpContract::GeluExact {
                        input: require_input(node, 0)?.to_string(),
                    }
                } else {
                    return Err(InteropError::new(format!(
                        "node '{}' Gelu approximation '{}' is unsupported; supported: tanh|none",
                        node.name, mode
                    )));
                }
            } else {
                IrOpContract::Gelu {
                    input: require_input(node, 0)?.to_string(),
                }
            }
        }
        "Log" => IrOpContract::Log {
            input: require_input(node, 0)?.to_string(),
        },
        "Exp" => IrOpContract::Exp {
            input: require_input(node, 0)?.to_string(),
        },
        "Softmax" => IrOpContract::Softmax {
            input: require_input(node, 0)?.to_string(),
        },
        "ReduceSum" => {
            let input = require_input(node, 0)?.to_string();
            let input_rank = value_ranks.get(&input).copied();
            let keepdims = attribute_i64(node, "keepdims").unwrap_or(1);
            if keepdims != 0 && keepdims != 1 {
                return Err(InteropError::new(format!(
                    "node '{}' ReduceSum keepdims must be 0 or 1, got {}",
                    node.name, keepdims
                )));
            }
            let axis = parse_reduce_axis(node, initializers, "ReduceSum", input_rank)?;
            IrOpContract::ReduceSum {
                input,
                axis,
                keepdims: keepdims == 1,
            }
        }
        "ReduceMax" => {
            let input = require_input(node, 0)?.to_string();
            let input_rank = value_ranks.get(&input).copied();
            let keepdims = attribute_i64(node, "keepdims").unwrap_or(1);
            if keepdims != 0 && keepdims != 1 {
                return Err(InteropError::new(format!(
                    "node '{}' ReduceMax keepdims must be 0 or 1, got {}",
                    node.name, keepdims
                )));
            }
            let axis = parse_reduce_axis(node, initializers, "ReduceMax", input_rank)?;
            IrOpContract::ReduceMax {
                input,
                axis,
                keepdims: keepdims == 1,
            }
        }
        "ReduceMean" => {
            let input = require_input(node, 0)?.to_string();
            let input_rank = value_ranks.get(&input).copied();
            let keepdims = attribute_i64(node, "keepdims").unwrap_or(1);
            if keepdims != 0 && keepdims != 1 {
                return Err(InteropError::new(format!(
                    "node '{}' ReduceMean keepdims must be 0 or 1, got {}",
                    node.name, keepdims
                )));
            }
            let axis = parse_reduce_axis(node, initializers, "ReduceMean", input_rank)?;
            IrOpContract::ReduceMean {
                input,
                axis,
                keepdims: keepdims == 1,
            }
        }
        "Reshape" => {
            let input = require_input(node, 0)?.to_string();
            let shape_tensor = require_initializer_input(node, 1, initializers)?;
            let shape = parse_usize_list(shape_tensor, "reshape target shape")?;
            IrOpContract::Reshape { input, shape }
        }
        "Concat" => {
            let inputs = node
                .input
                .iter()
                .filter(|name| !name.is_empty())
                .cloned()
                .collect::<Vec<_>>();
            if inputs.len() < 2 {
                return Err(InteropError::new(format!(
                    "node '{}' concat requires at least 2 inputs",
                    node.name
                )));
            }
            let axis = attribute_i64(node, "axis").unwrap_or(0);
            if axis < 0 {
                return Err(InteropError::new(format!(
                    "node '{}' concat axis must be non-negative for static import",
                    node.name
                )));
            }
            IrOpContract::Concat {
                inputs,
                axis: axis as usize,
            }
        }
        "Gather" => {
            let input = require_input(node, 0)?.to_string();
            let indices_tensor = require_initializer_input(node, 1, initializers)?;
            let indices = parse_usize_list(indices_tensor, "gather indices")?;
            let axis = attribute_i64(node, "axis").unwrap_or(0);
            if axis < 0 {
                return Err(InteropError::new(format!(
                    "node '{}' gather axis must be non-negative for static import",
                    node.name
                )));
            }
            IrOpContract::Gather {
                input,
                indices,
                axis: axis as usize,
            }
        }
        "Slice" => {
            let input = require_input(node, 0)?.to_string();
            let starts = parse_usize_list(
                require_initializer_input(node, 1, initializers)?,
                "slice starts",
            )?;
            let ends = parse_usize_list(
                require_initializer_input(node, 2, initializers)?,
                "slice ends",
            )?;
            let axes = if let Some(axis_name) = node.input.get(3).filter(|name| !name.is_empty()) {
                let axis_tensor = initializers.get(axis_name.as_str()).copied().ok_or_else(|| {
                    InteropError::new(format!(
                        "node '{}' slice axes input '{}' must be initializer for Wave 2 static import",
                        node.op_type, axis_name
                    ))
                })?;
                parse_usize_list(axis_tensor, "slice axes")?
            } else {
                (0..starts.len()).collect()
            };

            if let Some(step_name) = node.input.get(4).filter(|name| !name.is_empty()) {
                let step_tensor = initializers.get(step_name.as_str()).copied().ok_or_else(|| {
                    InteropError::new(format!(
                        "node '{}' slice steps input '{}' must be initializer for Wave 2 static import",
                        node.op_type, step_name
                    ))
                })?;
                let steps = parse_usize_list(step_tensor, "slice steps")?;
                if steps.iter().any(|step| *step != 1) {
                    return Err(InteropError::new(format!(
                        "node '{}' slice only supports step=1 in Wave 2 static import",
                        node.name
                    )));
                }
            }

            IrOpContract::Slice {
                input,
                starts,
                ends,
                axes,
            }
        }
        // Gemm: alpha * (A @ B) + beta * C
        // Spec: https://onnx.ai/onnx/operators/onnx__Gemm.html
        "Gemm" => {
            let lhs = require_input(node, 0)?.to_string();
            let rhs = require_input(node, 1)?.to_string();
            // Input[2] (bias) is optional — empty string means absent.
            let bias = node.input.get(2).filter(|name| !name.is_empty()).cloned();
            // Read float attributes, falling back to the ONNX-specified defaults.
            let alpha = attribute_f32(node, "alpha").unwrap_or(1.0_f32);
            let beta = attribute_f32(node, "beta").unwrap_or(1.0_f32);
            let trans_a_raw = attribute_i64(node, "transA").unwrap_or(0);
            let trans_b_raw = attribute_i64(node, "transB").unwrap_or(0);
            if !alpha.is_finite() {
                return Err(InteropError::new(format!(
                    "Gemm node '{}' alpha attribute must be finite, got {alpha}",
                    node.name
                )));
            }
            if !beta.is_finite() {
                return Err(InteropError::new(format!(
                    "Gemm node '{}' beta attribute must be finite, got {beta}",
                    node.name
                )));
            }
            if trans_a_raw != 0 && trans_a_raw != 1 {
                return Err(InteropError::new(format!(
                    "Gemm node '{}' transA must be 0 or 1, got {}",
                    node.name, trans_a_raw
                )));
            }
            if trans_b_raw != 0 && trans_b_raw != 1 {
                return Err(InteropError::new(format!(
                    "Gemm node '{}' transB must be 0 or 1, got {}",
                    node.name, trans_b_raw
                )));
            }
            IrOpContract::Gemm {
                lhs,
                rhs,
                bias,
                alpha,
                beta,
                trans_a: trans_a_raw == 1,
                trans_b: trans_b_raw == 1,
            }
        }
        unsupported => {
            return Err(InteropError::new(format!(
                "unsupported ONNX op '{unsupported}' in Wave 2 importer"
            )));
        }
    })
}

fn parse_initializer_literal(tensor: &pb::TensorProto) -> Result<IrTensorLiteral, InteropError> {
    let shape = tensor_dims(tensor)?;
    let data = parse_tensor_data_as_f32(tensor)?;
    let expected = shape.iter().copied().product::<usize>();
    if expected != data.len() {
        return Err(InteropError::new(format!(
            "initializer '{}' data len mismatch: expected {}, got {}",
            tensor.name,
            expected,
            data.len()
        )));
    }
    Ok(IrTensorLiteral { shape, data })
}

fn tensor_spec_from_initializer(tensor: &pb::TensorProto) -> Result<IrTensorSpec, InteropError> {
    Ok(IrTensorSpec {
        name: tensor.name.clone(),
        shape: tensor_dims(tensor)?,
        dtype: map_onnx_dtype(tensor.data_type)?,
    })
}

fn tensor_spec_from_value_info(value: &pb::ValueInfoProto) -> Result<IrTensorSpec, InteropError> {
    let dtype_and_shape = value
        .r#type
        .as_ref()
        .and_then(|kind| kind.value.as_ref())
        .map(|kind| match kind {
            pb::type_proto::Value::TensorType(tensor) => tensor,
        })
        .ok_or_else(|| {
            InteropError::new(format!(
                "value '{}' is missing TensorType metadata",
                value.name
            ))
        })?;

    let shape_proto = dtype_and_shape.shape.as_ref().ok_or_else(|| {
        InteropError::new(format!("value '{}' is missing shape metadata", value.name))
    })?;
    let shape = shape_proto
        .dim
        .iter()
        .map(|dim| match dim.value.as_ref() {
            Some(pb::tensor_shape_proto::dimension::Value::DimValue(v)) if *v > 0 => {
                Ok(*v as usize)
            }
            _ => Err(InteropError::new(format!(
                "value '{}' contains non-static or invalid dimension",
                value.name
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(IrTensorSpec {
        name: value.name.clone(),
        shape,
        dtype: map_onnx_dtype(dtype_and_shape.elem_type)?,
    })
}

fn tensor_dims(tensor: &pb::TensorProto) -> Result<Vec<usize>, InteropError> {
    tensor
        .dims
        .iter()
        .map(|dim| {
            if *dim <= 0 {
                Err(InteropError::new(format!(
                    "tensor '{}' has invalid non-positive dim {}",
                    tensor.name, dim
                )))
            } else {
                Ok(*dim as usize)
            }
        })
        .collect()
}

fn map_onnx_dtype(dtype: i32) -> Result<IrDataType, InteropError> {
    match pb::tensor_proto::DataType::from_i32(dtype)
        .ok_or_else(|| InteropError::new(format!("unknown ONNX dtype value {dtype}")))?
    {
        pb::tensor_proto::DataType::Float | pb::tensor_proto::DataType::Double => {
            Ok(IrDataType::F32)
        }
        pb::tensor_proto::DataType::Int64 | pb::tensor_proto::DataType::Int32 => {
            Ok(IrDataType::I64)
        }
        unsupported => Err(InteropError::new(format!(
            "unsupported ONNX dtype {:?} in Wave 1 importer",
            unsupported
        ))),
    }
}

fn parse_tensor_data_as_i64(tensor: &pb::TensorProto) -> Result<Vec<i64>, InteropError> {
    let dtype = pb::tensor_proto::DataType::from_i32(tensor.data_type)
        .ok_or_else(|| InteropError::new(format!("unknown ONNX dtype {}", tensor.data_type)))?;

    if !tensor.raw_data.is_empty() {
        return parse_raw_tensor_data_as_i64(tensor, dtype);
    }

    match dtype {
        pb::tensor_proto::DataType::Int64 => Ok(tensor.int64_data.clone()),
        pb::tensor_proto::DataType::Int32 => {
            Ok(tensor.int32_data.iter().map(|v| i64::from(*v)).collect())
        }
        pb::tensor_proto::DataType::Float => tensor
            .float_data
            .iter()
            .map(|value| {
                if value.fract() == 0.0 {
                    Ok(*value as i64)
                } else {
                    Err(InteropError::new(format!(
                        "initializer '{}' contains non-integer float {} where integer is required",
                        tensor.name, value
                    )))
                }
            })
            .collect(),
        pb::tensor_proto::DataType::Double => tensor
            .double_data
            .iter()
            .map(|value| {
                if value.fract() == 0.0 {
                    Ok(*value as i64)
                } else {
                    Err(InteropError::new(format!(
                        "initializer '{}' contains non-integer double {} where integer is required",
                        tensor.name, value
                    )))
                }
            })
            .collect(),
        unsupported => Err(InteropError::new(format!(
            "unsupported ONNX initializer dtype {:?} for integer parsing",
            unsupported
        ))),
    }
}

fn parse_raw_tensor_data_as_i64(
    tensor: &pb::TensorProto,
    dtype: pb::tensor_proto::DataType,
) -> Result<Vec<i64>, InteropError> {
    let bytes = tensor.raw_data.as_slice();
    let parse_err = |msg: &str| {
        InteropError::new(format!(
            "initializer '{}' has invalid raw_data: {}",
            tensor.name, msg
        ))
    };

    match dtype {
        pb::tensor_proto::DataType::Int64 => {
            if !bytes.len().is_multiple_of(8) {
                return Err(parse_err("int64 raw bytes are not multiple of 8"));
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect())
        }
        pb::tensor_proto::DataType::Int32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(parse_err("int32 raw bytes are not multiple of 4"));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| {
                    i64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                })
                .collect())
        }
        pb::tensor_proto::DataType::Float => {
            if !bytes.len().is_multiple_of(4) {
                return Err(parse_err("float raw bytes are not multiple of 4"));
            }
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if value.fract() != 0.0 {
                    return Err(parse_err("float raw_data contains non-integer value"));
                }
                out.push(value as i64);
            }
            Ok(out)
        }
        pb::tensor_proto::DataType::Double => {
            if !bytes.len().is_multiple_of(8) {
                return Err(parse_err("double raw bytes are not multiple of 8"));
            }
            let mut out = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.chunks_exact(8) {
                let value = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                if value.fract() != 0.0 {
                    return Err(parse_err("double raw_data contains non-integer value"));
                }
                out.push(value as i64);
            }
            Ok(out)
        }
        unsupported => Err(InteropError::new(format!(
            "unsupported ONNX raw_data dtype {:?} for integer parsing",
            unsupported
        ))),
    }
}

fn parse_tensor_data_as_f32(tensor: &pb::TensorProto) -> Result<Vec<f32>, InteropError> {
    let dtype = pb::tensor_proto::DataType::from_i32(tensor.data_type)
        .ok_or_else(|| InteropError::new(format!("unknown ONNX dtype {}", tensor.data_type)))?;

    if !tensor.raw_data.is_empty() {
        return parse_raw_tensor_data_as_f32(tensor, dtype);
    }

    match dtype {
        pb::tensor_proto::DataType::Float => Ok(tensor.float_data.clone()),
        pb::tensor_proto::DataType::Double => {
            Ok(tensor.double_data.iter().map(|v| *v as f32).collect())
        }
        pb::tensor_proto::DataType::Int64 => {
            Ok(tensor.int64_data.iter().map(|v| *v as f32).collect())
        }
        pb::tensor_proto::DataType::Int32 => {
            Ok(tensor.int32_data.iter().map(|v| *v as f32).collect())
        }
        unsupported => Err(InteropError::new(format!(
            "unsupported ONNX initializer dtype {:?}",
            unsupported
        ))),
    }
}

fn parse_raw_tensor_data_as_f32(
    tensor: &pb::TensorProto,
    dtype: pb::tensor_proto::DataType,
) -> Result<Vec<f32>, InteropError> {
    let bytes = tensor.raw_data.as_slice();
    let parse_err = |msg: &str| {
        InteropError::new(format!(
            "initializer '{}' has invalid raw_data: {}",
            tensor.name, msg
        ))
    };
    match dtype {
        pb::tensor_proto::DataType::Float => {
            if !bytes.len().is_multiple_of(4) {
                return Err(parse_err("float raw bytes are not multiple of 4"));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        pb::tensor_proto::DataType::Double => {
            if !bytes.len().is_multiple_of(8) {
                return Err(parse_err("double raw bytes are not multiple of 8"));
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|chunk| {
                    f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]) as f32
                })
                .collect())
        }
        pb::tensor_proto::DataType::Int64 => {
            if !bytes.len().is_multiple_of(8) {
                return Err(parse_err("int64 raw bytes are not multiple of 8"));
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]) as f32
                })
                .collect())
        }
        pb::tensor_proto::DataType::Int32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(parse_err("int32 raw bytes are not multiple of 4"));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
                .collect())
        }
        unsupported => Err(InteropError::new(format!(
            "unsupported ONNX raw_data dtype {:?}",
            unsupported
        ))),
    }
}

fn op_kind(op: &OnnxOpStub) -> &'static str {
    match op {
        OnnxOpStub::Input { .. } => "Input",
        OnnxOpStub::Parameter { .. } => "Parameter",
        OnnxOpStub::ConstTensor { .. } => "ConstTensor",
        OnnxOpStub::Add { .. } => "Add",
        OnnxOpStub::Sub { .. } => "Sub",
        OnnxOpStub::Mul { .. } => "Mul",
        OnnxOpStub::Div { .. } => "Div",
        OnnxOpStub::Neg { .. } => "Neg",
        OnnxOpStub::MatMul { .. } => "MatMul",
        OnnxOpStub::Transpose { .. } => "Transpose",
        OnnxOpStub::Relu { .. } => "Relu",
        OnnxOpStub::Sigmoid { .. } => "Sigmoid",
        OnnxOpStub::Gelu { .. } => "Gelu",
        OnnxOpStub::ReduceSum { .. } => "ReduceSum",
        OnnxOpStub::ReduceMax { .. } => "ReduceMax",
        OnnxOpStub::ReduceMean { .. } => "ReduceMean",
        OnnxOpStub::Log { .. } => "Log",
        OnnxOpStub::Exp { .. } => "Exp",
        OnnxOpStub::Softmax { .. } => "Softmax",
        OnnxOpStub::Reshape { .. } => "Reshape",
        OnnxOpStub::Concat { .. } => "Concat",
        OnnxOpStub::Gather { .. } => "Gather",
        OnnxOpStub::Slice { .. } => "Slice",
        OnnxOpStub::Gemm { .. } => "Gemm",
        OnnxOpStub::Output { .. } => "Output",
    }
}

fn op_kind_from_contract(op: &IrOpContract) -> &'static str {
    match op {
        IrOpContract::Input { .. } => "Input",
        IrOpContract::Parameter { .. } => "Parameter",
        IrOpContract::ConstTensor(_) => "ConstTensor",
        IrOpContract::Add { .. } => "Add",
        IrOpContract::Sub { .. } => "Sub",
        IrOpContract::Mul { .. } => "Mul",
        IrOpContract::Div { .. } => "Div",
        IrOpContract::Neg { .. } => "Neg",
        IrOpContract::MatMul { .. } => "MatMul",
        IrOpContract::Transpose { .. } => "Transpose",
        IrOpContract::Relu { .. } => "Relu",
        IrOpContract::Softmax { .. } => "Softmax",
        IrOpContract::Log { .. } => "Log",
        IrOpContract::Exp { .. } => "Exp",
        IrOpContract::Sigmoid { .. } => "Sigmoid",
        IrOpContract::Gelu { .. } => "Gelu",
        IrOpContract::GeluExact { .. } => "GeluExact",
        IrOpContract::ReduceSum { .. } => "ReduceSum",
        IrOpContract::ReduceMax { .. } => "ReduceMax",
        IrOpContract::ReduceMean { .. } => "ReduceMean",
        IrOpContract::Gemm { .. } => "Gemm",
        IrOpContract::Reshape { .. } => "Reshape",
        IrOpContract::Concat { .. } => "Concat",
        IrOpContract::Gather { .. } => "Gather",
        IrOpContract::Slice { .. } => "Slice",
        IrOpContract::Output { .. } => "Output",
    }
}

fn inferred_output_rank(op: &IrOpContract, value_ranks: &HashMap<String, usize>) -> Option<usize> {
    match op {
        IrOpContract::Input { .. }
        | IrOpContract::Parameter { .. }
        | IrOpContract::ConstTensor(_)
        | IrOpContract::Output { .. } => None,
        IrOpContract::Add { lhs, .. }
        | IrOpContract::Sub { lhs, .. }
        | IrOpContract::Mul { lhs, .. }
        | IrOpContract::Div { lhs, .. }
        | IrOpContract::Neg { input: lhs }
        | IrOpContract::Relu { input: lhs }
        | IrOpContract::Softmax { input: lhs }
        | IrOpContract::Log { input: lhs }
        | IrOpContract::Exp { input: lhs }
        | IrOpContract::Sigmoid { input: lhs }
        | IrOpContract::Gelu { input: lhs }
        | IrOpContract::GeluExact { input: lhs }
        | IrOpContract::Transpose { input: lhs } => value_ranks.get(lhs).copied(),
        IrOpContract::MatMul { lhs, rhs } | IrOpContract::Gemm { lhs, rhs, .. } => {
            match (value_ranks.get(lhs), value_ranks.get(rhs)) {
                (Some(2), Some(2)) => Some(2),
                _ => None,
            }
        }
        IrOpContract::ReduceSum {
            input,
            axis,
            keepdims,
        }
        | IrOpContract::ReduceMax {
            input,
            axis,
            keepdims,
        }
        | IrOpContract::ReduceMean {
            input,
            axis,
            keepdims,
        } => value_ranks.get(input).copied().map(|rank| {
            if *keepdims {
                if axis.is_some() { rank } else { rank.max(1) }
            } else if axis.is_some() {
                rank.saturating_sub(1)
            } else {
                1
            }
        }),
        IrOpContract::Reshape { shape, .. } => Some(shape.len()),
        IrOpContract::Concat { inputs, .. } => inputs
            .first()
            .and_then(|name| value_ranks.get(name).copied()),
        IrOpContract::Gather { input, .. } => value_ranks.get(input).copied(),
        IrOpContract::Slice { input, .. } => value_ranks.get(input).copied(),
    }
}
