use crate::interop::contract::{
    ImportedProgram, IrContractVersion, IrDataType, IrGraphContract, IrNodeContract, IrOpContract,
    IrTensorLiteral, IrTensorSpec,
};
use crate::interop::{InteropError, PluginRegistry};
use prost::Message;
use std::collections::HashSet;
use std::path::Path;
use tract_onnx::pb;

#[derive(Debug, Clone, PartialEq)]
pub enum OnnxOpStub {
    Input { name: String, shape: Vec<usize> },
    Parameter { name: String, shape: Vec<usize> },
    ConstTensor { shape: Vec<usize>, data: Vec<f32> },
    Add { lhs: String, rhs: String },
    Sub { lhs: String, rhs: String },
    Mul { lhs: String, rhs: String },
    Div { lhs: String, rhs: String },
    Neg { input: String },
    MatMul { lhs: String, rhs: String },
    Transpose { input: String },
    Relu { input: String },
    Softmax { input: String },
    Output { value: String },
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

        let initializer_names: HashSet<&str> = graph
            .initializer
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();

        for value in &graph.input {
            if initializer_names.contains(value.name.as_str()) {
                continue;
            }
            let mut spec = tensor_spec_from_value_info(value)?;
            spec.name = value.name.clone();
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

            let op = map_proto_op(proto_node)?;
            let node = IrNodeContract {
                id: id.to_string(),
                op,
            };
            if let Some(plugin) = self.plugins.find(op_kind_from_contract(&node.op)) {
                plugin.validate_node(&node)?;
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

pub fn import_onnx_bytes(bytes: &[u8]) -> Result<ImportedProgram, InteropError> {
    let model = pb::ModelProto::decode(bytes)
        .map_err(|err| InteropError::new(format!("failed to decode ONNX protobuf: {err}")))?;
    OnnxImporter::default().import_model_proto(&model)
}

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

fn map_proto_op(node: &pb::NodeProto) -> Result<IrOpContract, InteropError> {
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
        "Softmax" => IrOpContract::Softmax {
            input: require_input(node, 0)?.to_string(),
        },
        unsupported => {
            return Err(InteropError::new(format!(
                "unsupported ONNX op '{unsupported}' in Wave 1 importer"
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
        OnnxOpStub::Softmax { .. } => "Softmax",
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
        IrOpContract::Output { .. } => "Output",
    }
}
