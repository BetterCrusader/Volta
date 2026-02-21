use crate::interop::contract::{
    ImportedProgram, IrContractVersion, IrDataType, IrGraphContract, IrNodeContract, IrOpContract,
    IrTensorLiteral, IrTensorSpec,
};
use crate::interop::{InteropError, PluginRegistry};

#[derive(Debug, Clone, PartialEq)]
pub enum OnnxOpStub {
    Input { name: String, shape: Vec<usize> },
    Parameter { name: String, shape: Vec<usize> },
    ConstTensor { shape: Vec<usize>, data: Vec<f32> },
    Add { lhs: String, rhs: String },
    Sub { lhs: String, rhs: String },
    Mul { lhs: String, rhs: String },
    Div { lhs: String, rhs: String },
    MatMul { lhs: String, rhs: String },
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
                OnnxOpStub::MatMul { lhs, rhs } => IrOpContract::MatMul {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
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
}

pub fn import_onnx_stub_graph(source: &OnnxGraphStub) -> Result<ImportedProgram, InteropError> {
    OnnxImporter::default().import_graph(source)
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
        OnnxOpStub::MatMul { .. } => "MatMul",
        OnnxOpStub::Relu { .. } => "Relu",
        OnnxOpStub::Softmax { .. } => "Softmax",
        OnnxOpStub::Output { .. } => "Output",
    }
}
