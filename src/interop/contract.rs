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
    Input { name: String },
    Parameter { name: String },
    ConstTensor(IrTensorLiteral),
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

        Ok(())
    }

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
                IrOpContract::Output { value } => {
                    let value_id = resolve_value(&ids, value)?;
                    let (_, output_id) = graph
                        .add_op(block, Op::Output(value_id))
                        .map_err(|err| InteropError::new(err.message))?;
                    output = Some(output_id);
                    output_id
                }
            };
            ids.insert(node.id.clone(), value);
        }

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
        | IrOpContract::Output { value: input } => vec![input.as_str()],
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
