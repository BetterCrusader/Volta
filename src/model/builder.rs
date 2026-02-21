use std::collections::{HashMap, HashSet};

use crate::ir::{
    ExecutionPlan, Graph, NodeId, Op, Tensor, ValueId, build_execution_plan, verify_graph,
};
use crate::model::{Parameter, TensorShape};

#[derive(Debug, Clone)]
pub struct ModelBuildError {
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct CompiledModel {
    pub graph: Graph,
    pub output: ValueId,
    pub output_shape: TensorShape,
    pub loss: Option<ValueId>,
    pub parameters: HashMap<String, Tensor>,
    pub parameter_values: HashMap<String, ValueId>,
    pub inference_plan: ExecutionPlan,
    pub inference_ordered_nodes: Vec<NodeId>,
}

#[derive(Debug, Clone)]
pub struct ModelBuilder {
    graph: Graph,
    block: crate::ir::BasicBlockId,
    parameters: HashMap<String, Tensor>,
    parameter_values: HashMap<String, ValueId>,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelBuilder {
    pub fn new() -> Self {
        let mut graph = Graph::new();
        let block = graph.create_block();
        Self {
            graph,
            block,
            parameters: HashMap::new(),
            parameter_values: HashMap::new(),
        }
    }

    pub fn input(&mut self, name: &str) -> Result<ValueId, ModelBuildError> {
        let (_, value) = self
            .graph
            .add_op(self.block, Op::Input(name.to_string()))
            .map_err(|err| ModelBuildError {
                message: format!("Failed to add input '{name}': {}", err.message),
            })?;
        Ok(value)
    }

    pub fn input_with_shape(
        &mut self,
        name: &str,
        shape: Vec<usize>,
    ) -> Result<ValueId, ModelBuildError> {
        let value = self.input(name)?;
        self.graph.bind_input_shape(name, shape);
        Ok(value)
    }

    pub fn add_parameter(&mut self, parameter: Parameter) -> Result<ValueId, ModelBuildError> {
        let name = parameter.name.clone();
        if self.parameter_values.contains_key(&name) {
            return Err(ModelBuildError {
                message: format!("Duplicate parameter name: '{name}'"),
            });
        }

        let (_, value) = self
            .graph
            .add_op(self.block, Op::Parameter(name.clone()))
            .map_err(|err| ModelBuildError {
                message: format!("Failed to add parameter '{name}': {}", err.message),
            })?;

        self.graph
            .bind_parameter_shape(&name, parameter.tensor.shape.clone());

        if parameter.trainable {
            self.parameters.insert(name.clone(), parameter.tensor);
            self.parameter_values.insert(name, value);
        }

        Ok(value)
    }

    pub fn add_op(&mut self, op: Op) -> Result<ValueId, ModelBuildError> {
        let (_, value) = self
            .graph
            .add_op(self.block, op)
            .map_err(|err| ModelBuildError {
                message: format!("Failed to add op: {}", err.message),
            })?;
        Ok(value)
    }

    pub fn finalize(
        self,
        output: ValueId,
        output_shape: TensorShape,
        loss: Option<ValueId>,
    ) -> Result<CompiledModel, ModelBuildError> {
        verify_graph(&self.graph).map_err(|err| ModelBuildError {
            message: format!("Model graph verification failed: {}", err.message),
        })?;
        let inference_plan =
            build_execution_plan(&self.graph, &HashSet::new()).map_err(|err| ModelBuildError {
                message: format!(
                    "Infer plan build failed during model finalize: {}",
                    err.message
                ),
            })?;
        let inference_ordered_nodes =
            dependency_ordered_nodes(&self.graph, output, &inference_plan.schedule.ordered_nodes)?;

        Ok(CompiledModel {
            graph: self.graph,
            output,
            output_shape,
            loss,
            parameters: self.parameters,
            parameter_values: self.parameter_values,
            inference_plan,
            inference_ordered_nodes,
        })
    }
}

fn dependency_ordered_nodes(
    graph: &Graph,
    target: ValueId,
    ordered_nodes: &[NodeId],
) -> Result<Vec<NodeId>, ModelBuildError> {
    if target.0 >= graph.nodes.len() {
        return Err(ModelBuildError {
            message: format!("Infer target value out of range: {}", target.0),
        });
    }

    let mut required_values = HashSet::<ValueId>::new();
    let mut stack = vec![target];
    while let Some(value) = stack.pop() {
        if !required_values.insert(value) {
            continue;
        }
        let node = graph.nodes.get(value.0).ok_or_else(|| ModelBuildError {
            message: format!("Infer dependency value out of range: {}", value.0),
        })?;
        for input in node.op.input_values() {
            stack.push(input);
        }
    }

    let mut filtered = Vec::new();
    for node_id in ordered_nodes {
        let node = graph.nodes.get(node_id.0).ok_or_else(|| ModelBuildError {
            message: format!("Infer schedule node out of range: {}", node_id.0),
        })?;
        if required_values.contains(&node.output) {
            filtered.push(*node_id);
        }
    }

    Ok(filtered)
}
