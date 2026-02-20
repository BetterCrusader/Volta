use std::collections::HashMap;

use crate::ir::{Graph, Op, Tensor, ValueId, verify_graph};
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

        Ok(CompiledModel {
            graph: self.graph,
            output,
            output_shape,
            loss,
            parameters: self.parameters,
            parameter_values: self.parameter_values,
        })
    }
}
