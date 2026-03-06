use crate::ir::Tensor;

use crate::model::TensorShape;

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub tensor: Tensor,
    pub shape: TensorShape,
    pub trainable: bool,
}

impl Parameter {
    pub fn new(name: impl Into<String>, tensor: Tensor, trainable: bool) -> Self {
        Self {
            name: name.into(),
            shape: TensorShape(tensor.shape.clone()),
            tensor,
            trainable,
        }
    }
}
