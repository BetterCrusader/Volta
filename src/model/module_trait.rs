use crate::ir::ValueId;

use crate::model::{ModelBuildError, ModelBuilder, TensorShape};

pub trait Module {
    fn output_shape(&self, input: &TensorShape) -> Result<TensorShape, ModelBuildError>;
    fn build(
        &self,
        builder: &mut ModelBuilder,
        input_value: ValueId,
        input_shape: &TensorShape,
    ) -> Result<(ValueId, TensorShape), ModelBuildError>;
}
