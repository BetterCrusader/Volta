use crate::ir::{Op, Tensor, ValueId};

use crate::model::{ModelBuildError, ModelBuilder, Module, Parameter, TensorShape};

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    #[must_use]
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn push<M: Module + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer));
    }

    pub fn build(
        &self,
        builder: &mut ModelBuilder,
        input_value: ValueId,
        input_shape: TensorShape,
    ) -> Result<(ValueId, TensorShape), ModelBuildError> {
        let mut current_value = input_value;
        let mut current_shape = input_shape;
        for layer in &self.layers {
            let (next_value, next_shape) = layer.build(builder, current_value, &current_shape)?;
            current_value = next_value;
            current_shape = next_shape;
        }
        Ok((current_value, current_shape))
    }
}

#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub name: String,
    pub weight: Tensor,
}

impl Module for LinearLayer {
    fn output_shape(&self, input: &TensorShape) -> Result<TensorShape, ModelBuildError> {
        let weight_shape = TensorShape(self.weight.shape.clone());
        input
            .matmul_output(&weight_shape)
            .ok_or_else(|| ModelBuildError {
                message: format!(
                    "Linear shape mismatch: input {:?}, weight {:?}",
                    input.0, weight_shape.0
                ),
            })
    }

    fn build(
        &self,
        builder: &mut ModelBuilder,
        input_value: ValueId,
        input_shape: &TensorShape,
    ) -> Result<(ValueId, TensorShape), ModelBuildError> {
        let out_shape = self.output_shape(input_shape)?;
        let weight_value = builder.add_parameter(Parameter::new(
            format!("{}.weight", self.name),
            self.weight.clone(),
            true,
        ))?;
        let out = builder.add_op(Op::MatMul(input_value, weight_value))?;
        Ok((out, out_shape))
    }
}

#[derive(Debug, Clone)]
pub struct Conv2DLayer {
    pub name: String,
    pub kernel: Tensor,
}

impl Module for Conv2DLayer {
    fn output_shape(&self, input: &TensorShape) -> Result<TensorShape, ModelBuildError> {
        let kernel_shape = TensorShape(self.kernel.shape.clone());
        input
            .conv2d_output(&kernel_shape)
            .ok_or_else(|| ModelBuildError {
                message: format!(
                    "Conv2D shape mismatch: input {:?}, kernel {:?}",
                    input.0, kernel_shape.0
                ),
            })
    }

    fn build(
        &self,
        builder: &mut ModelBuilder,
        input_value: ValueId,
        input_shape: &TensorShape,
    ) -> Result<(ValueId, TensorShape), ModelBuildError> {
        let out_shape = self.output_shape(input_shape)?;
        let kernel_value = builder.add_parameter(Parameter::new(
            format!("{}.kernel", self.name),
            self.kernel.clone(),
            true,
        ))?;
        let out = builder.add_op(Op::Conv2D(input_value, kernel_value))?;
        Ok((out, out_shape))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReLULayer;

impl Module for ReLULayer {
    fn output_shape(&self, input: &TensorShape) -> Result<TensorShape, ModelBuildError> {
        Ok(input.clone())
    }

    fn build(
        &self,
        builder: &mut ModelBuilder,
        input_value: ValueId,
        input_shape: &TensorShape,
    ) -> Result<(ValueId, TensorShape), ModelBuildError> {
        let out = builder.add_op(Op::Relu(input_value))?;
        Ok((out, input_shape.clone()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SoftmaxLayer;

impl Module for SoftmaxLayer {
    fn output_shape(&self, input: &TensorShape) -> Result<TensorShape, ModelBuildError> {
        if input.rank() != 1 {
            return Err(ModelBuildError {
                message: format!("Softmax expects rank-1 shape, got {:?}", input.0),
            });
        }
        Ok(input.clone())
    }

    fn build(
        &self,
        builder: &mut ModelBuilder,
        input_value: ValueId,
        input_shape: &TensorShape,
    ) -> Result<(ValueId, TensorShape), ModelBuildError> {
        let out_shape = self.output_shape(input_shape)?;
        let out = builder.add_op(Op::Softmax(input_value))?;
        Ok((out, out_shape))
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{Tensor, verify_graph};
    use crate::model::{
        LinearLayer, MSELoss, ModelBuilder, Module, ReLULayer, Sequential, TensorShape,
    };

    #[test]
    fn sequential_builds_deterministic_graph() {
        let mut builder = ModelBuilder::new();
        let input = builder.input("x").expect("input should build");

        let mut seq = Sequential::new();
        seq.push(LinearLayer {
            name: "l1".to_string(),
            weight: Tensor::new(vec![1, 1], vec![0.5]).expect("tensor"),
        });
        seq.push(ReLULayer);

        let (pred, pred_shape) = seq
            .build(&mut builder, input, TensorShape(vec![1, 1]))
            .expect("build should pass");
        let target = builder.input("y").expect("input should build");
        let loss = MSELoss
            .build(
                &mut builder,
                pred,
                &pred_shape,
                target,
                &TensorShape(vec![1, 1]),
            )
            .expect("loss should build");

        let model = builder
            .finalize(pred, pred_shape, Some(loss))
            .expect("model should finalize");
        verify_graph(&model.graph).expect("graph should verify");
    }

    #[test]
    fn linear_rejects_invalid_shape() {
        let layer = LinearLayer {
            name: "bad".to_string(),
            weight: Tensor::new(vec![3, 2], vec![0.0; 6]).expect("tensor"),
        };
        let err = layer
            .output_shape(&TensorShape(vec![4, 4]))
            .expect_err("shape must fail");
        assert!(err.message.contains("shape mismatch"));
    }
}
