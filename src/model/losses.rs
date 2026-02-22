use crate::ir::{Op, ValueId};

use crate::model::{ModelBuildError, ModelBuilder, TensorShape};

#[derive(Debug, Clone, Copy)]
pub struct MSELoss;

impl MSELoss {
    pub fn build(
        &self,
        builder: &mut ModelBuilder,
        prediction: ValueId,
        prediction_shape: &TensorShape,
        target: ValueId,
        target_shape: &TensorShape,
    ) -> Result<ValueId, ModelBuildError> {
        if prediction_shape != target_shape {
            return Err(ModelBuildError {
                message: format!(
                    "MSELoss shape mismatch: prediction {:?} vs target {:?}",
                    prediction_shape.0, target_shape.0
                ),
            });
        }

        let diff = builder.add_op(Op::Sub(prediction, target))?;
        let sq = builder.add_op(Op::Mul(diff, diff))?;
        builder.add_op(Op::Output(sq))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn build(
        &self,
        builder: &mut ModelBuilder,
        logits: ValueId,
        logits_shape: &TensorShape,
        target: ValueId,
        target_shape: &TensorShape,
    ) -> Result<ValueId, ModelBuildError> {
        if logits_shape != target_shape {
            return Err(ModelBuildError {
                message: format!(
                    "CrossEntropyLoss shape mismatch: logits {:?} vs target {:?}",
                    logits_shape.0, target_shape.0
                ),
            });
        }

        // Real cross-entropy: -sum(target * log(softmax(logits)))
        let probs = builder.add_op(Op::Softmax(logits))?;
        let log_probs = builder.add_op(Op::Log(probs))?;
        let weighted = builder.add_op(Op::Mul(target, log_probs))?;
        let sum = builder.add_op(Op::ReduceSum {
            input: weighted,
            axis: None,
        })?;
        let neg = builder.add_op(Op::Neg(sum))?;
        builder.add_op(Op::Output(neg))
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::Tensor;
    use crate::model::{CrossEntropyLoss, ModelBuilder, TensorShape};

    #[test]
    fn cross_entropy_builds_with_equal_shapes() {
        let mut builder = ModelBuilder::new();
        let logits = builder.input("logits").expect("input should build");
        let target = builder.input("target").expect("input should build");

        let loss = CrossEntropyLoss
            .build(
                &mut builder,
                logits,
                &TensorShape(vec![3]),
                target,
                &TensorShape(vec![3]),
            )
            .expect("loss should build");

        let _model = builder
            .finalize(logits, TensorShape(vec![3]), Some(loss))
            .expect("finalize should pass");

        let _ = Tensor::new(vec![1], vec![1.0]).expect("tensor");
    }
}
