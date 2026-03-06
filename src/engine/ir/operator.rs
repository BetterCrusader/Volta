use crate::ir::{ExecutionContext, InterpreterError, Op, RuntimeValue, ShapeFact, ValueId};

pub trait Operator: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &str;

    fn fingerprint(&self, hasher: &mut dyn std::hash::Hasher);

    fn execute(
        &self,
        inputs: &[RuntimeValue],
        context: &ExecutionContext,
    ) -> Result<RuntimeValue, InterpreterError>;

    fn infer_shape(&self, input_shapes: &[ShapeFact]) -> Result<ShapeFact, String>;

    fn check_constraints(&self, _input_shapes: &[ShapeFact]) -> Result<(), String> {
        Ok(())
    }

    // For autograd
    fn get_backward_ops(
        &self,
        input_values: &[ValueId],
        output_value: ValueId,
        upstream_grads: &[ValueId],
    ) -> Vec<(ValueId, Op)>;

    /// Executes the operation on the CUDA backend.
    /// Returns `Err` if CUDA is not supported by this operator or execution fails.
    fn execute_cuda(
        &self,
        _inputs: &[crate::ir::RuntimeValue],
        _context: &crate::ir::ExecutionContext,
    ) -> Result<crate::ir::RuntimeValue, crate::ir::InterpreterError> {
        Err(crate::ir::InterpreterError {
            message: format!("Operator '{}' does not support CUDA", self.name()),
            node: None,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct AddOperator;

impl Operator for AddOperator {
    fn name(&self) -> &str {
        "Add"
    }

    fn check_constraints(&self, input_shapes: &[ShapeFact]) -> Result<(), String> {
        if input_shapes.len() != 2 {
            return Err("Add requires exactly 2 inputs".to_string());
        }
        Ok(())
    }

    fn fingerprint(&self, hasher: &mut dyn std::hash::Hasher) {
        hasher.write(b"Add");
    }

    fn execute(
        &self,
        inputs: &[RuntimeValue],
        _context: &ExecutionContext,
    ) -> Result<RuntimeValue, InterpreterError> {
        if inputs.len() != 2 {
            return Err(InterpreterError {
                message: "Add requires 2 inputs".to_string(),
                node: None,
            });
        }
        // Simplified for PoC, real impl should handle broadcasting and types
        match (&inputs[0], &inputs[1]) {
            (RuntimeValue::Float(a), RuntimeValue::Float(b)) => Ok(RuntimeValue::Float(a + b)),
            (RuntimeValue::Int(a), RuntimeValue::Int(b)) => Ok(RuntimeValue::Int(a + b)),
            _ => Err(InterpreterError {
                message: "Type mismatch in Add".to_string(),
                node: None,
            }),
        }
    }

    fn infer_shape(&self, input_shapes: &[ShapeFact]) -> Result<ShapeFact, String> {
        if input_shapes.len() != 2 {
            return Err("Add requires 2 inputs".to_string());
        }
        // Simplified: assuming same shapes or broadcasting
        Ok(input_shapes[0].clone())
    }

    fn get_backward_ops(
        &self,
        input_values: &[ValueId],
        _output_value: ValueId,
        upstream_grads: &[ValueId],
    ) -> Vec<(ValueId, Op)> {
        // gradient of a+b w.r.t a is 1, w.r.t b is 1.
        // so grads are just upstream_grads[0]
        vec![
            (input_values[0], Op::Identity(upstream_grads[0])),
            (input_values[1], Op::Identity(upstream_grads[0])),
        ]
    }
}
