use std::sync::Arc;
use crate::engine::ir::interpreter::{RuntimeValue, InterpreterError};
use crate::engine::ir::node::NodeId;

pub fn add_values(
    left: RuntimeValue,
    right: RuntimeValue,
    node: NodeId,
) -> Result<RuntimeValue, InterpreterError> {
    match (left, right) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => a
            .checked_add(b)
            .map(RuntimeValue::Int)
            .ok_or_else(|| error("Integer overflow in add".to_string(), node)),
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
            finite_float(a + b, "Float overflow in add", node)
        }
        (RuntimeValue::Tensor(t1), RuntimeValue::Tensor(t2)) => {
            let out_tensor = t1
                .add_broadcast(&t2)
                .map_err(|err| error(err.message, node))?;
            Ok(RuntimeValue::Tensor(Arc::new(out_tensor)))
        }
        _ => Err(error(
            "Type mismatch in add: expected matching numeric types".to_string(),
            node,
        )),
    }
}

pub fn sub_values(
    left: RuntimeValue,
    right: RuntimeValue,
    node: NodeId,
) -> Result<RuntimeValue, InterpreterError> {
    match (left, right) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => a
            .checked_sub(b)
            .map(RuntimeValue::Int)
            .ok_or_else(|| error("Integer overflow in sub".to_string(), node)),
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
            finite_float(a - b, "Float overflow in sub", node)
        }
        (RuntimeValue::Tensor(t1), RuntimeValue::Tensor(t2)) => {
            let out_tensor = t1
                .sub_broadcast(&t2)
                .map_err(|err| error(err.message, node))?;
            Ok(RuntimeValue::Tensor(Arc::new(out_tensor)))
        }
        _ => Err(error(
            "Type mismatch in sub: expected matching numeric types".to_string(),
            node,
        )),
    }
}

pub fn mul_values(
    left: RuntimeValue,
    right: RuntimeValue,
    node: NodeId,
) -> Result<RuntimeValue, InterpreterError> {
    match (left, right) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => a
            .checked_mul(b)
            .map(RuntimeValue::Int)
            .ok_or_else(|| error("Integer overflow in mul".to_string(), node)),
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
            finite_float(a * b, "Float overflow in mul", node)
        }
        (RuntimeValue::Tensor(t1), RuntimeValue::Tensor(t2)) => {
            let out_tensor = t1
                .mul_broadcast(&t2)
                .map_err(|err| error(err.message, node))?;
            Ok(RuntimeValue::Tensor(Arc::new(out_tensor)))
        }
        _ => Err(error(
            "Type mismatch in mul: expected matching numeric types".to_string(),
            node,
        )),
    }
}

pub fn div_values(
    left: RuntimeValue,
    right: RuntimeValue,
    node: NodeId,
) -> Result<RuntimeValue, InterpreterError> {
    match (left, right) {
        (RuntimeValue::Int(a), RuntimeValue::Int(b)) => {
            if b == 0 {
                return Err(error("Division by zero".to_string(), node));
            }
            a.checked_div(b)
                .map(RuntimeValue::Int)
                .ok_or_else(|| error("Integer overflow in div".to_string(), node))
        }
        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
            if b == 0.0 {
                return Err(error("Division by zero".to_string(), node));
            }
            finite_float(a / b, "Float overflow in div", node)
        }
        (RuntimeValue::Tensor(t1), RuntimeValue::Tensor(t2)) => {
            let out_tensor = t1
                .div_broadcast(&t2)
                .map_err(|err| error(err.message, node))?;
            Ok(RuntimeValue::Tensor(Arc::new(out_tensor)))
        }
        _ => Err(error(
            "Type mismatch in div: expected matching numeric types".to_string(),
            node,
        )),
    }
}

pub fn neg_value(
    value: RuntimeValue,
    node: NodeId,
) -> Result<RuntimeValue, InterpreterError> {
    match value {
        RuntimeValue::Int(a) => a
            .checked_neg()
            .map(RuntimeValue::Int)
            .ok_or_else(|| error("Integer overflow in neg".to_string(), node)),
        RuntimeValue::Float(a) => finite_float(-a, "Float overflow in neg", node),
        RuntimeValue::Tensor(t) => {
            let out_tensor = t.neg_elementwise().map_err(|err| error(err.message, node))?;
            Ok(RuntimeValue::Tensor(Arc::new(out_tensor)))
        }
    }
}

fn finite_float(value: f64, message: &str, node: NodeId) -> Result<RuntimeValue, InterpreterError> {
    if value.is_finite() {
        Ok(RuntimeValue::Float(value))
    } else {
        Err(error(message.to_string(), node))
    }
}

fn error(message: String, node: NodeId) -> InterpreterError {
    InterpreterError {
        message,
        node: Some(node),
    }
}
