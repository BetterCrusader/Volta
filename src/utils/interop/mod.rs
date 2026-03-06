pub mod contract;
#[cfg(feature = "onnx-import")]
pub mod onnx;
pub mod plugin;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteropError {
    pub message: String,
}

impl InteropError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn unsupported_op(node_name: &str, op_type: &str, phase: &str) -> Self {
        Self::new(format!(
            "[UNSUPPORTED_OP] [{phase}] node='{node_name}' op='{op_type}'"
        ))
    }

    pub fn unsupported_opset(actual: i64, max_supported: i64) -> Self {
        Self::new(format!(
            "[UNSUPPORTED_OPSET] [import] opset={actual} max_supported={max_supported}"
        ))
    }

    pub fn invalid_attribute(
        node_name: &str,
        op_type: &str,
        attr_name: &str,
        reason: &str,
    ) -> Self {
        Self::new(format!(
            "[INVALID_ATTRIBUTE] [import] node='{node_name}' op='{op_type}' attr='{attr_name}' reason='{reason}'"
        ))
    }
}

impl From<crate::ir::tensor::TensorError> for InteropError {
    fn from(err: crate::ir::tensor::TensorError) -> Self {
        InteropError {
            message: err.message,
        }
    }
}

impl From<crate::ir::VerifyError> for InteropError {
    fn from(err: crate::ir::VerifyError) -> Self {
        InteropError {
            message: err.message,
        }
    }
}

impl From<crate::ir::ShapeError> for InteropError {
    fn from(err: crate::ir::ShapeError) -> Self {
        InteropError {
            message: err.message,
        }
    }
}

pub use contract::{
    ImportedProgram, IrContractVersion, IrDataType, IrGraphContract, IrNodeContract, IrOpContract,
    IrTensorLiteral, IrTensorSpec,
};
pub use plugin::{OpImportPlugin, PluginRegistry};

#[cfg(feature = "onnx-import")]
pub use onnx::{
    OnnxGraphStub, OnnxImporter, OnnxNodeStub, OnnxOpStub, import_onnx_bytes, import_onnx_file,
    import_onnx_stub_graph,
};
pub mod python_exporter;
