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
