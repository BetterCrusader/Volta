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

pub use contract::{
    ImportedProgram, IrContractVersion, IrDataType, IrGraphContract, IrNodeContract, IrOpContract,
    IrTensorLiteral, IrTensorSpec,
};
pub use plugin::{OpImportPlugin, PluginRegistry};

#[cfg(feature = "onnx-import")]
pub use onnx::{OnnxGraphStub, OnnxImporter, OnnxNodeStub, OnnxOpStub, import_onnx_stub_graph};
