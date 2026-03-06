mod builder;
mod checkpoint;
mod gguf_loader;
mod mixed_precision;
mod training_logger;
mod dataset;
mod export;
mod gradient_checkpointing;
mod layers;
mod losses;
mod module_trait;
mod parameter;
mod safetensors;
mod scaling;
mod tensor_shape;
mod tiny_transformer;
mod train_api;

pub use builder::{CompiledModel, ModelBuildError, ModelBuilder};
pub use checkpoint::{load_checkpoint, save_checkpoint, save_checkpoint_compressed, load_checkpoint_compressed};
pub use safetensors::{load_safetensors, save_safetensors};
pub use gguf_loader::{GgufDtype, GgufInfo, GgufMetaValue, GgufTensorInfo, print_gguf_info, read_gguf_info};
pub use mixed_precision::{
    LossScalerState, MixedPrecisionConfig, MixedPrecisionTrainer, simulate_fp16,
};
pub use training_logger::{LogEvent, LogTarget, LoggerConfig, TrainingLogger};
pub use dataset::{BatchIterator, Dataset, Example};
pub use export::{ModelExportError, export_compiled_model_manifest};
pub use gradient_checkpointing::{
    GradientCheckpointPlan, GradientCheckpointingConfig, GradientCheckpointingError,
    RecomputeSegment, plan_gradient_checkpointing,
};
pub use layers::{BatchNormLayer, Conv2DLayer, LinearLayer, ReLULayer, Sequential, SoftmaxLayer};
pub use losses::{CrossEntropyLoss, MSELoss};
pub use module_trait::Module;
pub use parameter::Parameter;
pub use scaling::{ScalingError, ScalingProfile, ScalingTier, enforce_xl_budget};
pub use tensor_shape::TensorShape;
pub use tiny_transformer::{
    TinyTransformerFixtureDataset, build_tiny_transformer_fixture_for_tests,
};
pub use train_api::{
    ReproducibilityMode, TrainApiConfig, TrainApiError, TrainApiResult, infer, infer_with_backend,
    train, train_with_backend,
};
