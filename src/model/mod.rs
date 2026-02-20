mod builder;
mod checkpoint;
mod dataset;
mod layers;
mod losses;
mod module_trait;
mod parameter;
mod scaling;
mod tensor_shape;
mod tiny_transformer;
mod train_api;

pub use builder::{CompiledModel, ModelBuildError, ModelBuilder};
pub use checkpoint::{load_checkpoint, save_checkpoint};
pub use dataset::{BatchIterator, Dataset, Example};
pub use layers::{Conv2DLayer, LinearLayer, ReLULayer, Sequential, SoftmaxLayer};
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
