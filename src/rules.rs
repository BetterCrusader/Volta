pub const MODEL_REQUIRED_PROPERTIES: &[&str] = &["layers"];
pub const TRAIN_REQUIRED_PROPERTIES: &[&str] = &[];

pub const MODEL_KNOWN_PROPERTIES: &[&str] =
    &["layers", "activation", "optimizer", "precision", "memory"];
pub const DATASET_KNOWN_PROPERTIES: &[&str] = &["batch", "shuffle"];
pub const TRAIN_KNOWN_PROPERTIES: &[&str] =
    &["epochs", "device", "optimizer", "lr", "batch", "precision"];

pub const ACTIVATIONS: &[&str] = &["relu", "sigmoid", "tanh"];
pub const OPTIMIZERS: &[&str] = &["adam", "adamw", "sgd"];
pub const DEVICES: &[&str] = &["cpu", "gpu", "auto"];

pub const AUTOPILOT_DEFAULT_EPOCHS: i64 = 10;
pub const AUTOPILOT_DEFAULT_BATCH: i64 = 32;
pub const AUTOPILOT_DEFAULT_ACTIVATION: &str = "relu";

pub const AUTOPILOT_SMALL_MODEL_MAX_LAYERS: usize = 3;
pub const AUTOPILOT_MEDIUM_MODEL_MAX_LAYERS: usize = 6;

pub const AUTOPILOT_SMALL_MODEL_LR: f64 = 0.001;
pub const AUTOPILOT_MEDIUM_MODEL_LR: f64 = 0.0007;
pub const AUTOPILOT_LARGE_MODEL_LR: f64 = 0.0003;

pub const FLOAT_EPSILON: f64 = 1e-9;
pub const MAX_SAFE_INT_F64: i64 = 9_007_199_254_740_992;
