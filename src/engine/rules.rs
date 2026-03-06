pub const MODEL_REQUIRED_PROPERTIES: &[&str] = &["layers"];
pub const TRAIN_REQUIRED_PROPERTIES: &[&str] = &[];

pub const MODEL_KNOWN_PROPERTIES: &[&str] = &[
    "layers",
    "activation",
    "optimizer",
    "precision",
    "memory",
    "seed",
    "use",
    "dropout",
    "layernorm",
];
pub const DATASET_KNOWN_PROPERTIES: &[&str] = &[
    "batch",
    "shuffle",
    "source",
    "val_split",
    "label_col",
    "num_classes",
    "type",
];
pub const TRAIN_KNOWN_PROPERTIES: &[&str] =
    &["epochs", "device", "optimizer", "lr", "batch", "precision", "clip_grad", "weight_decay", "warmup_steps", "lr_schedule", "early_stopping", "gradient_accumulation"];

pub const ACTIVATIONS: &[&str] = &["relu", "sigmoid", "tanh", "softmax", "leaky_relu", "leakyrelu", "silu", "gelu"];
pub const OPTIMIZERS: &[&str] = &["adam", "adamw", "sgd", "rmsprop", "rms_prop", "adagrad"];
pub const DEVICES: &[&str] = &["cpu", "gpu", "cuda", "auto"];

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
