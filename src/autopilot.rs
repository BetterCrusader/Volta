use crate::rules::{
    AUTOPILOT_DEFAULT_BATCH, AUTOPILOT_DEFAULT_EPOCHS, AUTOPILOT_LARGE_MODEL_LR,
    AUTOPILOT_MEDIUM_MODEL_LR, AUTOPILOT_MEDIUM_MODEL_MAX_LAYERS, AUTOPILOT_SMALL_MODEL_LR,
    AUTOPILOT_SMALL_MODEL_MAX_LAYERS,
};

/// Model-level parameters fed into the autopilot decision engine.
#[derive(Debug, Clone)]
pub struct ModelAutopilotInput {
    /// Number of declared layers.
    pub layer_count: usize,
    /// Explicit optimizer name, if set at model level.
    pub optimizer: Option<String>,
    /// Explicit learning rate, if set at model level.
    pub lr: Option<f64>,
    /// Explicit precision tag, if set at model level.
    pub precision: Option<String>,
}

/// Dataset-level parameters fed into the autopilot decision engine.
#[derive(Debug, Clone)]
pub struct DatasetAutopilotInput {
    /// Batch size, if declared in the `dataset` block.
    pub batch: Option<i64>,
}

/// Train-statement parameters that override model/dataset defaults.
#[derive(Debug, Clone)]
pub struct TrainAutopilotInput {
    /// Explicit epoch count, if provided in `train`.
    pub epochs: Option<i64>,
    /// Explicit optimizer override for this training run.
    pub optimizer: Option<String>,
    /// Explicit learning rate override for this training run.
    pub lr: Option<f64>,
    /// Explicit batch-size override for this training run.
    pub batch: Option<i64>,
    /// Explicit precision override for this training run.
    pub precision: Option<String>,
    /// Target device, e.g. `"gpu"` or `"cpu"`.
    pub device: Option<String>,
}

/// Aggregated context passed to [`AutopilotEngine::resolve`].
#[derive(Debug, Clone)]
pub struct AutopilotContext {
    /// Model-level settings.
    pub model: ModelAutopilotInput,
    /// Dataset-level settings.
    pub dataset: DatasetAutopilotInput,
    /// Train-statement-level settings (override model/dataset).
    pub train: TrainAutopilotInput,
    /// Whether a GPU device was detected at runtime.
    pub gpu_available: bool,
}

/// Records a single autopilot decision: what was chosen, why, and from where.
#[derive(Debug, Clone)]
pub struct Decision {
    /// The configuration key this decision applies to, e.g. `"epochs"`.
    pub key: &'static str,
    /// The resolved value, formatted as a string.
    pub value: String,
    /// Origin of this decision.
    pub source: DecisionSource,
    /// Human-readable explanation for the value chosen.
    pub reason: String,
}

/// The origin of an autopilot decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionSource {
    /// Value was explicitly set by the user in source code.
    Explicit,
    /// Value was inferred by an autopilot rule (e.g. model size heuristic).
    Rule,
    /// Value fell back to a hard-coded default.
    Default,
}

/// The fully resolved training configuration, ready for the executor.
#[derive(Debug, Clone)]
pub struct ResolvedTrainConfig {
    /// Number of epochs to train.
    pub epochs: i64,
    /// Optimizer name.
    pub optimizer: String,
    /// Learning rate.
    pub lr: f64,
    /// Batch size.
    pub batch: i64,
    /// Numerical precision tag.
    pub precision: String,
    /// Target device.
    pub device: String,
    /// Ordered list of decisions made by the autopilot.
    pub decisions: Vec<Decision>,
}

/// Autopilot engine that fills in missing training hyperparameters
/// using rule-based heuristics and sensible defaults.
pub struct AutopilotEngine;

impl Default for AutopilotEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AutopilotEngine {
    /// Creates a new [`AutopilotEngine`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Resolves a complete [`ResolvedTrainConfig`] from the given context.
    ///
    /// For each hyperparameter the engine applies a three-tier priority:
    /// 1. **Explicit** — value set directly in the `train` block.
    /// 2. **Rule** — value inferred from model size or other heuristics.
    /// 3. **Default** — hard-coded fallback constant from `rules.rs`.
    ///
    /// All decisions are recorded in [`ResolvedTrainConfig::decisions`]
    /// so the executor can print a transparent configuration summary.
    #[must_use]
    pub fn resolve(&self, input: &AutopilotContext) -> ResolvedTrainConfig {
        let mut decisions = Vec::new();

        let epochs = if let Some(explicit) = input.train.epochs {
            decisions.push(Decision::explicit("epochs", explicit.to_string()));
            explicit
        } else {
            decisions.push(Decision::default(
                "epochs",
                AUTOPILOT_DEFAULT_EPOCHS.to_string(),
                format!("fallback epochs={AUTOPILOT_DEFAULT_EPOCHS}"),
            ));
            AUTOPILOT_DEFAULT_EPOCHS
        };

        let model_size = classify_model(input.model.layer_count);
        let optimizer = if let Some(explicit) = input.train.optimizer.as_ref() {
            decisions.push(Decision::explicit("optimizer", explicit.clone()));
            explicit.clone()
        } else if let Some(explicit) = input.model.optimizer.as_ref() {
            decisions.push(Decision::explicit("optimizer", explicit.clone()));
            explicit.clone()
        } else {
            let selected = match model_size {
                ModelSize::Small | ModelSize::Medium => "adam",
                ModelSize::Large => "adamw",
            };
            decisions.push(Decision::rule(
                "optimizer",
                selected.to_string(),
                format!("{} model", model_size.label()),
            ));
            selected.to_string()
        };

        let lr = if let Some(explicit) = input.train.lr {
            decisions.push(Decision::explicit("lr", explicit.to_string()));
            explicit
        } else if let Some(explicit) = input.model.lr {
            decisions.push(Decision::explicit("lr", explicit.to_string()));
            explicit
        } else {
            let selected = match model_size {
                ModelSize::Small => AUTOPILOT_SMALL_MODEL_LR,
                ModelSize::Medium => AUTOPILOT_MEDIUM_MODEL_LR,
                ModelSize::Large => AUTOPILOT_LARGE_MODEL_LR,
            };
            decisions.push(Decision::rule(
                "lr",
                selected.to_string(),
                format!("{} model", model_size.label()),
            ));
            selected
        };

        let batch = if let Some(explicit) = input.train.batch {
            decisions.push(Decision::explicit("batch", explicit.to_string()));
            explicit
        } else if let Some(explicit) = input.dataset.batch {
            decisions.push(Decision::explicit("batch", explicit.to_string()));
            explicit
        } else {
            decisions.push(Decision::default(
                "batch",
                AUTOPILOT_DEFAULT_BATCH.to_string(),
                format!("fallback batch={AUTOPILOT_DEFAULT_BATCH}"),
            ));
            AUTOPILOT_DEFAULT_BATCH
        };

        let device = if let Some(explicit) = input.train.device.as_ref() {
            if explicit == "auto" {
                let selected = detected_device(input.gpu_available);
                decisions.push(Decision::rule(
                    "device",
                    selected.to_string(),
                    "train.device=auto".to_string(),
                ));
                selected.to_string()
            } else {
                decisions.push(Decision::explicit("device", explicit.clone()));
                explicit.clone()
            }
        } else {
            let selected = detected_device(input.gpu_available);
            decisions.push(Decision::rule(
                "device",
                selected.to_string(),
                "hardware detection".to_string(),
            ));
            selected.to_string()
        };

        let precision = if let Some(explicit) = input.train.precision.as_ref() {
            decisions.push(Decision::explicit("precision", explicit.clone()));
            explicit.clone()
        } else if let Some(explicit) = input.model.precision.as_ref() {
            decisions.push(Decision::explicit("precision", explicit.clone()));
            explicit.clone()
        } else {
            let selected = if device == "gpu" { "fp16" } else { "fp32" };
            decisions.push(Decision::rule(
                "precision",
                selected.to_string(),
                format!("device={device}"),
            ));
            selected.to_string()
        };

        ResolvedTrainConfig {
            epochs,
            optimizer,
            lr,
            batch,
            precision,
            device,
            decisions,
        }
    }
}

impl Decision {
    fn explicit(key: &'static str, value: String) -> Self {
        Self {
            key,
            value,
            source: DecisionSource::Explicit,
            reason: "explicit override".to_string(),
        }
    }

    fn rule(key: &'static str, value: String, reason: String) -> Self {
        Self {
            key,
            value,
            source: DecisionSource::Rule,
            reason,
        }
    }

    fn default(key: &'static str, value: String, reason: String) -> Self {
        Self {
            key,
            value,
            source: DecisionSource::Default,
            reason,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ModelSize {
    Small,
    Medium,
    Large,
}

impl ModelSize {
    #[must_use]
    const fn label(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }
}

#[must_use]
fn classify_model(layer_count: usize) -> ModelSize {
    if layer_count <= AUTOPILOT_SMALL_MODEL_MAX_LAYERS {
        ModelSize::Small
    } else if layer_count <= AUTOPILOT_MEDIUM_MODEL_MAX_LAYERS {
        ModelSize::Medium
    } else {
        ModelSize::Large
    }
}

#[must_use]
fn detected_device(gpu_available: bool) -> &'static str {
    if gpu_available { "gpu" } else { "cpu" }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_context() -> AutopilotContext {
        AutopilotContext {
            model: ModelAutopilotInput {
                layer_count: 3,
                optimizer: None,
                lr: None,
                precision: None,
            },
            dataset: DatasetAutopilotInput { batch: None },
            train: TrainAutopilotInput {
                epochs: None,
                optimizer: None,
                lr: None,
                batch: None,
                precision: None,
                device: None,
            },
            gpu_available: true,
        }
    }

    #[test]
    fn resolves_defaults_from_rules() {
        let engine = AutopilotEngine::new();
        let resolved = engine.resolve(&base_context());

        assert_eq!(resolved.epochs, AUTOPILOT_DEFAULT_EPOCHS);
        assert_eq!(resolved.optimizer, "adam");
        assert_eq!(resolved.lr, AUTOPILOT_SMALL_MODEL_LR);
        assert_eq!(resolved.batch, AUTOPILOT_DEFAULT_BATCH);
        assert_eq!(resolved.device, "gpu");
        assert_eq!(resolved.precision, "fp16");
        assert_eq!(resolved.decisions.len(), 6);
    }

    #[test]
    fn explicit_overrides_take_priority() {
        let mut input = base_context();
        input.model.optimizer = Some("adamw".to_string());
        input.model.lr = Some(0.004);
        input.dataset.batch = Some(64);
        input.train.optimizer = Some("sgd".to_string());
        input.train.lr = Some(0.05);
        input.train.batch = Some(16);
        input.train.precision = Some("fp32".to_string());
        input.train.device = Some("cpu".to_string());
        input.train.epochs = Some(20);

        let engine = AutopilotEngine::new();
        let resolved = engine.resolve(&input);

        assert_eq!(resolved.epochs, 20);
        assert_eq!(resolved.optimizer, "sgd");
        assert_eq!(resolved.lr, 0.05);
        assert_eq!(resolved.batch, 16);
        assert_eq!(resolved.device, "cpu");
        assert_eq!(resolved.precision, "fp32");
        assert!(
            resolved
                .decisions
                .iter()
                .all(|decision| decision.source == DecisionSource::Explicit)
        );
    }

    #[test]
    fn auto_device_uses_detected_hardware() {
        let mut input = base_context();
        input.gpu_available = false;
        input.train.device = Some("auto".to_string());

        let engine = AutopilotEngine::new();
        let resolved = engine.resolve(&input);

        assert_eq!(resolved.device, "cpu");
        assert_eq!(resolved.precision, "fp32");
    }
}
