use crate::rules::{
    AUTOPILOT_DEFAULT_BATCH, AUTOPILOT_DEFAULT_EPOCHS, AUTOPILOT_LARGE_MODEL_LR,
    AUTOPILOT_MEDIUM_MODEL_LR, AUTOPILOT_MEDIUM_MODEL_MAX_LAYERS, AUTOPILOT_SMALL_MODEL_LR,
    AUTOPILOT_SMALL_MODEL_MAX_LAYERS,
};

#[derive(Debug, Clone)]
pub struct ModelAutopilotInput {
    pub layer_count: usize,
    pub optimizer: Option<String>,
    pub lr: Option<f64>,
    pub precision: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DatasetAutopilotInput {
    pub batch: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct TrainAutopilotInput {
    pub epochs: Option<i64>,
    pub optimizer: Option<String>,
    pub lr: Option<f64>,
    pub batch: Option<i64>,
    pub precision: Option<String>,
    pub device: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AutopilotContext {
    pub model: ModelAutopilotInput,
    pub dataset: DatasetAutopilotInput,
    pub train: TrainAutopilotInput,
    pub gpu_available: bool,
}

#[derive(Debug, Clone)]
pub struct Decision {
    pub key: &'static str,
    pub value: String,
    pub source: DecisionSource,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionSource {
    Explicit,
    Rule,
    Default,
}

#[derive(Debug, Clone)]
pub struct ResolvedTrainConfig {
    pub epochs: i64,
    pub optimizer: String,
    pub lr: f64,
    pub batch: i64,
    pub precision: String,
    pub device: String,
    pub decisions: Vec<Decision>,
}

pub struct AutopilotEngine;

impl Default for AutopilotEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AutopilotEngine {
    pub fn new() -> Self {
        Self
    }

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
    fn label(self) -> &'static str {
        match self {
            ModelSize::Small => "small",
            ModelSize::Medium => "medium",
            ModelSize::Large => "large",
        }
    }
}

fn classify_model(layer_count: usize) -> ModelSize {
    if layer_count <= AUTOPILOT_SMALL_MODEL_MAX_LAYERS {
        ModelSize::Small
    } else if layer_count <= AUTOPILOT_MEDIUM_MODEL_MAX_LAYERS {
        ModelSize::Medium
    } else {
        ModelSize::Large
    }
}

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
