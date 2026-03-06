use crate::ir::{StaticMemoryBudget, StaticMemoryBudgetReport, evaluate_static_memory_budget};
use crate::model::CompiledModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingTier {
    Baseline,
    Large,
    Xl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalingProfile {
    pub tier: ScalingTier,
    pub max_peak_live_bytes: usize,
    pub max_peak_live_values: usize,
    pub gradient_checkpointing: bool,
}

impl ScalingProfile {
    #[must_use]
    pub fn xl_default() -> Self {
        Self {
            tier: ScalingTier::Xl,
            max_peak_live_bytes: 64 * 1024 * 1024,
            max_peak_live_values: 250_000,
            gradient_checkpointing: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScalingError {
    pub message: String,
}

pub fn enforce_xl_budget(
    model: &CompiledModel,
    profile: &ScalingProfile,
) -> Result<StaticMemoryBudgetReport, ScalingError> {
    evaluate_static_memory_budget(
        &model.graph,
        &StaticMemoryBudget {
            max_peak_live_bytes: profile.max_peak_live_bytes,
            max_peak_live_values: profile.max_peak_live_values,
        },
    )
    .map_err(|err| ScalingError {
        message: format!(
            "{} profile rejected model: {}",
            profile_label(*profile),
            err.message
        ),
    })
}

fn profile_label(profile: ScalingProfile) -> &'static str {
    match profile.tier {
        ScalingTier::Baseline => "baseline",
        ScalingTier::Large => "large",
        ScalingTier::Xl => "xl",
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{
        ScalingProfile, build_tiny_transformer_fixture_for_tests, enforce_xl_budget,
    };

    #[test]
    fn xl_default_profile_accepts_tiny_transformer_fixture() {
        let (model, _dataset, _cfg, _infer_input) = build_tiny_transformer_fixture_for_tests();
        let report = enforce_xl_budget(&model, &ScalingProfile::xl_default())
            .expect("fixture should pass xl default profile");
        assert!(report.within_budget);
    }
}
