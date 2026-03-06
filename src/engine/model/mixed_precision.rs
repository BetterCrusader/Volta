use crate::ir::Tensor;
use crate::ir::node::ValueId;
/// Mixed-precision training utilities for Volta.
///
/// Implements automatic loss scaling to prevent FP16/BF16 gradient underflow,
/// gradient finite-checking, and a `MixedPrecisionTrainer` wrapper that manages
/// the scale factor automatically.
///
/// While Volta's internal tensor storage is always FP32, this module provides:
/// 1. Dynamic loss scaling (scale loss up → scale gradients down after backward pass)
/// 2. Infinite/NaN gradient detection → skip update and reduce scale
/// 3. Automatic scale increase after N consecutive clean steps
use std::collections::HashMap;

/// Configuration for automatic mixed-precision (AMP) loss scaling.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Initial loss scale factor (default: 65536.0 = 2^16).
    pub initial_scale: f32,
    /// Multiply scale by this factor after `scale_growth_interval` clean steps.
    pub scale_growth_factor: f32,
    /// Divide scale by this factor after a NaN/Inf gradient is detected.
    pub scale_reduction_factor: f32,
    /// Number of clean consecutive steps before growing the scale.
    pub scale_growth_interval: usize,
    /// Minimum allowed scale (stop reducing below this).
    pub min_scale: f32,
    /// Maximum allowed scale.
    pub max_scale: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            initial_scale: 65536.0,
            scale_growth_factor: 2.0,
            scale_reduction_factor: 2.0,
            scale_growth_interval: 2000,
            min_scale: 1.0,
            max_scale: 2.0f32.powi(24),
        }
    }
}

/// State of the mixed-precision loss scaler.
#[derive(Debug, Clone)]
pub struct LossScalerState {
    pub current_scale: f32,
    pub steps_since_last_overflow: usize,
    pub total_overflows: usize,
    pub total_steps: usize,
}

impl LossScalerState {
    pub fn new(initial_scale: f32) -> Self {
        Self {
            current_scale: initial_scale,
            steps_since_last_overflow: 0,
            total_overflows: 0,
            total_steps: 0,
        }
    }
}

/// Mixed-precision trainer that wraps the loss scaling lifecycle.
pub struct MixedPrecisionTrainer {
    pub config: MixedPrecisionConfig,
    pub state: LossScalerState,
}

impl MixedPrecisionTrainer {
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let initial_scale = config.initial_scale;
        Self {
            state: LossScalerState::new(initial_scale),
            config,
        }
    }

    /// Scale a loss value before backward pass.
    /// The caller must divide gradients by `current_scale` before the optimizer step.
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.state.current_scale
    }

    /// Unscale gradients in-place: grad = grad / current_scale.
    /// Returns `true` if gradients are finite and the optimizer step should proceed.
    /// Returns `false` if any gradient is Inf or NaN (skip the step).
    pub fn unscale_and_check(&mut self, gradients: &mut HashMap<ValueId, Tensor>) -> bool {
        let inv_scale = 1.0 / self.state.current_scale;
        let mut overflow = false;

        for tensor in gradients.values_mut() {
            let data = std::sync::Arc::make_mut(&mut tensor.data);
            for v in data.iter_mut() {
                *v *= inv_scale;
                if !v.is_finite() {
                    overflow = true;
                }
            }
        }

        self.state.total_steps += 1;

        if overflow {
            self.state.total_overflows += 1;
            self.state.steps_since_last_overflow = 0;
            // Reduce scale, but not below min
            let new_scale = (self.state.current_scale / self.config.scale_reduction_factor)
                .max(self.config.min_scale);
            self.state.current_scale = new_scale;
            false
        } else {
            self.state.steps_since_last_overflow += 1;
            // Grow scale after sufficient clean steps
            if self.state.steps_since_last_overflow >= self.config.scale_growth_interval {
                let new_scale = (self.state.current_scale * self.config.scale_growth_factor)
                    .min(self.config.max_scale);
                self.state.current_scale = new_scale;
                self.state.steps_since_last_overflow = 0;
            }
            true
        }
    }

    /// Print a summary of the scaler state.
    pub fn print_report(&self) {
        println!(
            "[MixedPrecision] scale={:.1} overflows={} steps={}",
            self.state.current_scale, self.state.total_overflows, self.state.total_steps,
        );
    }
}

/// Quantize an f32 tensor to simulated FP16 precision (round to FP16, store as f32).
/// Useful for testing mixed-precision pipelines without hardware FP16 support.
pub fn simulate_fp16(tensor: &Tensor) -> Tensor {
    let data: Vec<f32> = tensor
        .data
        .iter()
        .map(|&v| f32_to_simulated_fp16(v))
        .collect();
    Tensor::new(tensor.shape.clone(), data).expect("simulate_fp16: valid tensor")
}

fn f32_to_simulated_fp16(v: f32) -> f32 {
    // Convert to f16 bit pattern and back
    let bits = v.to_bits();
    let sign = bits & 0x8000_0000u32;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf or NaN — keep as-is
        return v;
    }

    let exp16 = exp - 127 + 15;
    if exp16 <= 0 {
        // Underflow to zero (subnormals not handled for simplicity)
        return f32::from_bits(sign);
    }
    if exp16 >= 31 {
        // Overflow to Inf
        return f32::from_bits(sign | 0x7F800000);
    }

    // Round mantissa to 10 bits
    let mant10 = (mant >> 13) + ((mant >> 12) & 1); // round half-up
    let mant10 = mant10.min(0x3FF);

    // Reconstruct as f32
    let exp32 = ((exp16 as u32) + 127 - 15) << 23;
    f32::from_bits(sign | exp32 | (mant10 << 13))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Tensor;
    use crate::ir::node::ValueId;

    fn vid(n: usize) -> ValueId {
        ValueId(n)
    }

    #[test]
    fn loss_scaling_multiplies_correctly() {
        let trainer = MixedPrecisionTrainer::new(MixedPrecisionConfig {
            initial_scale: 1024.0,
            ..Default::default()
        });
        assert!((trainer.scale_loss(2.0) - 2048.0).abs() < 1e-4);
    }

    #[test]
    fn unscale_and_check_finite_gradients() {
        let mut trainer = MixedPrecisionTrainer::new(MixedPrecisionConfig {
            initial_scale: 4.0,
            ..Default::default()
        });
        let mut grads = HashMap::new();
        grads.insert(vid(0), Tensor::new(vec![2], vec![4.0, 8.0]).unwrap());

        let ok = trainer.unscale_and_check(&mut grads);
        assert!(ok, "finite gradients should not trigger overflow");
        let g = &grads[&vid(0)];
        assert!((g.data[0] - 1.0).abs() < 1e-6);
        assert!((g.data[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn unscale_reduces_scale_on_nan() {
        let mut trainer = MixedPrecisionTrainer::new(MixedPrecisionConfig {
            initial_scale: 1024.0,
            scale_reduction_factor: 2.0,
            min_scale: 1.0,
            ..Default::default()
        });
        let mut grads = HashMap::new();
        grads.insert(vid(0), Tensor::new(vec![1], vec![f32::NAN]).unwrap());

        let ok = trainer.unscale_and_check(&mut grads);
        assert!(!ok, "NaN gradients should trigger overflow");
        assert!((trainer.state.current_scale - 512.0).abs() < 1.0);
        assert_eq!(trainer.state.total_overflows, 1);
    }

    #[test]
    fn scale_grows_after_clean_steps() {
        let mut trainer = MixedPrecisionTrainer::new(MixedPrecisionConfig {
            initial_scale: 8.0,
            scale_growth_factor: 2.0,
            scale_growth_interval: 3,
            max_scale: 1024.0,
            scale_reduction_factor: 2.0,
            min_scale: 1.0,
        });

        for _ in 0..3 {
            let mut grads = HashMap::new();
            grads.insert(vid(0), Tensor::new(vec![1], vec![1.0]).unwrap());
            trainer.unscale_and_check(&mut grads);
        }

        assert!(
            (trainer.state.current_scale - 16.0).abs() < 1e-4,
            "scale should have doubled after 3 clean steps, got {}",
            trainer.state.current_scale
        );
    }

    #[test]
    fn simulate_fp16_identity_on_round_numbers() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, -4.0]).unwrap();
        let t2 = simulate_fp16(&t);
        for (a, b) in t2.data.iter().zip(t.data.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }
}
