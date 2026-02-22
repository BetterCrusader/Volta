use std::collections::HashMap;

use crate::ir::tensor::Tensor;

#[derive(Debug, Clone)]
pub enum OptimizerConfig {
    Sgd {
        lr: f32,
    },
    Adam {
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OptimizerState {
    pub(crate) step: usize,
    pub(crate) adam_m: HashMap<String, Tensor>,
    pub(crate) adam_v: HashMap<String, Tensor>,
}

#[derive(Debug, Clone)]
pub struct OptimizerError {
    pub message: String,
}

pub fn apply_gradients(
    parameters: &mut HashMap<String, Tensor>,
    gradients: &HashMap<String, Tensor>,
    config: &OptimizerConfig,
    state: &mut OptimizerState,
) -> Result<(), OptimizerError> {
    state.step = state.step.saturating_add(1);

    match config {
        OptimizerConfig::Sgd { lr } => apply_sgd(parameters, gradients, *lr),
        OptimizerConfig::Adam {
            lr,
            beta1,
            beta2,
            epsilon,
        } => apply_adam(parameters, gradients, *lr, *beta1, *beta2, *epsilon, state),
    }
}

fn apply_sgd(
    parameters: &mut HashMap<String, Tensor>,
    gradients: &HashMap<String, Tensor>,
    lr: f32,
) -> Result<(), OptimizerError> {
    // BUG FIX: a NaN or non-positive lr would silently corrupt all parameters
    // without any error message. Validate before use.
    if !lr.is_finite() || lr <= 0.0 {
        return Err(OptimizerError {
            message: format!(
                "SGD learning rate must be a finite positive number, got {lr}"
            ),
        });
    }

    for (name, parameter) in parameters {
        let Some(gradient) = gradients.get(name) else {
            continue;
        };
        if parameter.shape != gradient.shape {
            return Err(OptimizerError {
                message: format!(
                    "Shape mismatch in SGD for '{name}': {:?} vs {:?}",
                    parameter.shape, gradient.shape
                ),
            });
        }
        for (p, g) in parameter.data.iter_mut().zip(gradient.data.iter()) {
            *p -= lr * *g;
        }
    }
    Ok(())
}

fn apply_adam(
    parameters: &mut HashMap<String, Tensor>,
    gradients: &HashMap<String, Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    state: &mut OptimizerState,
) -> Result<(), OptimizerError> {
    // BUG FIX: validate all hyperparameters before use.
    //
    // If beta1 == 1.0:  bias1 = 1 - 1^step = 0.0 → m_hat = m / 0 = NaN.
    // If beta2 == 1.0:  bias2 = 1 - 1^step = 0.0 → v_hat = v / 0 = NaN.
    // If epsilon <= 0:  denominator can reach 0 → parameter update = ±Inf.
    //
    // In all cases the corruption propagated silently without any error.
    if !lr.is_finite() || lr <= 0.0 {
        return Err(OptimizerError {
            message: format!(
                "Adam learning rate must be a finite positive number, got {lr}"
            ),
        });
    }
    if !(0.0..1.0).contains(&beta1) || !beta1.is_finite() {
        return Err(OptimizerError {
            message: format!("Adam beta1 must be in [0, 1), got {beta1}"),
        });
    }
    if !(0.0..1.0).contains(&beta2) || !beta2.is_finite() {
        return Err(OptimizerError {
            message: format!("Adam beta2 must be in [0, 1), got {beta2}"),
        });
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimizerError {
            message: format!(
                "Adam epsilon must be a finite positive number, got {epsilon}"
            ),
        });
    }

    let step_i32 = i32::try_from(state.step).map_err(|_| OptimizerError {
        message: "Optimizer step overflow for Adam bias correction".to_string(),
    })?;

    // Guaranteed > 0 because beta values are strictly in [0, 1) and step >= 1.
    let bias1 = 1.0_f32 - beta1.powi(step_i32);
    let bias2 = 1.0_f32 - beta2.powi(step_i32);
    // Sanity guard against extreme floating-point underflow (extremely rare).
    if bias1 <= 0.0 || bias2 <= 0.0 {
        return Err(OptimizerError {
            message: format!(
                "Adam bias correction underflowed (bias1={bias1}, bias2={bias2}); \
                 step={} with beta1={beta1}, beta2={beta2}",
                state.step,
            ),
        });
    }

    for (name, parameter) in parameters {
        let Some(gradient) = gradients.get(name) else {
            continue;
        };
        if parameter.shape != gradient.shape {
            return Err(OptimizerError {
                message: format!(
                    "Shape mismatch in Adam for '{name}': {:?} vs {:?}",
                    parameter.shape, gradient.shape
                ),
            });
        }

        let m = state.adam_m.entry(name.clone()).or_insert(
            Tensor::zeros(parameter.shape.clone()).map_err(|err| OptimizerError {
                message: format!(
                    "Failed to initialize Adam m buffer for '{name}': {}",
                    err.message
                ),
            })?,
        );
        let v = state.adam_v.entry(name.clone()).or_insert(
            Tensor::zeros(parameter.shape.clone()).map_err(|err| OptimizerError {
                message: format!(
                    "Failed to initialize Adam v buffer for '{name}': {}",
                    err.message
                ),
            })?,
        );

        let one_minus_beta1 = 1.0_f32 - beta1;
        let one_minus_beta2 = 1.0_f32 - beta2;

        for ((p, g), (m_i, v_i)) in parameter
            .data
            .iter_mut()
            .zip(gradient.data.iter())
            .zip(m.data.iter_mut().zip(v.data.iter_mut()))
        {
            *m_i = beta1 * *m_i + one_minus_beta1 * *g;
            *v_i = beta2 * *v_i + one_minus_beta2 * (*g * *g);

            let m_hat = *m_i / bias1;
            let v_hat = *v_i / bias2;
            *p -= lr * (m_hat / (v_hat.sqrt() + epsilon));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ir::optimizer::{OptimizerConfig, OptimizerState, apply_gradients};
    use crate::ir::tensor::Tensor;

    #[test]
    fn sgd_updates_parameter_values() {
        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1], vec![1.0]).expect("valid tensor"),
        );

        let mut grads = HashMap::new();
        grads.insert(
            "w".to_string(),
            Tensor::new(vec![1], vec![0.5]).expect("valid tensor"),
        );

        let mut state = OptimizerState::default();
        apply_gradients(
            &mut params,
            &grads,
            &OptimizerConfig::Sgd { lr: 0.1 },
            &mut state,
        )
        .expect("sgd update should succeed");

        let w = params.get("w").expect("param exists");
        assert!((w.data[0] - 0.95).abs() < 1e-6);
    }

    // ── Regression tests for optimizer bugs found in the production code audit ─

    #[test]
    fn sgd_rejects_non_positive_learning_rate() {
        // BUG FIX: SGD with lr=0 or negative silently corrupts parameters.
        let mut params = HashMap::new();
        params.insert("w".to_string(), Tensor::new(vec![1], vec![1.0]).unwrap());
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Tensor::new(vec![1], vec![1.0]).unwrap());
        let mut state = OptimizerState::default();

        for bad_lr in [0.0_f32, -0.01, f32::NAN, f32::INFINITY] {
            let err = apply_gradients(
                &mut params,
                &grads,
                &OptimizerConfig::Sgd { lr: bad_lr },
                &mut state,
            )
            .expect_err(&format!("lr={bad_lr} must be rejected"));
            assert!(
                err.message.contains("learning rate"),
                "expected 'learning rate' in error for lr={bad_lr}, got: {}",
                err.message
            );
        }
    }

    #[test]
    fn adam_rejects_beta_equal_to_one() {
        // BUG FIX: beta1=1.0 → bias1=0 → m_hat=m/0=NaN propagated silently.
        let mut params = HashMap::new();
        params.insert("w".to_string(), Tensor::new(vec![1], vec![1.0]).unwrap());
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Tensor::new(vec![1], vec![1.0]).unwrap());
        let mut state = OptimizerState::default();

        let err = apply_gradients(
            &mut params,
            &grads,
            &OptimizerConfig::Adam {
                lr: 0.001,
                beta1: 1.0, // invalid – bias1 = 0
                beta2: 0.999,
                epsilon: 1e-8,
            },
            &mut state,
        )
        .expect_err("beta1=1.0 must be rejected");
        assert!(
            err.message.contains("beta1"),
            "expected beta1 rejection, got: {}",
            err.message
        );
    }

    #[test]
    fn adam_rejects_non_positive_epsilon() {
        // BUG FIX: epsilon <= 0 makes the denominator reach 0 → Inf update.
        let mut params = HashMap::new();
        params.insert("w".to_string(), Tensor::new(vec![1], vec![1.0]).unwrap());
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Tensor::new(vec![1], vec![1.0]).unwrap());
        let mut state = OptimizerState::default();

        let err = apply_gradients(
            &mut params,
            &grads,
            &OptimizerConfig::Adam {
                lr: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 0.0, // invalid
            },
            &mut state,
        )
        .expect_err("epsilon=0 must be rejected");
        assert!(
            err.message.contains("epsilon"),
            "expected epsilon rejection, got: {}",
            err.message
        );
    }
}
