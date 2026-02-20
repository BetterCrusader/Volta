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

#[derive(Debug, Clone, Default)]
pub struct OptimizerState {
    step: usize,
    adam_m: HashMap<String, Tensor>,
    adam_v: HashMap<String, Tensor>,
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
    let step_i32 = i32::try_from(state.step).map_err(|_| OptimizerError {
        message: "Optimizer step overflow for Adam bias correction".to_string(),
    })?;

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
        let bias1 = 1.0_f32 - beta1.powi(step_i32);
        let bias2 = 1.0_f32 - beta2.powi(step_i32);

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
}
