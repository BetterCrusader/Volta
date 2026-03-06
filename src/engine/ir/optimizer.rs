use std::collections::HashMap;

use crate::ir::tensor::Tensor;

/// Learning rate schedule applied per epoch.
#[derive(Debug, Clone)]
pub enum LrSchedule {
    /// Constant learning rate (default).
    Constant,
    /// Cosine annealing: lr = lr_min + 0.5*(lr_init - lr_min)*(1 + cos(π*epoch/total_epochs))
    Cosine { total_epochs: usize, lr_min: f32 },
    /// Step decay: multiply lr by `gamma` every `step_size` epochs.
    Step { step_size: usize, gamma: f32 },
}

impl LrSchedule {
    pub fn compute_lr(&self, base_lr: f32, epoch: usize) -> f32 {
        match self {
            LrSchedule::Constant => base_lr,
            LrSchedule::Cosine { total_epochs, lr_min } => {
                let t = epoch as f32 / (*total_epochs).max(1) as f32;
                let cos_val = (std::f32::consts::PI * t).cos();
                lr_min + 0.5 * (base_lr - lr_min) * (1.0 + cos_val)
            }
            LrSchedule::Step { step_size, gamma } => {
                let steps = epoch / step_size.max(&1);
                base_lr * gamma.powi(steps as i32)
            }
        }
    }
}

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
    AdamW {
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    RmsProp {
        lr: f32,
        /// Smoothing constant (momentum for squared gradients), default 0.99
        alpha: f32,
        epsilon: f32,
        weight_decay: f32,
        momentum: f32,
    },
    Adagrad {
        lr: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    /// LARS (Layer-wise Adaptive Rate Scaling) optimizer for large-batch training.
    /// Scales the learning rate per-layer based on the ratio of weight norm to gradient norm.
    Lars {
        lr: f32,
        momentum: f32,
        weight_decay: f32,
        /// Trust coefficient η (typically 0.001)
        trust_coeff: f32,
        epsilon: f32,
    },
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OptimizerState {
    pub(crate) step: usize,
    pub(crate) adam_m: HashMap<crate::ir::node::ValueId, Tensor>,
    pub(crate) adam_v: HashMap<crate::ir::node::ValueId, Tensor>,
    /// RMSprop squared gradient running average
    pub(crate) rms_v: HashMap<crate::ir::node::ValueId, Tensor>,
    /// RMSprop momentum buffer
    pub(crate) rms_buf: HashMap<crate::ir::node::ValueId, Tensor>,
    /// Adagrad accumulated squared gradient sum
    pub(crate) adagrad_acc: HashMap<crate::ir::node::ValueId, Tensor>,
    /// LARS momentum velocity
    pub(crate) lars_vel: HashMap<crate::ir::node::ValueId, Tensor>,
}

#[derive(Debug, Clone)]
pub struct OptimizerError {
    pub message: String,
}

impl From<crate::ir::tensor::TensorError> for OptimizerError {
    fn from(err: crate::ir::tensor::TensorError) -> Self {
        OptimizerError {
            message: err.message,
        }
    }
}

pub fn apply_gradients(
    parameters: &mut HashMap<crate::ir::node::ValueId, Tensor>,
    gradients: &HashMap<crate::ir::node::ValueId, Tensor>,
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
        OptimizerConfig::AdamW {
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
        } => apply_adamw(parameters, gradients, *lr, *beta1, *beta2, *epsilon, *weight_decay, state),
        OptimizerConfig::RmsProp {
            lr,
            alpha,
            epsilon,
            weight_decay,
            momentum,
        } => apply_rmsprop(parameters, gradients, *lr, *alpha, *epsilon, *weight_decay, *momentum, state),
        OptimizerConfig::Adagrad { lr, epsilon, weight_decay } =>
            apply_adagrad(parameters, gradients, *lr, *epsilon, *weight_decay, state),
        OptimizerConfig::Lars { lr, momentum, weight_decay, trust_coeff, epsilon } =>
            apply_lars(parameters, gradients, *lr, *momentum, *weight_decay, *trust_coeff, *epsilon, state),
    }
}

use std::sync::Arc;

fn apply_sgd(
    parameters: &mut HashMap<crate::ir::node::ValueId, Tensor>,
    gradients: &HashMap<crate::ir::node::ValueId, Tensor>,
    lr: f32,
) -> Result<(), OptimizerError> {
    if !lr.is_finite() || lr <= 0.0 {
        return Err(OptimizerError {
            message: format!("SGD learning rate must be a finite positive number, got {lr}"),
        });
    }

    for (value_id, parameter) in parameters {
        let Some(gradient) = gradients.get(value_id) else {
            continue;
        };
        if parameter.shape != gradient.shape {
            return Err(OptimizerError {
                message: format!(
                    "Shape mismatch in SGD for ValueId {}: {:?} vs {:?}",
                    value_id.0, parameter.shape, gradient.shape
                ),
            });
        }

        let p_data = Arc::make_mut(&mut parameter.data);
        let g_contig = gradient.make_contiguous()?;

        for (p, g) in p_data.iter_mut().zip(g_contig.data.iter()) {
            *p -= lr * *g;
        }
    }
    Ok(())
}

fn apply_adam(
    parameters: &mut HashMap<crate::ir::node::ValueId, Tensor>,
    gradients: &HashMap<crate::ir::node::ValueId, Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    state: &mut OptimizerState,
) -> Result<(), OptimizerError> {
    if !lr.is_finite() || lr <= 0.0 {
        return Err(OptimizerError {
            message: format!("Adam learning rate must be a finite positive number, got {lr}"),
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
            message: format!("Adam epsilon must be a finite positive number, got {epsilon}"),
        });
    }

    let step_i32 = i32::try_from(state.step).map_err(|_| OptimizerError {
        message: "Optimizer step overflow for Adam bias correction".to_string(),
    })?;

    let bias1 = 1.0_f32 - beta1.powi(step_i32);
    let bias2 = 1.0_f32 - beta2.powi(step_i32);
    if bias1 <= 0.0 || bias2 <= 0.0 {
        return Err(OptimizerError {
            message: format!(
                "Adam bias correction underflowed (bias1={bias1}, bias2={bias2}); step={}",
                state.step
            ),
        });
    }

    for (value_id, parameter) in parameters {
        let Some(gradient) = gradients.get(value_id) else {
            continue;
        };
        if parameter.shape != gradient.shape {
            return Err(OptimizerError {
                message: format!(
                    "Shape mismatch in Adam for ValueId {}: {:?} vs {:?}",
                    value_id.0, parameter.shape, gradient.shape
                ),
            });
        }

        let m = state.adam_m.entry(*value_id).or_insert(
            Tensor::zeros(parameter.shape.clone()).map_err(|err| OptimizerError {
                message: format!(
                    "Failed to initialize Adam m buffer for ValueId {}: {}",
                    value_id.0, err.message
                ),
            })?,
        );
        let v = state.adam_v.entry(*value_id).or_insert(
            Tensor::zeros(parameter.shape.clone()).map_err(|err| OptimizerError {
                message: format!(
                    "Failed to initialize Adam v buffer for ValueId {}: {}",
                    value_id.0, err.message
                ),
            })?,
        );

        let one_minus_beta1 = 1.0_f32 - beta1;
        let one_minus_beta2 = 1.0_f32 - beta2;

        let p_data = Arc::make_mut(&mut parameter.data);
        let g_contig = gradient.make_contiguous()?;
        let m_data = Arc::make_mut(&mut m.data);
        let v_data = Arc::make_mut(&mut v.data);

        for ((p, g), (m_i, v_i)) in p_data
            .iter_mut()
            .zip(g_contig.data.iter())
            .zip(m_data.iter_mut().zip(v_data.iter_mut()))
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

fn apply_adamw(
    parameters: &mut HashMap<crate::ir::node::ValueId, Tensor>,
    gradients: &HashMap<crate::ir::node::ValueId, Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    state: &mut OptimizerState,
) -> Result<(), OptimizerError> {
    if !lr.is_finite() || lr <= 0.0 {
        return Err(OptimizerError {
            message: format!("AdamW learning rate must be a finite positive number, got {lr}"),
        });
    }
    let step_i32 = i32::try_from(state.step).map_err(|_| OptimizerError {
        message: "Optimizer step overflow for AdamW bias correction".to_string(),
    })?;

    let bias1 = 1.0_f32 - beta1.powi(step_i32);
    let bias2 = 1.0_f32 - beta2.powi(step_i32);

    for (value_id, parameter) in parameters {
        let Some(gradient) = gradients.get(value_id) else {
            continue;
        };

        let m = state.adam_m.entry(*value_id).or_insert(
            Tensor::zeros(parameter.shape.clone()).map_err(|err| OptimizerError {
                message: format!("Failed to init AdamW m: {}", err.message),
            })?,
        );
        let v = state.adam_v.entry(*value_id).or_insert(
            Tensor::zeros(parameter.shape.clone()).map_err(|err| OptimizerError {
                message: format!("Failed to init AdamW v: {}", err.message),
            })?,
        );

        let one_minus_beta1 = 1.0_f32 - beta1;
        let one_minus_beta2 = 1.0_f32 - beta2;

        let p_data = Arc::make_mut(&mut parameter.data);
        let g_contig = gradient.make_contiguous()?;
        let m_data = Arc::make_mut(&mut m.data);
        let v_data = Arc::make_mut(&mut v.data);

        for ((p, g), (m_i, v_i)) in p_data
            .iter_mut()
            .zip(g_contig.data.iter())
            .zip(m_data.iter_mut().zip(v_data.iter_mut()))
        {
            *m_i = beta1 * *m_i + one_minus_beta1 * *g;
            *v_i = beta2 * *v_i + one_minus_beta2 * (*g * *g);

            let m_hat = *m_i / bias1;
            let v_hat = *v_i / bias2;
            // AdamW: decouple weight decay from gradient — apply directly to parameter
            *p -= lr * (m_hat / (v_hat.sqrt() + epsilon) + weight_decay * *p);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_rmsprop(
    parameters: &mut HashMap<crate::ir::node::ValueId, Tensor>,
    gradients: &HashMap<crate::ir::node::ValueId, Tensor>,
    lr: f32,
    alpha: f32,
    epsilon: f32,
    weight_decay: f32,
    momentum: f32,
    state: &mut OptimizerState,
) -> Result<(), OptimizerError> {
    if !lr.is_finite() || lr <= 0.0 {
        return Err(OptimizerError {
            message: format!("RMSprop learning rate must be a finite positive number, got {lr}"),
        });
    }

    for (value_id, parameter) in parameters {
        let Some(gradient) = gradients.get(value_id) else {
            continue;
        };

        let v = state.rms_v.entry(*value_id).or_insert(
            Tensor::zeros(parameter.shape.clone()).map_err(|e| OptimizerError {
                message: format!("Failed to init RMSprop v: {}", e.message),
            })?,
        );

        let p_data = Arc::make_mut(&mut parameter.data);
        let g_contig = gradient.make_contiguous()?;
        let v_data = Arc::make_mut(&mut v.data);

        if momentum > 0.0 {
            let buf = state.rms_buf.entry(*value_id).or_insert(
                Tensor::zeros(parameter.shape.clone()).map_err(|e| OptimizerError {
                    message: format!("Failed to init RMSprop buf: {}", e.message),
                })?,
            );
            let buf_data = Arc::make_mut(&mut buf.data);

            for ((p, g), (v_i, b_i)) in p_data.iter_mut()
                .zip(g_contig.data.iter())
                .zip(v_data.iter_mut().zip(buf_data.iter_mut()))
            {
                let gwd = *g + weight_decay * *p;
                *v_i = alpha * *v_i + (1.0 - alpha) * gwd * gwd;
                *b_i = momentum * *b_i + lr * gwd / (v_i.sqrt() + epsilon);
                *p -= *b_i;
            }
        } else {
            for ((p, g), v_i) in p_data.iter_mut()
                .zip(g_contig.data.iter())
                .zip(v_data.iter_mut())
            {
                let gwd = *g + weight_decay * *p;
                *v_i = alpha * *v_i + (1.0 - alpha) * gwd * gwd;
                *p -= lr * gwd / (v_i.sqrt() + epsilon);
            }
        }
    }
    Ok(())
}

fn apply_adagrad(
    parameters: &mut HashMap<crate::ir::node::ValueId, Tensor>,
    gradients: &HashMap<crate::ir::node::ValueId, Tensor>,
    lr: f32,
    epsilon: f32,
    weight_decay: f32,
    state: &mut OptimizerState,
) -> Result<(), OptimizerError> {
    if !lr.is_finite() || lr <= 0.0 {
        return Err(OptimizerError {
            message: format!("Adagrad learning rate must be a finite positive number, got {lr}"),
        });
    }

    for (value_id, parameter) in parameters {
        let Some(gradient) = gradients.get(value_id) else {
            continue;
        };

        let acc = state.adagrad_acc.entry(*value_id).or_insert(
            Tensor::zeros(parameter.shape.clone()).map_err(|e| OptimizerError {
                message: format!("Failed to init Adagrad acc: {}", e.message),
            })?,
        );

        let p_data = Arc::make_mut(&mut parameter.data);
        let g_contig = gradient.make_contiguous()?;
        let acc_data = Arc::make_mut(&mut acc.data);

        for ((p, g), acc_i) in p_data.iter_mut()
            .zip(g_contig.data.iter())
            .zip(acc_data.iter_mut())
        {
            let gwd = *g + weight_decay * *p;
            *acc_i += gwd * gwd;
            *p -= lr * gwd / (*acc_i + epsilon).sqrt();
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_lars(
    parameters: &mut HashMap<crate::ir::node::ValueId, Tensor>,
    gradients: &HashMap<crate::ir::node::ValueId, Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    trust_coeff: f32,
    epsilon: f32,
    state: &mut OptimizerState,
) -> Result<(), OptimizerError> {
    if !lr.is_finite() || lr <= 0.0 {
        return Err(OptimizerError {
            message: format!("LARS learning rate must be a finite positive number, got {lr}"),
        });
    }

    for (value_id, parameter) in parameters {
        let Some(gradient) = gradients.get(value_id) else {
            continue;
        };

        // Compute L2 norms for the layer
        let p_norm: f32 = parameter.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let g_norm: f32 = gradient.data.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Layer-wise adaptive learning rate
        let local_lr = if p_norm > 0.0 && g_norm > 0.0 {
            lr * trust_coeff * p_norm / (g_norm + weight_decay * p_norm + epsilon)
        } else {
            lr
        };

        let vel = state.lars_vel.entry(*value_id).or_insert(
            Tensor::zeros(parameter.shape.clone()).map_err(|e| OptimizerError {
                message: format!("Failed to init LARS velocity: {}", e.message),
            })?,
        );

        let p_data = Arc::make_mut(&mut parameter.data);
        let g_contig = gradient.make_contiguous()?;
        let vel_data = Arc::make_mut(&mut vel.data);

        for ((p, g), v) in p_data.iter_mut()
            .zip(g_contig.data.iter())
            .zip(vel_data.iter_mut())
        {
            let update = local_lr * (*g + weight_decay * *p);
            *v = momentum * *v + update;
            *p -= *v;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ir::node::ValueId;
    use crate::ir::optimizer::{OptimizerConfig, OptimizerState, apply_gradients};
    use crate::ir::tensor::Tensor;

    fn vid(n: usize) -> ValueId {
        ValueId(n)
    }

    #[test]
    fn sgd_updates_parameter_values() {
        let mut params = HashMap::new();
        params.insert(
            vid(0),
            Tensor::new(vec![1], vec![1.0]).expect("valid tensor"),
        );

        let mut grads = HashMap::new();
        grads.insert(
            vid(0),
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

        let w = params.get(&vid(0)).expect("param exists");
        assert!((w.data[0] - 0.95).abs() < 1e-6);
    }

    // ── Regression tests for optimizer bugs found in the production code audit ─

    #[test]
    fn sgd_rejects_non_positive_learning_rate() {
        let mut params = HashMap::new();
        params.insert(
            vid(0),
            Tensor::new(vec![1], vec![1.0]).expect("valid tensor"),
        );
        let mut grads = HashMap::new();
        grads.insert(
            vid(0),
            Tensor::new(vec![1], vec![1.0]).expect("valid tensor"),
        );
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
        let mut params = HashMap::new();
        params.insert(
            vid(0),
            Tensor::new(vec![1], vec![1.0]).expect("valid tensor"),
        );
        let mut grads = HashMap::new();
        grads.insert(
            vid(0),
            Tensor::new(vec![1], vec![1.0]).expect("valid tensor"),
        );
        let mut state = OptimizerState::default();

        let err = apply_gradients(
            &mut params,
            &grads,
            &OptimizerConfig::Adam {
                lr: 0.001,
                beta1: 1.0,
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
    fn adagrad_reduces_loss_over_steps() {
        let mut params = HashMap::new();
        params.insert(vid(0), Tensor::new(vec![2], vec![1.0, -1.0]).expect("valid"));
        let mut grads = HashMap::new();
        grads.insert(vid(0), Tensor::new(vec![2], vec![0.5, -0.5]).expect("valid"));
        let mut state = OptimizerState::default();
        let config = OptimizerConfig::Adagrad { lr: 0.1, epsilon: 1e-8, weight_decay: 0.0 };

        // Run 3 steps with same gradient — parameter should monotonically approach 0
        let mut prev_p0 = 1.0f32;
        for _ in 0..3 {
            apply_gradients(&mut params, &grads, &config, &mut state).expect("ok");
            let p0 = params[&vid(0)].data[0];
            assert!(p0 < prev_p0, "param should decrease: {p0} vs {prev_p0}");
            prev_p0 = p0;
        }
    }

    #[test]
    fn adam_rejects_non_positive_epsilon() {
        let mut params = HashMap::new();
        params.insert(
            vid(0),
            Tensor::new(vec![1], vec![1.0]).expect("valid tensor"),
        );
        let mut grads = HashMap::new();
        grads.insert(
            vid(0),
            Tensor::new(vec![1], vec![1.0]).expect("valid tensor"),
        );
        let mut state = OptimizerState::default();

        let err = apply_gradients(
            &mut params,
            &grads,
            &OptimizerConfig::Adam {
                lr: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 0.0,
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
