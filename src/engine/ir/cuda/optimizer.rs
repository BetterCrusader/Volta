use std::collections::HashMap;

use crate::ir::cuda::device::CudaDevice;
use crate::ir::cuda::kernels::add::{
    add_f32, add_scalar_f32, div_f32, div_scalar_f32, mul_f32, scale_f32, sqrt_f32, sub_f32,
};
use crate::ir::{DeterminismLevel, OptimizerConfig, OptimizerError, OptimizerState, Tensor};

pub fn apply_gradients_cuda(
    parameters: &mut HashMap<crate::ir::node::ValueId, Tensor>,
    gradients: &HashMap<crate::ir::node::ValueId, Tensor>,
    config: &OptimizerConfig,
    state: &mut OptimizerState,
    determinism: DeterminismLevel,
) -> Result<(), OptimizerError> {
    validate_optimizer_config(config)?;

    let device = CudaDevice::default_shared().map_err(|err| OptimizerError {
        message: format!("CUDA device init failed for optimizer: {}", err.message),
    })?;

    state.step = state.step.saturating_add(1);
    match config {
        OptimizerConfig::Sgd { lr } => {
            apply_sgd_cuda(parameters, gradients, *lr, &device, determinism)
        }
        OptimizerConfig::Adam {
            lr,
            beta1,
            beta2,
            epsilon,
        } => apply_adam_cuda(
            parameters,
            gradients,
            *lr,
            *beta1,
            *beta2,
            *epsilon,
            state,
            &device,
            determinism,
        ),
    }
}

fn apply_sgd_cuda(
    parameters: &mut HashMap<crate::ir::node::ValueId, Tensor>,
    gradients: &HashMap<crate::ir::node::ValueId, Tensor>,
    lr: f32,
    device: &CudaDevice,
    determinism: DeterminismLevel,
) -> Result<(), OptimizerError> {
    for (value_id, parameter) in parameters {
        let Some(gradient) = gradients.get(value_id) else {
            continue;
        };
        if parameter.shape != gradient.shape {
            return Err(OptimizerError {
                message: format!(
                    "Shape mismatch in CUDA SGD for '{value_id}': {:?} vs {:?}",
                    parameter.shape, gradient.shape
                ),
            });
        }

        let scaled = scale_f32(device, &gradient.data, lr, determinism).map_err(|message| {
            OptimizerError {
                message: format!("CUDA SGD scale kernel failed for '{value_id}': {message}"),
            }
        })?;
        parameter.data = sub_f32(device, &parameter.data, &scaled, determinism)
            .map_err(|message| OptimizerError {
                message: format!("CUDA SGD update kernel failed for '{value_id}': {message}"),
            })?
            .into();
    }
    Ok(())
}

fn validate_optimizer_config(config: &OptimizerConfig) -> Result<(), OptimizerError> {
    match config {
        OptimizerConfig::Sgd { lr } => validate_sgd_lr(*lr),
        OptimizerConfig::Adam {
            lr,
            beta1,
            beta2,
            epsilon,
        } => validate_adam_hparams(*lr, *beta1, *beta2, *epsilon),
    }
}

fn validate_sgd_lr(lr: f32) -> Result<(), OptimizerError> {
    if !lr.is_finite() || lr <= 0.0 {
        return Err(OptimizerError {
            message: format!("CUDA SGD learning rate must be a finite positive number, got {lr}"),
        });
    }
    Ok(())
}

fn validate_adam_hparams(
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) -> Result<(), OptimizerError> {
    if !lr.is_finite() || lr <= 0.0 {
        return Err(OptimizerError {
            message: format!("CUDA Adam learning rate must be a finite positive number, got {lr}"),
        });
    }
    if !(0.0..1.0).contains(&beta1) || !beta1.is_finite() {
        return Err(OptimizerError {
            message: format!("CUDA Adam beta1 must be in [0, 1), got {beta1}"),
        });
    }
    if !(0.0..1.0).contains(&beta2) || !beta2.is_finite() {
        return Err(OptimizerError {
            message: format!("CUDA Adam beta2 must be in [0, 1), got {beta2}"),
        });
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimizerError {
            message: format!("CUDA Adam epsilon must be a finite positive number, got {epsilon}"),
        });
    }
    Ok(())
}

fn adam_bias_terms(step: usize, beta1: f32, beta2: f32) -> Result<(f32, f32), OptimizerError> {
    let step_i32 = i32::try_from(step).map_err(|_| OptimizerError {
        message: "Optimizer step overflow for Adam bias correction".to_string(),
    })?;
    let bias1 = 1.0_f32 - beta1.powi(step_i32);
    let bias2 = 1.0_f32 - beta2.powi(step_i32);
    if bias1 <= 0.0 || bias2 <= 0.0 {
        return Err(OptimizerError {
            message: format!(
                "CUDA Adam bias correction underflowed (bias1={bias1}, bias2={bias2}); \
                 step={step} with beta1={beta1}, beta2={beta2}"
            ),
        });
    }
    Ok((bias1, bias2))
}

#[allow(clippy::too_many_arguments)]
fn apply_adam_cuda(
    parameters: &mut HashMap<crate::ir::node::ValueId, Tensor>,
    gradients: &HashMap<crate::ir::node::ValueId, Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    state: &mut OptimizerState,
    device: &CudaDevice,
    determinism: DeterminismLevel,
) -> Result<(), OptimizerError> {
    let one_minus_beta1 = 1.0_f32 - beta1;
    let one_minus_beta2 = 1.0_f32 - beta2;
    let (bias1, bias2) = adam_bias_terms(state.step, beta1, beta2)?;

    for (value_id, parameter) in parameters {
        let Some(gradient) = gradients.get(value_id) else {
            continue;
        };
        if parameter.shape != gradient.shape {
            return Err(OptimizerError {
                message: format!(
                    "Shape mismatch in CUDA Adam for '{value_id}': {:?} vs {:?}",
                    parameter.shape, gradient.shape
                ),
            });
        }

        let m = state.adam_m.entry(*value_id).or_insert(
            Tensor::zeros(parameter.shape.clone()).map_err(|err| OptimizerError {
                message: format!(
                    "Failed to initialize CUDA Adam m buffer for '{value_id}': {}",
                    err.message
                ),
            })?,
        );
        let v = state.adam_v.entry(*value_id).or_insert(
            Tensor::zeros(parameter.shape.clone()).map_err(|err| OptimizerError {
                message: format!(
                    "Failed to initialize CUDA Adam v buffer for '{value_id}': {}",
                    err.message
                ),
            })?,
        );

        let m_beta =
            scale_f32(device, &m.data, beta1, determinism).map_err(|message| OptimizerError {
                message: format!("CUDA Adam m*beta1 kernel failed for '{value_id}': {message}"),
            })?;
        let g_beta =
            scale_f32(device, &gradient.data, one_minus_beta1, determinism).map_err(|message| {
                OptimizerError {
                    message: format!(
                        "CUDA Adam g*(1-beta1) kernel failed for '{value_id}': {message}"
                    ),
                }
            })?;
        m.data = add_tensors(
            device,
            &m_beta,
            &g_beta,
            determinism,
            *value_id,
            "adam-m update",
        )?
        .into();

        let v_beta =
            scale_f32(device, &v.data, beta2, determinism).map_err(|message| OptimizerError {
                message: format!("CUDA Adam v*beta2 kernel failed for '{value_id}': {message}"),
            })?;
        let g_sq = mul_tensors(
            device,
            &gradient.data,
            &gradient.data,
            determinism,
            *value_id,
            "adam g^2",
        )?;
        let g_sq_beta =
            scale_f32(device, &g_sq, one_minus_beta2, determinism).map_err(|message| {
                OptimizerError {
                    message: format!(
                        "CUDA Adam g^2*(1-beta2) kernel failed for '{value_id}': {message}"
                    ),
                }
            })?;
        v.data = add_tensors(
            device,
            &v_beta,
            &g_sq_beta,
            determinism,
            *value_id,
            "adam-v update",
        )?
        .into();

        let m_hat = div_scalar_f32(device, &m.data, bias1, determinism).map_err(|message| {
            OptimizerError {
                message: format!("CUDA Adam m_hat kernel failed for '{value_id}': {message}"),
            }
        })?;
        let v_hat = div_scalar_f32(device, &v.data, bias2, determinism).map_err(|message| {
            OptimizerError {
                message: format!("CUDA Adam v_hat kernel failed for '{value_id}': {message}"),
            }
        })?;
        let denom = add_scalar_f32(
            device,
            &sqrt_f32(device, &v_hat, determinism).map_err(|message| OptimizerError {
                message: format!("CUDA Adam sqrt kernel failed for '{value_id}': {message}"),
            })?,
            epsilon,
            determinism,
        )
        .map_err(|message| OptimizerError {
            message: format!("CUDA Adam denominator kernel failed for '{value_id}': {message}"),
        })?;
        let ratio = div_tensors(device, &m_hat, &denom, determinism, *value_id, "adam ratio")?;
        let step =
            scale_f32(device, &ratio, lr, determinism).map_err(|message| OptimizerError {
                message: format!("CUDA Adam lr*ratio kernel failed for '{value_id}': {message}"),
            })?;
        parameter.data = sub_tensors(
            device,
            &parameter.data,
            &step,
            determinism,
            *value_id,
            "adam update",
        )?
        .into();
    }

    Ok(())
}

fn add_tensors(
    device: &CudaDevice,
    left: &[f32],
    right: &[f32],
    determinism: DeterminismLevel,
    name: crate::ir::node::ValueId,
    stage: &str,
) -> Result<Vec<f32>, OptimizerError> {
    add_f32(device, left, right, determinism).map_err(|message| OptimizerError {
        message: format!("CUDA {stage} failed for '{name}': {message}"),
    })
}

fn sub_tensors(
    device: &CudaDevice,
    left: &[f32],
    right: &[f32],
    determinism: DeterminismLevel,
    name: crate::ir::node::ValueId,
    stage: &str,
) -> Result<Vec<f32>, OptimizerError> {
    sub_f32(device, left, right, determinism).map_err(|message| OptimizerError {
        message: format!("CUDA {stage} failed for '{name}': {message}"),
    })
}

fn mul_tensors(
    device: &CudaDevice,
    left: &[f32],
    right: &[f32],
    determinism: DeterminismLevel,
    name: crate::ir::node::ValueId,
    stage: &str,
) -> Result<Vec<f32>, OptimizerError> {
    mul_f32(device, left, right, determinism).map_err(|message| OptimizerError {
        message: format!("CUDA {stage} failed for '{name}': {message}"),
    })
}

fn div_tensors(
    device: &CudaDevice,
    left: &[f32],
    right: &[f32],
    determinism: DeterminismLevel,
    name: crate::ir::node::ValueId,
    stage: &str,
) -> Result<Vec<f32>, OptimizerError> {
    div_f32(device, left, right, determinism).map_err(|message| OptimizerError {
        message: format!("CUDA {stage} failed for '{name}': {message}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_positive_sgd_lr() {
        let err = validate_sgd_lr(0.0).expect_err("lr=0 must fail");
        assert!(err.message.contains("finite positive"));
    }

    #[test]
    fn rejects_invalid_adam_hparams() {
        let err = validate_adam_hparams(0.001, 1.0, 0.999, 1e-8).expect_err("beta1=1 must fail");
        assert!(err.message.contains("beta1"));

        let err = validate_adam_hparams(0.001, 0.9, 0.999, 0.0).expect_err("epsilon<=0 must fail");
        assert!(err.message.contains("epsilon"));
    }

    #[test]
    fn rejects_adam_bias_underflow() {
        let err = adam_bias_terms(1, 1.0, 0.999).expect_err("beta1=1 makes bias1 zero");
        assert!(err.message.contains("bias correction underflowed"));
    }
}
