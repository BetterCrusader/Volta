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
        parameter.data =
            sub_f32(device, &parameter.data, &scaled, determinism).map_err(|message| {
                OptimizerError {
                    message: format!("CUDA SGD update kernel failed for '{value_id}': {message}"),
                }
            })?;
    }
    Ok(())
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
    let step_i32 = i32::try_from(state.step).map_err(|_| OptimizerError {
        message: "Optimizer step overflow for Adam bias correction".to_string(),
    })?;
    let one_minus_beta1 = 1.0_f32 - beta1;
    let one_minus_beta2 = 1.0_f32 - beta2;
    let bias1 = 1.0_f32 - beta1.powi(step_i32);
    let bias2 = 1.0_f32 - beta2.powi(step_i32);

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
        )?;

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
        )?;

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
        )?;
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
