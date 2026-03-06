use crate::engine::ir::tensor::{Tensor, TensorError};
use crate::engine::ir::op::ElementwiseUnaryOp;
use crate::engine::ir::kernels::utils::erf_approx;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn relu(input: &Tensor) -> Result<Tensor, TensorError> {
    let contig = input.make_contiguous()?;
    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if crate::engine::ir::kernels::utils::should_par(contig.data.len()) {
            contig.data.par_iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
        } else {
            contig.data.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
        }
        #[cfg(not(feature = "parallel"))]
        contig.data.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
    };
    Tensor::new(input.shape.clone(), out)
}

pub fn relu_backward(input: &Tensor, grad_output: &Tensor) -> Result<Tensor, TensorError> {
    if input.shape != grad_output.shape {
        return Err(TensorError { message: "Shape mismatch in relu_backward".to_string() });
    }
    let left = input.make_contiguous()?;
    let right = grad_output.make_contiguous()?;
    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if crate::engine::ir::kernels::utils::should_par(left.data.len()) {
            left.data.par_iter().zip(right.data.par_iter()).map(|(&x, &g)| if x > 0.0 { g } else { 0.0 }).collect()
        } else {
            left.data.iter().zip(right.data.iter()).map(|(&x, &g)| if x > 0.0 { g } else { 0.0 }).collect()
        }
        #[cfg(not(feature = "parallel"))]
        left.data.iter().zip(right.data.iter()).map(|(&x, &g)| if x > 0.0 { g } else { 0.0 }).collect()
    };
    Tensor::new(input.shape.clone(), out)
}

pub fn sigmoid(input: &Tensor) -> Result<Tensor, TensorError> {
    let contig = input.make_contiguous()?;
    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if crate::engine::ir::kernels::utils::should_par(contig.data.len()) {
            contig.data.par_iter().map(|&x| sigmoid_fn(x)).collect()
        } else {
            contig.data.iter().map(|&x| sigmoid_fn(x)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        contig.data.iter().map(|&x| sigmoid_fn(x)).collect()
    };
    Tensor::new(input.shape.clone(), out)
}

#[inline(always)]
fn sigmoid_fn(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let ex = x.exp(); ex / (1.0 + ex) }
}

pub fn sigmoid_backward(input: &Tensor, grad_output: &Tensor) -> Result<Tensor, TensorError> {
    if input.shape != grad_output.shape {
        return Err(TensorError { message: "Shape mismatch in sigmoid_backward".to_string() });
    }
    let left = input.make_contiguous()?;
    let go = grad_output.make_contiguous()?;
    let f = |(&x, &g): (&f32, &f32)| -> f32 {
        let sig = sigmoid_fn(x);
        g * sig * (1.0 - sig)
    };
    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if crate::engine::ir::kernels::utils::should_par(left.data.len()) {
            left.data.par_iter().zip(go.data.par_iter()).map(f).collect()
        } else {
            left.data.iter().zip(go.data.iter()).map(f).collect()
        }
        #[cfg(not(feature = "parallel"))]
        left.data.iter().zip(go.data.iter()).map(f).collect()
    };
    Tensor::new(input.shape.clone(), out)
}

pub fn gelu(input: &Tensor) -> Result<Tensor, TensorError> {
    let contig = input.make_contiguous()?;
    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if crate::engine::ir::kernels::utils::should_par(contig.data.len()) {
            contig.data.par_iter().map(|&x| gelu_fn(x)).collect()
        } else {
            contig.data.iter().map(|&x| gelu_fn(x)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        contig.data.iter().map(|&x| gelu_fn(x)).collect()
    };
    Tensor::new(input.shape.clone(), out)
}

#[inline(always)]
fn gelu_fn(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    let tval = inner.tanh();
    x * 0.5 * (1.0 + tval)
}

pub fn gelu_exact(input: &Tensor) -> Result<Tensor, TensorError> {
    let contig = input.make_contiguous()?;
    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if crate::engine::ir::kernels::utils::should_par(contig.data.len()) {
            contig.data.par_iter().map(|&x| gelu_exact_fn(x)).collect()
        } else {
            contig.data.iter().map(|&x| gelu_exact_fn(x)).collect()
        }
        #[cfg(not(feature = "parallel"))]
        contig.data.iter().map(|&x| gelu_exact_fn(x)).collect()
    };
    Tensor::new(input.shape.clone(), out)
}

#[inline(always)]
fn gelu_exact_fn(x: f32) -> f32 {
    const INV_SQRT_2: f32 = core::f32::consts::FRAC_1_SQRT_2;
    let z = x * INV_SQRT_2;
    let erf = erf_approx(z);
    0.5 * x * (1.0 + erf)
}

pub fn gelu_backward(input: &Tensor, grad_output: &Tensor) -> Result<Tensor, TensorError> {
    if input.shape != grad_output.shape {
        return Err(TensorError { message: "Shape mismatch in gelu_backward".to_string() });
    }
    let left = input.make_contiguous()?;
    let go = grad_output.make_contiguous()?;
    let f = |(&x, &g): (&f32, &f32)| -> f32 {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const GELU_COEFF: f32 = 0.044_715;
        const GELU_COEFF_3: f32 = 3.0 * GELU_COEFF;
        let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        let tval = inner.tanh();
        let dtval = 1.0 - tval * tval;
        let d_inner = SQRT_2_OVER_PI * (1.0 + GELU_COEFF_3 * x * x);
        g * (0.5 * (1.0 + tval) + x * 0.5 * dtval * d_inner)
    };
    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if crate::engine::ir::kernels::utils::should_par(left.data.len()) {
            left.data.par_iter().zip(go.data.par_iter()).map(f).collect()
        } else {
            left.data.iter().zip(go.data.iter()).map(f).collect()
        }
        #[cfg(not(feature = "parallel"))]
        left.data.iter().zip(go.data.iter()).map(f).collect()
    };
    Tensor::new(input.shape.clone(), out)
}

pub fn gelu_exact_backward(input: &Tensor, grad_output: &Tensor) -> Result<Tensor, TensorError> {
    if input.shape != grad_output.shape {
        return Err(TensorError { message: "Shape mismatch in gelu_exact_backward".to_string() });
    }
    let left = input.make_contiguous()?;
    let go = grad_output.make_contiguous()?;
    let f = |(&x, &g): (&f32, &f32)| -> f32 {
        const INV_SQRT_2: f32 = core::f32::consts::FRAC_1_SQRT_2;
        const INV_SQRT_2PI: f32 = 0.398_942_3;
        let z = x * INV_SQRT_2;
        let erf = erf_approx(z);
        let d_erf = INV_SQRT_2PI * (-0.5 * x * x).exp();
        g * (0.5 * (1.0 + erf) + x * d_erf)
    };
    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if crate::engine::ir::kernels::utils::should_par(left.data.len()) {
            left.data.par_iter().zip(go.data.par_iter()).map(f).collect()
        } else {
            left.data.iter().zip(go.data.iter()).map(f).collect()
        }
        #[cfg(not(feature = "parallel"))]
        left.data.iter().zip(go.data.iter()).map(f).collect()
    };
    Tensor::new(input.shape.clone(), out)
}

pub fn apply_elementwise_chain(input: &Tensor, ops: &[ElementwiseUnaryOp]) -> Result<Tensor, TensorError> {
    let contig = input.make_contiguous()?;
    let f = |&x: &f32| -> f32 {
        let mut val = x;
        for op in ops {
            val = match op {
                ElementwiseUnaryOp::Neg => -val,
                ElementwiseUnaryOp::Relu => if val > 0.0 { val } else { 0.0 },
                ElementwiseUnaryOp::LeakyRelu(alpha) => if val > 0.0 { val } else { val * alpha },
                ElementwiseUnaryOp::Sigmoid => sigmoid_fn(val),
                ElementwiseUnaryOp::Gelu => gelu_fn(val),
                ElementwiseUnaryOp::GeluExact => gelu_exact_fn(val),
                ElementwiseUnaryOp::Exp => val.exp(),
                ElementwiseUnaryOp::Log => val.ln(),
            };
        }
        val
    };
    let out: Vec<f32> = {
        #[cfg(feature = "parallel")]
        if crate::engine::ir::kernels::utils::should_par(contig.data.len()) {
            contig.data.par_iter().map(f).collect()
        } else {
            contig.data.iter().map(f).collect()
        }
        #[cfg(not(feature = "parallel"))]
        contig.data.iter().map(f).collect()
    };
    Tensor::new(input.shape.clone(), out)
}
