//! Dense f32 tensors with support for shared storage, arbitrary strides, and offsets.
//!
//! This module provides the core `Tensor` engine for the Volta project.
//! The architecture is designed for efficiency, allowing zero-copy "Views"
//! (like `reshape`, `transpose`, and `slice`) by sharing the underlying `Arc` storage.

use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::engine::ir::kernels::{
    activations, math, norm, reduce, utils,
};

/// Maximum number of elements allowed in a single tensor (512 Mi f32 = 2 GiB).
/// Exceeding this limit returns a `TensorError` instead of attempting allocation.
pub const MAX_TENSOR_ELEMENTS: usize = 1 << 29; // 536_870_912

// ── Core types ────────────────────────────────────────────────────────────────

/// Dense f32 tensor with support for arbitrary strides and offsets (Views).
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Dimension sizes.
    pub shape: Vec<usize>,
    /// Number of elements to skip in storage to move one step along each dimension.
    pub strides: Vec<usize>,
    /// Index of the first element in the underlying storage.
    pub offset: usize,
    /// Shared underlying data buffer.
    pub data: Arc<Vec<f32>>,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape { return false; }

        // Logical comparison following strides.
        let self_contig = self.make_contiguous().unwrap();
        let other_contig = other.make_contiguous().unwrap();
        let len = self.logical_len();
        self_contig.data[..len] == other_contig.data[..len]
    }
}

#[derive(Debug, Clone)]
pub struct TensorError {
    pub message: String,
}

// ── Implementation ────────────────────────────────────────────────────────────

impl Tensor {
    // ── Constructors ──

    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
        let expected = utils::element_count(&shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        if expected > MAX_TENSOR_ELEMENTS {
            return Err(TensorError {
                message: format!(
                    "Tensor too large: {expected} elements exceeds limit of {MAX_TENSOR_ELEMENTS}"
                ),
            });
        }
        if expected != data.len() {
            return Err(TensorError {
                message: format!(
                    "Tensor shape/data mismatch: shape implies {expected} elements, got {}",
                    data.len()
                ),
            });
        }
        let strides = utils::default_strides(&shape);
        Ok(Self {
            shape,
            strides,
            offset: 0,
            data: Arc::new(data),
        })
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let count = utils::element_count(&shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        if count > MAX_TENSOR_ELEMENTS {
            return Err(TensorError {
                message: format!(
                    "Tensor too large: {count} elements exceeds limit of {MAX_TENSOR_ELEMENTS}"
                ),
            });
        }
        Ok(Self {
            shape: shape.clone(),
            strides: utils::default_strides(&shape),
            offset: 0,
            data: Arc::new(vec![0.0; count]),
        })
    }

    pub fn ones(shape: Vec<usize>) -> Result<Self, TensorError> {
        let count = utils::element_count(&shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        if count > MAX_TENSOR_ELEMENTS {
            return Err(TensorError {
                message: format!(
                    "Tensor too large: {count} elements exceeds limit of {MAX_TENSOR_ELEMENTS}"
                ),
            });
        }
        Ok(Self {
            shape: shape.clone(),
            strides: utils::default_strides(&shape),
            offset: 0,
            data: Arc::new(vec![1.0; count]),
        })
    }

    #[must_use]
    pub fn scalar(value: f32) -> Self {
        Self {
            shape: vec![1],
            strides: vec![1],
            offset: 0,
            data: Arc::new(vec![value]),
        }
    }

    // ── Metadata & Utilities ──

    /// Returns the total number of logical elements in this tensor.
    pub fn logical_len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns true if the tensor elements are stored contiguously in memory (C-style)
    /// and the offset is zero (starts at the beginning of the storage).
    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 { return false; }
        if self.shape.is_empty() { return true; }
        let mut acc = 1;
        for i in (0..self.shape.len()).rev() {
            if self.shape[i] > 1 && self.strides[i] != acc {
                return false;
            }
            acc *= self.shape[i];
        }
        true
    }

    /// Returns a contiguous version of this tensor.
    /// If it's already contiguous, it clones the Arc (O(1)).
    /// Otherwise, it performs a deep copy into a new contiguous allocation (O(N)).
    pub fn make_contiguous(&self) -> Result<Self, TensorError> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let total_elements = self.logical_len();
        let mut new_data = vec![0.0; total_elements];
        self.copy_to_slice(&mut new_data);

        Self::new(self.shape.clone(), new_data)
    }

    fn copy_to_slice(&self, dst: &mut [f32]) {
        let mut indices = vec![0; self.shape.len()];
        for i in 0..dst.len() {
            let mut data_idx = self.offset;
            for (dim, &idx) in indices.iter().enumerate() {
                data_idx += idx * self.strides[dim];
            }
            dst[i] = self.data[data_idx];

            // Update indices (row-major)
            for j in (0..self.shape.len()).rev() {
                indices[j] += 1;
                if indices[j] < self.shape[j] {
                    break;
                }
                indices[j] = 0;
            }
        }
    }

    // ── Shape ops (O(1) Views) ──

    /// Zero-copy reshape.
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, TensorError> {
        let new_count = utils::element_count(&shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        if new_count != self.logical_len() {
             return Err(TensorError {
                message: format!(
                    "Shape mismatch in reshape: current size {} vs new size {}",
                    self.logical_len(),
                    new_count
                ),
            });
        }

        if !self.is_contiguous() {
            return self.make_contiguous()?.reshape(shape);
        }

        Ok(Self {
            shape: shape.clone(),
            strides: utils::default_strides(&shape),
            offset: self.offset,
            data: self.data.clone(),
        })
    }

    /// Zero-copy reshape that consumes self.
    pub fn reshape_inplace(self, shape: Vec<usize>) -> Result<Self, TensorError> {
        self.reshape(shape)
    }

    /// O(1) 2D Transpose.
    pub fn transpose_2d(&self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError {
                message: format!("transpose_2d expects rank-2 tensor, got {:?}", self.shape),
            });
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(0, 1);
        new_strides.swap(0, 1);

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            data: self.data.clone(),
        })
    }

    /// O(1) Slice.
    pub fn slice(&self, starts: &[usize], ends: &[usize], axes: &[usize]) -> Result<Self, TensorError> {
        if starts.len() != ends.len() || starts.len() != axes.len() {
            return Err(TensorError {
                message: "slice starts/ends/axes lengths must match".to_string(),
            });
        }

        let mut new_shape = self.shape.clone();
        let mut new_offset = self.offset;

        for i in 0..axes.len() {
            let axis = axes[i];
            let start = starts[i];
            let end = ends[i];

            if axis >= self.shape.len() {
                return Err(TensorError { message: format!("slice axis {axis} out of bounds") });
            }
            if start > end || end > self.shape[axis] {
                return Err(TensorError { message: format!("invalid slice range for axis {axis}") });
            }

            new_offset += start * self.strides[axis];
            new_shape[axis] = end - start;
        }

        Ok(Self {
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
            data: self.data.clone(),
        })
    }

    // ── Kernel Wrappers ──

    pub fn add(&self, other: &Self) -> Result<Self, TensorError> { math::add(self, other) }
    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> { math::sub(self, other) }
    pub fn mul_elementwise(&self, other: &Self) -> Result<Self, TensorError> { math::mul_elementwise(self, other) }
    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> { math::matmul(self, other) }

    pub fn add_broadcast(&self, other: &Self) -> Result<Self, TensorError> { utils::elementwise_broadcast_binary(self, other, "add_broadcast", |a, b| Ok(a + b)) }
    pub fn sub_broadcast(&self, other: &Self) -> Result<Self, TensorError> { utils::elementwise_broadcast_binary(self, other, "sub_broadcast", |a, b| Ok(a - b)) }
    pub fn mul_broadcast(&self, other: &Self) -> Result<Self, TensorError> { utils::elementwise_broadcast_binary(self, other, "mul_broadcast", |a, b| Ok(a * b)) }
    pub fn div_broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        utils::elementwise_broadcast_binary(self, other, "div_broadcast", |a, b| {
            if b == 0.0 { Err(TensorError { message: "Division by zero".to_string() }) } else { Ok(a / b) }
        })
    }

    pub fn scale(&self, factor: f32) -> Result<Self, TensorError> {
        let contig = self.make_contiguous()?;
        let out: Vec<f32> = {
            let data = &contig.data;
            #[cfg(feature = "parallel")]
            {
                if utils::should_par(data.len()) { data.par_iter().map(|v| v * factor).collect() }
                else { data.iter().map(|v| v * factor).collect() }
            }
            #[cfg(not(feature = "parallel"))]
            { data.iter().map(|v| v * factor).collect() }
        };
        Self::new(self.shape.clone(), out)
    }

    pub fn add_inplace_scaled(&mut self, grad: &Self, scale: f32) -> Result<(), TensorError> {
        if self.shape != grad.shape { return Err(TensorError { message: "shape mismatch".to_string() }); }
        if !self.is_contiguous() { *self = self.make_contiguous()?; }
        let data = Arc::make_mut(&mut self.data);
        let grad_contig = grad.make_contiguous()?;
        for (value, delta) in data.iter_mut().zip(grad_contig.data.iter()) { *value += *delta * scale; }
        Ok(())
    }

    pub fn neg_elementwise(&self) -> Result<Self, TensorError> { math::neg_elementwise(self) }
    pub fn relu(&self) -> Result<Self, TensorError> { activations::relu(self) }
    pub fn sigmoid(&self) -> Result<Self, TensorError> { activations::sigmoid(self) }
    pub fn gelu(&self) -> Result<Self, TensorError> { activations::gelu(self) }
    pub fn gelu_exact(&self) -> Result<Self, TensorError> { activations::gelu_exact(self) }

    pub fn log_elementwise(&self) -> Result<Self, TensorError> { math::log_elementwise(self) }
    pub fn exp_elementwise(&self) -> Result<Self, TensorError> { math::exp_elementwise(self) }

    pub fn layer_norm(&self, weight: &Self, bias: &Self, epsilon: f32) -> Result<Self, TensorError> { norm::layer_norm(self, weight, bias, epsilon) }

    pub fn reduce_sum(&self, axis: Option<usize>) -> Result<Self, TensorError> { reduce::reduce_sum(self, axis) }
    pub fn reduce_mean(&self, axis: Option<usize>) -> Result<Self, TensorError> { reduce::reduce_mean(self, axis) }
    pub fn reduce_max(&self, axis: Option<usize>) -> Result<Self, TensorError> { reduce::reduce_max(self, axis) }
    pub fn reduce_sum_keepdims(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        let target = utils::reduce_keepdims_shape(&self.shape, axis)?;
        self.reduce_sum(axis)?.reshape(target)
    }
    pub fn reduce_mean_keepdims(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        let target = utils::reduce_keepdims_shape(&self.shape, axis)?;
        self.reduce_mean(axis)?.reshape(target)
    }
    pub fn reduce_max_keepdims(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        let target = utils::reduce_keepdims_shape(&self.shape, axis)?;
        self.reduce_max(axis)?.reshape(target)
    }

    pub fn softmax(&self) -> Result<Self, TensorError> { reduce::softmax(self) }
    pub fn mean(&self) -> Result<Self, TensorError> {
        let contig = self.make_contiguous()?;
        let sum: f32 = contig.data.iter().copied().sum();
        Ok(Self::scalar(sum / contig.logical_len() as f32))
    }
    pub fn argmax_axis_1(&self) -> Result<Vec<usize>, TensorError> {
        let contig = self.make_contiguous()?;
        if contig.shape.len() != 2 { return Err(TensorError { message: "argmax_axis_1 expects rank-2".to_string() }); }
        let (batch, classes) = (contig.shape[0], contig.shape[1]);
        let mut out = Vec::with_capacity(batch);
        for i in 0..batch {
            let row = &contig.data[i * classes..(i + 1) * classes];
            let (mut max_idx, mut max_val) = (0, row[0]);
            for (j, &val) in row.iter().enumerate().skip(1) { if val > max_val { max_val = val; max_idx = j; } }
            out.push(max_idx);
        }
        Ok(out)
    }

    pub fn gemm(&self, rhs: &Self, bias: Option<&Self>, alpha: f32, beta: f32) -> Result<Self, TensorError> {
        let mut out = self.matmul(rhs)?;
        if (alpha - 1.0).abs() > f32::EPSILON { out = out.scale(alpha)?; }
        if let Some(b) = bias && beta != 0.0 {
            let scaled_bias = if (beta - 1.0).abs() > f32::EPSILON { b.scale(beta)? } else { b.clone() };
            out = out.add_broadcast(&scaled_bias)?;
        }
        Ok(out)
    }

    pub fn concat(tensors: &[Self], axis: usize) -> Result<Self, TensorError> { math::concat(tensors, axis) }
    pub fn gather(&self, indices: &[usize], axis: usize) -> Result<Self, TensorError> { math::gather(self, indices, axis) }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn matmul_works_for_small_matrices() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn reshape_is_zero_copy() {
        let t = Tensor::new(vec![2, 4], vec![1.0; 8]).unwrap();
        let ptr = t.data.as_ptr();
        let r = t.reshape(vec![4, 2]).unwrap();
        assert_eq!(r.shape, vec![4, 2]);
        assert_eq!(r.data.as_ptr(), ptr);
    }

    #[test]
    fn transpose_is_zero_copy() {
        let t = Tensor::new(vec![2, 3], (0..6).map(|i| i as f32).collect()).unwrap();
        let ptr = t.data.as_ptr();
        let tr = t.transpose_2d().unwrap();
        assert_eq!(tr.shape, vec![3, 2]);
        assert_eq!(tr.data.as_ptr(), ptr);
        assert!(!tr.is_contiguous());
        let contig = tr.make_contiguous().unwrap();
        assert_eq!(contig.data.as_slice(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn slice_is_zero_copy() {
        let t = Tensor::new(vec![4, 4], (0..16).map(|i| i as f32).collect()).unwrap();
        let ptr = t.data.as_ptr();
        let s = t.slice(&[1, 1], &[3, 3], &[0, 1]).unwrap();
        assert_eq!(s.shape, vec![2, 2]);
        assert_eq!(s.data.as_ptr(), ptr);
        assert_eq!(s.offset, 5);
        let contig = s.make_contiguous().unwrap();
        assert_eq!(contig.data.as_slice(), &[5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn ops_work_with_views() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let a_tr = a.transpose_2d().unwrap();
        let res = a_tr.add(&b).unwrap();
        assert_eq!(res.data.as_slice(), &[11.0, 23.0, 32.0, 44.0]);
    }

    #[test]
    fn parallel_add_matches_sequential() {
        let a = Tensor::new(vec![1024], (0..1024).map(|i| i as f32).collect()).unwrap();
        let b = Tensor::new(vec![1024], (0..1024).map(|i| i as f32 * 2.0).collect()).unwrap();
        let res = a.add(&b).unwrap();
        for (i, &v) in res.data.iter().enumerate() { assert!((v - i as f32 * 3.0).abs() < 1e-6); }
    }

    #[test]
    fn add_broadcast_extends_rank() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::new(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
        let out = a.add_broadcast(&b).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(*out.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn add_broadcast_two_sided() {
        let a = Tensor::new(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::new(vec![1, 4], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let out = a.add_broadcast(&b).unwrap();
        assert_eq!(out.shape, vec![3, 4]);
        assert_eq!(*out.data, vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0]);
    }

    #[test]
    fn add_broadcast_scalar() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new(vec![1], vec![5.0]).unwrap();
        let out = a.add_broadcast(&b).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(*out.data, vec![6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn gemm_accepts_vector_bias_via_broadcast() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new(vec![2, 3], vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
        let bias = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let out = a.gemm(&b, Some(&bias), 1.0, 1.0).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(*out.data, vec![22.0, 26.0, 30.0, 48.0, 56.0, 64.0]);
    }

    #[test]
    fn reduce_mean_all_elements() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 3.0, 5.0, 7.0]).unwrap();
        let mean = t.reduce_mean(None).unwrap();
        assert_eq!(mean.shape, vec![1]);
        assert_eq!(*mean.data, vec![4.0]);
    }

    #[test]
    fn reduce_mean_over_axis() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 3.0, 5.0, 7.0]).unwrap();
        let mean = t.reduce_mean(Some(0)).unwrap();
        assert_eq!(mean.shape, vec![2]);
        assert_eq!(*mean.data, vec![3.0, 5.0]);
    }

    #[test]
    fn layer_norm_computes_correctly() {
        let x = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let w = Tensor::new(vec![3], vec![2.0, 2.0, 2.0]).unwrap();
        let b = Tensor::new(vec![3], vec![1.0, 1.0, 1.0]).unwrap();
        let out = x.layer_norm(&w, &b, 1e-5).unwrap();
        let expected = vec![-1.449_489_7, 1.0, 3.449_489_8, -1.449_489_7, 1.0, 3.449_489_8];
        assert_eq!(out.shape, vec![2, 3]);
        for (&got, &exp) in out.data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4);
        }
    }
}
