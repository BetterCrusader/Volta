#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TensorError {
    pub message: String,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
        let expected = element_count(&shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        if expected != data.len() {
            return Err(TensorError {
                message: format!(
                    "Tensor shape/data mismatch: shape implies {expected} elements, got {}",
                    data.len()
                ),
            });
        }
        Ok(Self { shape, data })
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let count = element_count(&shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        Ok(Self {
            shape,
            data: vec![0.0; count],
        })
    }

    pub fn scalar(value: f32) -> Self {
        Self {
            shape: vec![1],
            data: vec![value],
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError {
                message: format!(
                    "Shape mismatch in add: {:?} vs {:?}",
                    self.shape, other.shape
                ),
            });
        }
        let mut out = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            out.push(*a + *b);
        }
        Self::new(self.shape.clone(), out)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError {
                message: format!(
                    "Shape mismatch in sub: {:?} vs {:?}",
                    self.shape, other.shape
                ),
            });
        }
        let mut out = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            out.push(*a - *b);
        }
        Self::new(self.shape.clone(), out)
    }

    pub fn mul_elementwise(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError {
                message: format!(
                    "Shape mismatch in mul_elementwise: {:?} vs {:?}",
                    self.shape, other.shape
                ),
            });
        }
        let mut out = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            out.push(*a * *b);
        }
        Self::new(self.shape.clone(), out)
    }

    pub fn scale(&self, factor: f32) -> Result<Self, TensorError> {
        let mut out = Vec::with_capacity(self.data.len());
        for value in &self.data {
            out.push(*value * factor);
        }
        Self::new(self.shape.clone(), out)
    }

    pub fn add_inplace_scaled(&mut self, grad: &Self, scale: f32) -> Result<(), TensorError> {
        if self.shape != grad.shape {
            return Err(TensorError {
                message: format!(
                    "Shape mismatch in add_inplace_scaled: {:?} vs {:?}",
                    self.shape, grad.shape
                ),
            });
        }
        for (value, delta) in self.data.iter_mut().zip(grad.data.iter()) {
            *value += *delta * scale;
        }
        Ok(())
    }

    pub fn transpose_2d(&self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError {
                message: format!("transpose_2d expects rank-2 tensor, got {:?}", self.shape),
            });
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut out = vec![0.0_f32; self.data.len()];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = self.data[r * cols + c];
            }
        }
        Self::new(vec![cols, rows], out)
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError {
                message: format!(
                    "matmul expects rank-2 tensors, got {:?} and {:?}",
                    self.shape, other.shape
                ),
            });
        }
        let m = self.shape[0];
        let k = self.shape[1];
        let k_rhs = other.shape[0];
        let n = other.shape[1];
        if k != k_rhs {
            return Err(TensorError {
                message: format!(
                    "Shape mismatch in matmul: {:?} x {:?}",
                    self.shape, other.shape
                ),
            });
        }
        let mut out = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for t in 0..k {
                    acc += self.data[i * k + t] * other.data[t * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        Self::new(vec![m, n], out)
    }

    pub fn relu(&self) -> Result<Self, TensorError> {
        let mut out = self.data.clone();
        for value in &mut out {
            if *value < 0.0 {
                *value = 0.0;
            }
        }
        Self::new(self.shape.clone(), out)
    }

    pub fn relu_backward(&self, grad_output: &Self) -> Result<Self, TensorError> {
        if self.shape != grad_output.shape {
            return Err(TensorError {
                message: format!(
                    "Shape mismatch in relu_backward: {:?} vs {:?}",
                    self.shape, grad_output.shape
                ),
            });
        }
        let mut out = Vec::with_capacity(self.data.len());
        for (x, g) in self.data.iter().zip(grad_output.data.iter()) {
            out.push(if *x > 0.0 { *g } else { 0.0 });
        }
        Self::new(self.shape.clone(), out)
    }

    pub fn mean(&self) -> Result<Self, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError {
                message: "mean expects non-empty tensor".to_string(),
            });
        }
        let sum = self.data.iter().copied().sum::<f32>();
        let denom = self.data.len() as f32;
        Ok(Self::scalar(sum / denom))
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected = element_count(&shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        if expected != self.data.len() {
            return Err(TensorError {
                message: format!(
                    "Shape mismatch in reshape: {:?} ({} elements) cannot reshape to {:?} ({} elements)",
                    self.shape,
                    self.data.len(),
                    shape,
                    expected
                ),
            });
        }
        Self::new(shape, self.data.clone())
    }

    pub fn concat(tensors: &[Self], axis: usize) -> Result<Self, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError {
                message: "concat expects at least one tensor".to_string(),
            });
        }
        let rank = tensors[0].shape.len();
        if axis >= rank {
            return Err(TensorError {
                message: format!(
                    "concat axis {} out of bounds for rank {} tensor",
                    axis, rank
                ),
            });
        }

        let mut out_shape = tensors[0].shape.clone();
        let mut axis_sum = 0usize;
        for tensor in tensors {
            if tensor.shape.len() != rank {
                return Err(TensorError {
                    message: format!(
                        "concat rank mismatch: expected rank {}, got shape {:?}",
                        rank, tensor.shape
                    ),
                });
            }
            for (dim_idx, (lhs, rhs)) in out_shape.iter().zip(tensor.shape.iter()).enumerate() {
                if dim_idx != axis && lhs != rhs {
                    return Err(TensorError {
                        message: format!(
                            "concat shape mismatch at dim {}: expected {}, got {}",
                            dim_idx, lhs, rhs
                        ),
                    });
                }
            }
            axis_sum = axis_sum
                .checked_add(tensor.shape[axis])
                .ok_or_else(|| TensorError {
                    message: "concat axis size overflow".to_string(),
                })?;
        }
        out_shape[axis] = axis_sum;

        let outer = product_prefix(&out_shape, axis)?;
        let inner = product_suffix(&out_shape, axis + 1)?;
        let mut out = Vec::with_capacity(element_count(&out_shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?);

        for outer_idx in 0..outer {
            for tensor in tensors {
                let axis_dim = tensor.shape[axis];
                let start = outer_idx
                    .checked_mul(axis_dim)
                    .and_then(|v| v.checked_mul(inner))
                    .ok_or_else(|| TensorError {
                        message: "concat index overflow".to_string(),
                    })?;
                let end = start
                    .checked_add(axis_dim.checked_mul(inner).ok_or_else(|| TensorError {
                        message: "concat block size overflow".to_string(),
                    })?)
                    .ok_or_else(|| TensorError {
                        message: "concat index overflow".to_string(),
                    })?;
                out.extend_from_slice(&tensor.data[start..end]);
            }
        }

        Self::new(out_shape, out)
    }

    pub fn gather(&self, indices: &[usize], axis: usize) -> Result<Self, TensorError> {
        let rank = self.shape.len();
        if axis >= rank {
            return Err(TensorError {
                message: format!(
                    "gather axis {} out of bounds for rank {} tensor",
                    axis, rank
                ),
            });
        }
        let axis_dim = self.shape[axis];
        for index in indices {
            if *index >= axis_dim {
                return Err(TensorError {
                    message: format!(
                        "gather index {} out of bounds for axis {} with size {}",
                        index, axis, axis_dim
                    ),
                });
            }
        }

        let mut out_shape = self.shape.clone();
        out_shape[axis] = indices.len();
        let outer = product_prefix(&self.shape, axis)?;
        let inner = product_suffix(&self.shape, axis + 1)?;
        let mut out = Vec::with_capacity(element_count(&out_shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?);

        for outer_idx in 0..outer {
            for index in indices {
                let start = outer_idx
                    .checked_mul(axis_dim)
                    .and_then(|v| v.checked_mul(inner))
                    .and_then(|v| v.checked_add(index.checked_mul(inner)?))
                    .ok_or_else(|| TensorError {
                        message: "gather index overflow".to_string(),
                    })?;
                let end = start.checked_add(inner).ok_or_else(|| TensorError {
                    message: "gather index overflow".to_string(),
                })?;
                out.extend_from_slice(&self.data[start..end]);
            }
        }

        Self::new(out_shape, out)
    }

    pub fn slice(
        &self,
        starts: &[usize],
        ends: &[usize],
        axes: &[usize],
    ) -> Result<Self, TensorError> {
        if starts.is_empty() || ends.is_empty() || axes.is_empty() {
            return Err(TensorError {
                message: "slice starts/ends/axes must be non-empty".to_string(),
            });
        }
        if starts.len() != ends.len() || starts.len() != axes.len() {
            return Err(TensorError {
                message: "slice starts/ends/axes lengths must match".to_string(),
            });
        }

        let rank = self.shape.len();
        let mut out_shape = self.shape.clone();
        let mut axis_seen = std::collections::HashSet::new();
        for idx in 0..axes.len() {
            let axis = axes[idx];
            if axis >= rank {
                return Err(TensorError {
                    message: format!("slice axis {} out of bounds for rank {} tensor", axis, rank),
                });
            }
            if !axis_seen.insert(axis) {
                return Err(TensorError {
                    message: format!("slice axis {} specified more than once", axis),
                });
            }
            let start = starts[idx];
            let end = ends[idx];
            let dim = self.shape[axis];
            if start >= end {
                return Err(TensorError {
                    message: format!(
                        "slice requires start < end per axis, got start={} end={} at axis {}",
                        start, end, axis
                    ),
                });
            }
            if end > dim {
                return Err(TensorError {
                    message: format!(
                        "slice end {} out of bounds for axis {} with size {}",
                        end, axis, dim
                    ),
                });
            }
            out_shape[axis] = end - start;
        }

        let out_count = element_count(&out_shape).ok_or_else(|| TensorError {
            message: "Invalid tensor shape: overflow while computing element count".to_string(),
        })?;
        let out_strides = strides(&out_shape).ok_or_else(|| TensorError {
            message: "slice output stride overflow".to_string(),
        })?;
        let in_strides = strides(&self.shape).ok_or_else(|| TensorError {
            message: "slice input stride overflow".to_string(),
        })?;

        let mut out = vec![0.0_f32; out_count];
        for (linear, out_value) in out.iter_mut().enumerate().take(out_count) {
            let mut rem = linear;
            let mut in_offset = 0usize;
            for dim_idx in 0..out_shape.len() {
                let stride = out_strides[dim_idx];
                let coord = if stride == 0 { 0 } else { rem / stride };
                if stride != 0 {
                    rem %= stride;
                }

                let start_shift = axes
                    .iter()
                    .position(|axis| *axis == dim_idx)
                    .map(|idx| starts[idx])
                    .unwrap_or(0);
                in_offset = in_offset
                    .checked_add(
                        coord
                            .checked_add(start_shift)
                            .and_then(|v| v.checked_mul(in_strides[dim_idx]))
                            .ok_or_else(|| TensorError {
                                message: "slice index overflow".to_string(),
                            })?,
                    )
                    .ok_or_else(|| TensorError {
                        message: "slice index overflow".to_string(),
                    })?;
            }
            *out_value = self.data[in_offset];
        }

        Self::new(out_shape, out)
    }
}

fn element_count(shape: &[usize]) -> Option<usize> {
    let mut count = 1usize;
    for dim in shape {
        count = count.checked_mul(*dim)?;
    }
    Some(count)
}

fn product_prefix(shape: &[usize], end: usize) -> Result<usize, TensorError> {
    shape
        .iter()
        .take(end)
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| TensorError {
            message: "tensor shape overflow while computing prefix product".to_string(),
        })
}

fn product_suffix(shape: &[usize], start: usize) -> Result<usize, TensorError> {
    shape
        .iter()
        .skip(start)
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| TensorError {
            message: "tensor shape overflow while computing suffix product".to_string(),
        })
}

fn strides(shape: &[usize]) -> Option<Vec<usize>> {
    let mut out = vec![0usize; shape.len()];
    let mut stride = 1usize;
    for idx in (0..shape.len()).rev() {
        out[idx] = stride;
        stride = stride.checked_mul(shape[idx])?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn matmul_works_for_small_matrices() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid tensor");
        let b = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).expect("valid tensor");
        let c = a.matmul(&b).expect("matmul should pass");
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
