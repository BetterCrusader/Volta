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
}

fn element_count(shape: &[usize]) -> Option<usize> {
    let mut count = 1usize;
    for dim in shape {
        count = count.checked_mul(*dim)?;
    }
    Some(count)
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
