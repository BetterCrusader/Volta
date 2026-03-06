#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorShape(pub Vec<usize>);

impl TensorShape {
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn element_count(&self) -> Option<usize> {
        self.0
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
    }

    #[must_use]
    pub fn matmul_output(&self, rhs: &Self) -> Option<Self> {
        if self.rank() != 2 || rhs.rank() != 2 {
            return None;
        }
        if self.0[1] != rhs.0[0] {
            return None;
        }
        Some(Self(vec![self.0[0], rhs.0[1]]))
    }

    #[must_use]
    pub fn conv2d_output(&self, kernel: &Self) -> Option<Self> {
        if self.rank() != 2 || kernel.rank() != 2 {
            return None;
        }
        if kernel.0[0] == 0 || kernel.0[1] == 0 {
            return None;
        }
        if kernel.0[0] > self.0[0] || kernel.0[1] > self.0[1] {
            return None;
        }
        Some(Self(vec![
            self.0[0] - kernel.0[0] + 1,
            self.0[1] - kernel.0[1] + 1,
        ]))
    }
}
