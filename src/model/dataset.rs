use std::collections::HashMap;

use crate::ir::Tensor;

use crate::model::TrainApiError;

#[derive(Debug, Clone)]
pub struct Example {
    pub inputs: HashMap<String, Tensor>,
}

pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn example(&self, index: usize) -> Result<Example, TrainApiError>;
}

pub struct BatchIterator {
    indices: Vec<usize>,
    position: usize,
    batch_size: usize,
}

impl BatchIterator {
    #[must_use]
    pub fn new(size: usize, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        let mut indices = (0..size).collect::<Vec<_>>();
        if shuffle {
            deterministic_shuffle(&mut indices, seed);
        }
        Self {
            indices,
            position: 0,
            batch_size,
        }
    }
}

impl Iterator for BatchIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }
        let end = (self.position + self.batch_size).min(self.indices.len());
        let batch = self.indices[self.position..end].to_vec();
        self.position = end;
        Some(batch)
    }
}

fn deterministic_shuffle(indices: &mut [usize], seed: u64) {
    let mut state = seed | 1;
    for i in (1..indices.len()).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use crate::model::BatchIterator;

    #[test]
    fn deterministic_shuffle_is_stable() {
        let first = BatchIterator::new(10, 3, true, 42).collect::<Vec<_>>();
        let second = BatchIterator::new(10, 3, true, 42).collect::<Vec<_>>();
        assert_eq!(first, second);
    }
}
