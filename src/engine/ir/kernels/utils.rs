use crate::engine::ir::tensor::{Tensor, TensorError};

/// Minimum element count before spawning Rayon threads for elementwise ops.
/// Below this threshold, single-threaded SIMD is faster due to thread sync overhead.
/// Rule of thumb: 0.5ms thread overhead / (16 GB/s single-thread / 4 bytes per f32) ≈ 2M elements.
/// We use 512K as a conservative threshold (still saves ~95% of Rayon overhead).
#[cfg(feature = "parallel")]
pub const PARALLEL_THRESHOLD: usize = 1 << 19; // 524_288 elements (~2MB)

#[inline(always)]
pub fn should_par(n: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        n >= PARALLEL_THRESHOLD
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = n;
        false
    }
}

pub fn broadcast_shape(
    left: &[usize],
    right: &[usize],
    op: &str,
) -> Result<Vec<usize>, TensorError> {
    let max_len = left.len().max(right.len());
    let mut out = vec![0; max_len];
    for i in 0..max_len {
        let l = if i < max_len - left.len() {
            1
        } else {
            left[i - (max_len - left.len())]
        };
        let r = if i < max_len - right.len() {
            1
        } else {
            right[i - (max_len - right.len())]
        };
        out[i] = if l == r {
            l
        } else if l == 1 {
            r
        } else if r == 1 {
            l
        } else {
            return Err(TensorError {
                message: format!("Cannot broadcast {left:?} and {right:?} in {op}"),
            });
        };
    }
    Ok(out)
}

pub fn broadcast_index(
    linear: usize,
    out_shape: &[usize],
    out_strides: &[usize],
    in_shape: &[usize],
    in_strides: &[usize],
) -> usize {
    let offset = out_shape.len() - in_shape.len();
    let (mut rem, mut input_linear) = (linear, 0);
    for (axis, &stride) in out_strides.iter().enumerate() {
        let coord = if stride == 0 { 0 } else { rem / stride };
        if stride != 0 {
            rem %= stride;
        }
        if axis >= offset {
            let in_axis = axis - offset;
            if in_shape[in_axis] != 1 {
                input_linear += coord * in_strides[in_axis];
            }
        }
    }
    input_linear
}

pub fn elementwise_broadcast_binary<F>(
    left: &Tensor,
    right: &Tensor,
    op: &str,
    f: F,
) -> Result<Tensor, TensorError>
where
    F: Fn(f32, f32) -> Result<f32, TensorError> + Sync + Send,
{
    let (l_c, r_c) = (left.make_contiguous()?, right.make_contiguous()?);
    let out_shape = broadcast_shape(&l_c.shape, &r_c.shape, op)?;
    let count: usize = out_shape.iter().product();
    if count == 0 {
        return Tensor::new(out_shape, vec![]);
    }

    let o_s = default_strides(&out_shape);
    let l_s = default_strides(&l_c.shape);
    let r_s = default_strides(&r_c.shape);

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let l_idx = broadcast_index(i, &out_shape, &o_s, &l_c.shape, &l_s);
        let r_idx = broadcast_index(i, &out_shape, &o_s, &r_c.shape, &r_s);
        out.push(f(l_c.data[l_idx], r_c.data[r_idx])?);
    }
    Tensor::new(out_shape, out)
}

pub fn default_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

pub fn reduce_keepdims_shape(
    shape: &[usize],
    axis: Option<usize>,
) -> Result<Vec<usize>, TensorError> {
    match axis {
        Some(a) => {
            if a >= shape.len() {
                return Err(TensorError {
                    message: "out of bounds".to_string(),
                });
            }
            let mut out = shape.to_vec();
            out[a] = 1;
            Ok(out)
        }
        None => Ok(if shape.is_empty() {
            vec![1]
        } else {
            vec![1; shape.len()]
        }),
    }
}

pub fn ensure_same_shape(a: &[usize], b: &[usize], op: &str) -> Result<(), TensorError> {
    if a == b {
        Ok(())
    } else {
        Err(TensorError {
            message: format!("Shape mismatch in {op}: {a:?} vs {b:?}"),
        })
    }
}

pub fn element_count(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

pub fn product_prefix(shape: &[usize], end: usize) -> Result<usize, TensorError> {
    shape
        .iter()
        .take(end)
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| TensorError {
            message: "overflow".to_string(),
        })
}

pub fn product_suffix(shape: &[usize], start: usize) -> Result<usize, TensorError> {
    shape
        .iter()
        .skip(start)
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| TensorError {
            message: "overflow".to_string(),
        })
}

pub fn erf_approx(x: f32) -> f32 {
    const A: [f32; 5] = [
        0.254_829_6,
        -0.284_496_72,
        1.421_413_8,
        -1.453_152_1,
        1.061_405_4,
    ];
    let (sign, xa) = (if x < 0.0 { -1.0 } else { 1.0 }, x.abs());
    let t = 1.0 / (1.0 + 0.327_591_1 * xa);
    let poly = ((((A[4] * t + A[3]) * t + A[2]) * t + A[1]) * t + A[0]) * t * (-(xa * xa)).exp();
    sign * (1.0 - poly)
}
