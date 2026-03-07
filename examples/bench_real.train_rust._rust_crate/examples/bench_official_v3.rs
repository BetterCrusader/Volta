// bench_official_v3.rs — AVX-512 fused bias+relu, relu_mask, transpose
// AVX-512F: 16 floats/cycle instead of 8 (AVX2)
// Run: cargo run --release --example bench_official_v3
#![allow(non_snake_case)]
use std::time::Instant;

// AVX-512: transpose 16×16 block — twice the throughput of 8×8 AVX2
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[target_feature(enable = "avx512f")]
unsafe fn transpose_16x16_avx512(
    dst: *mut f32,
    dst_rows: usize,
    src: *const f32,
    src_cols: usize,
    bi: usize,
    bj: usize,
) {
    use std::arch::x86_64::*;
    // load 16 rows × 16 cols
    let r0 = _mm512_loadu_ps(src.add((bi) * src_cols + bj));
    let r1 = _mm512_loadu_ps(src.add((bi + 1) * src_cols + bj));
    let r2 = _mm512_loadu_ps(src.add((bi + 2) * src_cols + bj));
    let r3 = _mm512_loadu_ps(src.add((bi + 3) * src_cols + bj));
    let r4 = _mm512_loadu_ps(src.add((bi + 4) * src_cols + bj));
    let r5 = _mm512_loadu_ps(src.add((bi + 5) * src_cols + bj));
    let r6 = _mm512_loadu_ps(src.add((bi + 6) * src_cols + bj));
    let r7 = _mm512_loadu_ps(src.add((bi + 7) * src_cols + bj));
    let r8 = _mm512_loadu_ps(src.add((bi + 8) * src_cols + bj));
    let r9 = _mm512_loadu_ps(src.add((bi + 9) * src_cols + bj));
    let r10 = _mm512_loadu_ps(src.add((bi + 10) * src_cols + bj));
    let r11 = _mm512_loadu_ps(src.add((bi + 11) * src_cols + bj));
    let r12 = _mm512_loadu_ps(src.add((bi + 12) * src_cols + bj));
    let r13 = _mm512_loadu_ps(src.add((bi + 13) * src_cols + bj));
    let r14 = _mm512_loadu_ps(src.add((bi + 14) * src_cols + bj));
    let r15 = _mm512_loadu_ps(src.add((bi + 15) * src_cols + bj));
    // unpack lo/hi pairs
    let t0 = _mm512_unpacklo_ps(r0, r1);
    let t1 = _mm512_unpackhi_ps(r0, r1);
    let t2 = _mm512_unpacklo_ps(r2, r3);
    let t3 = _mm512_unpackhi_ps(r2, r3);
    let t4 = _mm512_unpacklo_ps(r4, r5);
    let t5 = _mm512_unpackhi_ps(r4, r5);
    let t6 = _mm512_unpacklo_ps(r6, r7);
    let t7 = _mm512_unpackhi_ps(r6, r7);
    let t8 = _mm512_unpacklo_ps(r8, r9);
    let t9 = _mm512_unpackhi_ps(r8, r9);
    let t10 = _mm512_unpacklo_ps(r10, r11);
    let t11 = _mm512_unpackhi_ps(r10, r11);
    let t12 = _mm512_unpacklo_ps(r12, r13);
    let t13 = _mm512_unpackhi_ps(r12, r13);
    let t14 = _mm512_unpacklo_ps(r14, r15);
    let t15 = _mm512_unpackhi_ps(r14, r15);
    // shuffle into 4-wide groups
    let s0 = _mm512_shuffle_ps(t0, t2, 0x44);
    let s1 = _mm512_shuffle_ps(t0, t2, 0xEE);
    let s2 = _mm512_shuffle_ps(t1, t3, 0x44);
    let s3 = _mm512_shuffle_ps(t1, t3, 0xEE);
    let s4 = _mm512_shuffle_ps(t4, t6, 0x44);
    let s5 = _mm512_shuffle_ps(t4, t6, 0xEE);
    let s6 = _mm512_shuffle_ps(t5, t7, 0x44);
    let s7 = _mm512_shuffle_ps(t5, t7, 0xEE);
    let s8 = _mm512_shuffle_ps(t8, t10, 0x44);
    let s9 = _mm512_shuffle_ps(t8, t10, 0xEE);
    let s10 = _mm512_shuffle_ps(t9, t11, 0x44);
    let s11 = _mm512_shuffle_ps(t9, t11, 0xEE);
    let s12 = _mm512_shuffle_ps(t12, t14, 0x44);
    let s13 = _mm512_shuffle_ps(t12, t14, 0xEE);
    let s14 = _mm512_shuffle_ps(t13, t15, 0x44);
    let s15 = _mm512_shuffle_ps(t13, t15, 0xEE);
    // permute 128-bit lanes to get 8-wide groups
    let u0 = _mm512_shuffle_f32x4(s0, s4, 0x44);
    let u1 = _mm512_shuffle_f32x4(s1, s5, 0x44);
    let u2 = _mm512_shuffle_f32x4(s2, s6, 0x44);
    let u3 = _mm512_shuffle_f32x4(s3, s7, 0x44);
    let u4 = _mm512_shuffle_f32x4(s0, s4, 0xEE);
    let u5 = _mm512_shuffle_f32x4(s1, s5, 0xEE);
    let u6 = _mm512_shuffle_f32x4(s2, s6, 0xEE);
    let u7 = _mm512_shuffle_f32x4(s3, s7, 0xEE);
    let u8 = _mm512_shuffle_f32x4(s8, s12, 0x44);
    let u9 = _mm512_shuffle_f32x4(s9, s13, 0x44);
    let u10 = _mm512_shuffle_f32x4(s10, s14, 0x44);
    let u11 = _mm512_shuffle_f32x4(s11, s15, 0x44);
    let u12 = _mm512_shuffle_f32x4(s8, s12, 0xEE);
    let u13 = _mm512_shuffle_f32x4(s9, s13, 0xEE);
    let u14 = _mm512_shuffle_f32x4(s10, s14, 0xEE);
    let u15 = _mm512_shuffle_f32x4(s11, s15, 0xEE);
    // final 256-bit lane permute to get full 16-wide rows
    let o0 = _mm512_shuffle_f32x4(u0, u8, 0x44);
    let o1 = _mm512_shuffle_f32x4(u1, u9, 0x44);
    let o2 = _mm512_shuffle_f32x4(u2, u10, 0x44);
    let o3 = _mm512_shuffle_f32x4(u3, u11, 0x44);
    let o4 = _mm512_shuffle_f32x4(u4, u12, 0x44);
    let o5 = _mm512_shuffle_f32x4(u5, u13, 0x44);
    let o6 = _mm512_shuffle_f32x4(u6, u14, 0x44);
    let o7 = _mm512_shuffle_f32x4(u7, u15, 0x44);
    let o8 = _mm512_shuffle_f32x4(u0, u8, 0xEE);
    let o9 = _mm512_shuffle_f32x4(u1, u9, 0xEE);
    let o10 = _mm512_shuffle_f32x4(u2, u10, 0xEE);
    let o11 = _mm512_shuffle_f32x4(u3, u11, 0xEE);
    let o12 = _mm512_shuffle_f32x4(u4, u12, 0xEE);
    let o13 = _mm512_shuffle_f32x4(u5, u13, 0xEE);
    let o14 = _mm512_shuffle_f32x4(u6, u14, 0xEE);
    let o15 = _mm512_shuffle_f32x4(u7, u15, 0xEE);
    _mm512_storeu_ps(dst.add((bj) * dst_rows + bi), o0);
    _mm512_storeu_ps(dst.add((bj + 1) * dst_rows + bi), o1);
    _mm512_storeu_ps(dst.add((bj + 2) * dst_rows + bi), o2);
    _mm512_storeu_ps(dst.add((bj + 3) * dst_rows + bi), o3);
    _mm512_storeu_ps(dst.add((bj + 4) * dst_rows + bi), o4);
    _mm512_storeu_ps(dst.add((bj + 5) * dst_rows + bi), o5);
    _mm512_storeu_ps(dst.add((bj + 6) * dst_rows + bi), o6);
    _mm512_storeu_ps(dst.add((bj + 7) * dst_rows + bi), o7);
    _mm512_storeu_ps(dst.add((bj + 8) * dst_rows + bi), o8);
    _mm512_storeu_ps(dst.add((bj + 9) * dst_rows + bi), o9);
    _mm512_storeu_ps(dst.add((bj + 10) * dst_rows + bi), o10);
    _mm512_storeu_ps(dst.add((bj + 11) * dst_rows + bi), o11);
    _mm512_storeu_ps(dst.add((bj + 12) * dst_rows + bi), o12);
    _mm512_storeu_ps(dst.add((bj + 13) * dst_rows + bi), o13);
    _mm512_storeu_ps(dst.add((bj + 14) * dst_rows + bi), o14);
    _mm512_storeu_ps(dst.add((bj + 15) * dst_rows + bi), o15);
}

fn fast_transpose(dst: *mut f32, src: *const f32, rows: usize, cols: usize) {
    const T: usize = 32;
    let mut i = 0;
    while i < rows {
        let imax = (i + T).min(rows);
        let mut j = 0;
        while j < cols {
            let jmax = (j + T).min(cols);
            unsafe {
                for ii in i..imax {
                    for jj in j..jmax {
                        *dst.add(jj * rows + ii) = *src.add(ii * cols + jj);
                    }
                }
            }
            j += T;
        }
        i += T;
    }
}

fn par(m: usize, k: usize, n: usize) -> gemm::Parallelism {
    let ops = 2 * m * k * n;
    if ops < (1 << 20) {
        gemm::Parallelism::None
    } else if ops < (1 << 25) {
        gemm::Parallelism::Rayon(5)
    } else {
        gemm::Parallelism::Rayon(0)
    }
}
fn sgemm(c: *mut f32, a: *const f32, b: *const f32, m: usize, k: usize, n: usize) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c,
            1isize,
            n as isize,
            false,
            a,
            1isize,
            k as isize,
            b,
            1isize,
            n as isize,
            0f32,
            1f32,
            false,
            false,
            false,
            par(m, k, n),
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
fn prefetch_ro(ptr: *const f32, bytes: usize) {
    use std::arch::x86_64::*;
    let mut p = ptr as *const i8;
    let end = unsafe { (ptr as *const i8).add(bytes) };
    while p < end {
        unsafe {
            _mm_prefetch(p, _MM_HINT_T1);
        }
        p = unsafe { p.add(64) };
    }
}
fn sgd_fused_tn(w: *mut f32, a: *const f32, b: *const f32, m: usize, k: usize, n: usize, lr: f32) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            w,
            1isize,
            n as isize,
            true,
            a,
            m as isize,
            1isize,
            b,
            1isize,
            n as isize,
            1f32,
            -lr,
            false,
            false,
            false,
            par(m, k, n),
        );
    }
}

// bwd_delta_Wt: delta_prev[B×r] = delta[B×c] @ W^T, no intermediate transpose buffer needed
// Uses gemm with non-unit col-stride on W to read it as W^T
fn bwd_delta_Wt(
    delta_prev: *mut f32,
    w: *const f32,
    delta: *const f32,
    r: usize,
    c: usize,
    batch: usize,
) {
    unsafe {
        gemm::gemm(
            batch,
            r,
            c,
            delta_prev,
            1isize,
            r as isize,
            false,
            delta,
            1isize,
            c as isize,
            w,
            c as isize,
            1isize,
            0f32,
            1f32,
            false,
            false,
            false,
            par(batch, c, r),
        );
    }
}

// fused_bwd_sgd: for layer W[r×c], batch B
//   1) delta_prev[B×r] = delta[B×c] @ W[r×c]^T   (backward pass)
//   2) W[r×c] -= lr * acts[B×r]^T @ delta[B×c]   (SGD update)
// Both operations read W[r×c]. This kernel tiles W so each tile is loaded once
// from memory and used for both computations.
//
// Layout: W is row-major [r×c], acts [B×r], delta [B×c], delta_prev [B×r]
//
// We tile over rows of W (dim r). For each tile of R_TILE rows of W:
//   - Compute delta_prev[:, ri..ri+R_TILE] += delta @ W[ri..ri+R_TILE, :]^T
//   - Update W[ri..ri+R_TILE, :] -= lr * acts[:, ri..ri+R_TILE]^T @ delta
//
// This is beneficial when W doesn't fit in L2 (e.g. W1 = 1024×1024 = 4MB).
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_bwd_sgd_avx2(
    w: *mut f32,
    acts: *const f32,     // [B × r]
    delta: *const f32,    // [B × c]
    delta_prev: *mut f32, // [B × r], must be zero-initialized before call
    r: usize,
    c: usize,
    batch: usize,
    lr: f32,
) {
    use std::arch::x86_64::*;
    let vlr = _mm256_set1_ps(lr);
    const R_TILE: usize = 4; // process 4 rows of W at a time

    // zero delta_prev (count = number of f32 elements)
    std::ptr::write_bytes(delta_prev, 0, batch * r);

    let mut ri = 0usize;
    while ri + R_TILE <= r {
        // For each tile of R_TILE rows of W[ri..ri+R_TILE, 0..c]:
        // W_tile is row-major: W[ri+t, j] = *w.add((ri+t)*c + j)

        // Pass 1: accumulate delta_prev[:, ri..ri+R_TILE] += delta @ W_tile^T
        // delta_prev[bi, ri+t] += sum_j delta[bi, j] * W[ri+t, j]
        for bi in 0..batch {
            let d_row = delta.add(bi * c);
            for t in 0usize..R_TILE {
                let w_row = w.add((ri + t) * c);
                let mut sum = _mm256_setzero_ps();
                let mut j = 0usize;
                while j + 8 <= c {
                    let dv = _mm256_loadu_ps(d_row.add(j));
                    let wv = _mm256_loadu_ps(w_row.add(j));
                    sum = _mm256_fmadd_ps(dv, wv, sum);
                    j += 8;
                }
                // horizontal sum of sum register
                let hi = _mm256_extractf128_ps(sum, 1);
                let lo = _mm256_castps256_ps128(sum);
                let s128 = _mm_add_ps(lo, hi);
                let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
                let s32 = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 1));
                let mut acc = _mm_cvtss_f32(s32);
                // scalar tail
                while j < c {
                    acc += *d_row.add(j) * *w_row.add(j);
                    j += 1;
                }
                *delta_prev.add(bi * r + ri + t) += acc;
            }
        }

        // Pass 2: W_tile -= lr * acts[:, ri..ri+R_TILE]^T @ delta
        // W[ri+t, j] -= lr * sum_bi acts[bi, ri+t] * delta[bi, j]
        // Process j in chunks of 8 (AVX2)
        for t in 0usize..R_TILE {
            let w_row = w.add((ri + t) * c);
            let mut j = 0usize;
            while j + 8 <= c {
                let mut grad = _mm256_setzero_ps();
                for bi in 0..batch {
                    let a_val = _mm256_set1_ps(*acts.add(bi * r + ri + t));
                    let dv = _mm256_loadu_ps(delta.add(bi * c + j));
                    grad = _mm256_fmadd_ps(a_val, dv, grad);
                }
                let wv = _mm256_loadu_ps(w_row.add(j));
                _mm256_storeu_ps(w_row.add(j), _mm256_fnmadd_ps(vlr, grad, wv));
                j += 8;
            }
            // scalar tail
            while j < c {
                let mut grad = 0.0f32;
                for bi in 0..batch {
                    grad += *acts.add(bi * r + ri + t) * *delta.add(bi * c + j);
                }
                *w_row.add(j) -= lr * grad;
                j += 1;
            }
        }

        ri += R_TILE;
    }

    // handle remaining rows (if r % R_TILE != 0)
    while ri < r {
        let w_row = w.add(ri * c);
        // backward
        for bi in 0..batch {
            let d_row = delta.add(bi * c);
            let mut acc = 0.0f32;
            let mut j = 0usize;
            while j + 8 <= c {
                let dv = _mm256_loadu_ps(d_row.add(j));
                let wv = _mm256_loadu_ps(w_row.add(j));
                let prod = _mm256_mul_ps(dv, wv);
                let hi = _mm256_extractf128_ps(prod, 1);
                let lo = _mm256_castps256_ps128(prod);
                let s = _mm_add_ps(lo, hi);
                let s2 = _mm_add_ps(s, _mm_movehl_ps(s, s));
                let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
                acc += _mm_cvtss_f32(s1);
                j += 8;
            }
            while j < c {
                acc += *d_row.add(j) * *w_row.add(j);
                j += 1;
            }
            *delta_prev.add(bi * r + ri) += acc;
        }
        // SGD
        let mut j = 0usize;
        while j + 8 <= c {
            let mut grad = _mm256_setzero_ps();
            for bi in 0..batch {
                let a_val = _mm256_set1_ps(*acts.add(bi * r + ri));
                let dv = _mm256_loadu_ps(delta.add(bi * c + j));
                grad = _mm256_fmadd_ps(a_val, dv, grad);
            }
            let wv = _mm256_loadu_ps(w_row.add(j));
            _mm256_storeu_ps(w_row.add(j), _mm256_fnmadd_ps(vlr, grad, wv));
            j += 8;
        }
        while j < c {
            let mut grad = 0.0f32;
            for bi in 0..batch {
                grad += *acts.add(bi * r + ri) * *delta.add(bi * c + j);
            }
            *w_row.add(j) -= lr * grad;
            j += 1;
        }
        ri += 1;
    }
}

// ── AVX-512 fused bias + relu — 16 floats/cycle ──────────────────────────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn bias_relu_avx512(
    pre: *mut f32,
    act: *mut f32,
    bias: *const f32,
    rows: usize,
    cols: usize,
) {
    use std::arch::x86_64::*;
    let zero = _mm512_setzero_ps();
    for bi in 0..rows {
        let base = bi * cols;
        let mut j = 0usize;
        while j + 16 <= cols {
            let b = _mm512_loadu_ps(bias.add(j));
            let p = _mm512_add_ps(_mm512_loadu_ps(pre.add(base + j)), b);
            _mm512_storeu_ps(pre.add(base + j), p);
            _mm512_storeu_ps(act.add(base + j), _mm512_max_ps(p, zero));
            j += 16;
        }
        // tail: up to 15 remaining with mask
        if j < cols {
            let rem = cols - j;
            let mask: u16 = (1u16 << rem) - 1;
            let b = _mm512_maskz_loadu_ps(mask, bias.add(j));
            let p = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, pre.add(base + j)), b);
            _mm512_mask_storeu_ps(pre.add(base + j), mask, p);
            _mm512_mask_storeu_ps(act.add(base + j), mask, _mm512_max_ps(p, zero));
        }
    }
}

// ── AVX-512 relu mask + db accumulate — 16 floats/cycle ──────────────────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn relu_mask_db_avx512(
    delta: *mut f32,
    pre: *const f32,
    db: *mut f32,
    rows: usize,
    cols: usize,
) {
    use std::arch::x86_64::*;
    std::ptr::write_bytes(db, 0, cols * std::mem::size_of::<f32>());
    let zero = _mm512_setzero_ps();
    for bi in 0..rows {
        let base = bi * cols;
        let mut j = 0usize;
        while j + 16 <= cols {
            let p = _mm512_loadu_ps(pre.add(base + j));
            let d = _mm512_loadu_ps(delta.add(base + j));
            let mask = _mm512_cmp_ps_mask(p, zero, _CMP_GT_OQ);
            let d2 = _mm512_maskz_mov_ps(mask, d);
            _mm512_storeu_ps(delta.add(base + j), d2);
            let acc = _mm512_loadu_ps(db.add(j));
            _mm512_storeu_ps(db.add(j), _mm512_add_ps(acc, d2));
            j += 16;
        }
        if j < cols {
            let rem = (cols - j) as u16;
            let kmask: u16 = (1u16 << rem) - 1;
            let p = _mm512_maskz_loadu_ps(kmask, pre.add(base + j));
            let d = _mm512_maskz_loadu_ps(kmask, delta.add(base + j));
            let cmask = _mm512_cmp_ps_mask(p, zero, _CMP_GT_OQ);
            let d2 = _mm512_maskz_mov_ps(cmask & kmask, d);
            _mm512_mask_storeu_ps(delta.add(base + j), kmask, d2);
            let acc = _mm512_maskz_loadu_ps(kmask, db.add(j));
            _mm512_mask_storeu_ps(db.add(j), kmask, _mm512_add_ps(acc, d2));
        }
    }
}

fn lcg_xavier(w: &mut [f32], r: usize, c: usize, rng: &mut u64) {
    let lim = (6.0f32 / (r as f32 + c as f32)).sqrt();
    for x in w.iter_mut() {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = ((*rng >> 11) & ((1u64 << 53) - 1)) as f32 / (1u64 << 53) as f32;
        *x = (f * 2.0 - 1.0) * lim;
    }
}

const LAYERS: [usize; 6] = [512, 1024, 1024, 512, 256, 1];
const N: usize = 5;
const B: usize = 64;

struct State {
    ws: [Vec<f32>; N],
    bs: [Vec<f32>; N],
    dbs: [Vec<f32>; N],
    tmps: [Vec<f32>; N],
    dts: [Vec<f32>; N],
    acts: [Vec<f32>; 6],
    pres: [Vec<f32>; 4],
    deltas: [Vec<f32>; N],
    last_loss: f32,
}

impl State {
    fn new() -> Self {
        let mut rng = 42u64;
        let mut ws: [Vec<f32>; N] =
            std::array::from_fn(|i| vec![0.0f32; LAYERS[i] * LAYERS[i + 1]]);
        lcg_xavier(&mut ws[0], 512, 1024, &mut rng);
        lcg_xavier(&mut ws[1], 1024, 1024, &mut rng);
        lcg_xavier(&mut ws[2], 1024, 512, &mut rng);
        lcg_xavier(&mut ws[3], 512, 256, &mut rng);
        lcg_xavier(&mut ws[4], 256, 1, &mut rng);
        State {
            ws,
            bs: std::array::from_fn(|i| vec![0.0f32; LAYERS[i + 1]]),
            dbs: std::array::from_fn(|i| vec![0.0f32; LAYERS[i + 1]]),
            tmps: std::array::from_fn(|i| vec![0.0f32; LAYERS[i] * B]),
            dts: std::array::from_fn(|i| vec![0.0f32; LAYERS[i + 1] * B]),
            acts: std::array::from_fn(|i| vec![0.0f32; B * LAYERS[i]]),
            pres: std::array::from_fn(|i| vec![0.0f32; B * LAYERS[i + 1]]),
            deltas: std::array::from_fn(|i| vec![0.0f32; B * LAYERS[i + 1]]),
            last_loss: 0.0,
        }
    }

    fn step(&mut self, x: &[f32], y: &[f32], lr: f32) {
        // ── FORWARD ────────────────────────────────────────────────────────
        self.acts[0].copy_from_slice(x);

        for i in 0..N {
            let (r, c) = (LAYERS[i], LAYERS[i + 1]);
            let is_last = i == N - 1;

            sgemm(
                if is_last {
                    self.acts[N].as_mut_ptr()
                } else {
                    self.pres[i].as_mut_ptr()
                },
                self.acts[i].as_ptr(),
                self.ws[i].as_ptr(),
                B,
                r,
                c,
            );

            if is_last {
                for bi in 0..B {
                    for j in 0..c {
                        self.acts[N][bi * c + j] += self.bs[i][j];
                    }
                }
            } else {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    bias_relu_avx512(
                        self.pres[i].as_mut_ptr(),
                        self.acts[i + 1].as_mut_ptr(),
                        self.bs[i].as_ptr(),
                        B,
                        c,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                for bi in 0..B {
                    for j in 0..c {
                        let v = self.pres[i][bi * c + j] + self.bs[i][j];
                        self.pres[i][bi * c + j] = v;
                        self.acts[i + 1][bi * c + j] = if v > 0.0 { v } else { 0.0 };
                    }
                }
            }
        }

        // ── LOSS (MSE) ─────────────────────────────────────────────────────
        let nt = B * LAYERS[N];
        let mut lacc = 0.0f32;
        for k in 0..nt {
            let d = self.acts[N][k] - y[k];
            lacc += d * d;
            self.deltas[N - 1][k] = 2.0 * d / (nt as f32);
        }
        self.last_loss = lacc / (nt as f32);

        // ── BACKWARD ───────────────────────────────────────────────────────
        for i in (0..N).rev() {
            let (r, c) = (LAYERS[i], LAYERS[i + 1]);
            let is_last = i == N - 1;

            if !is_last {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    relu_mask_db_avx512(
                        self.deltas[i].as_mut_ptr(),
                        self.pres[i].as_ptr(),
                        self.dbs[i].as_mut_ptr(),
                        B,
                        c,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.dbs[i].fill(0.0);
                    for bi in 0..B {
                        for j in 0..c {
                            let mask = if self.pres[i][bi * c + j] > 0.0 {
                                1.0f32
                            } else {
                                0.0
                            };
                            let d = self.deltas[i][bi * c + j] * mask;
                            self.deltas[i][bi * c + j] = d;
                            self.dbs[i][j] += d;
                        }
                    }
                }
                for k in 0..c {
                    self.bs[i][k] -= lr * self.dbs[i][k];
                }
            }

            if i > 0 {
                // For r >= c (shrinking layers): direct delta@W^T avoids 2 transposes and is faster
                // For r < c (expanding layers): baseline transpose+sgemm+transpose is faster
                if r > c {
                    bwd_delta_Wt(
                        self.deltas[i - 1].as_mut_ptr(),
                        self.ws[i].as_ptr(),
                        self.deltas[i].as_ptr(),
                        r,
                        c,
                        B,
                    );
                } else {
                    fast_transpose(self.dts[i].as_mut_ptr(), self.deltas[i].as_ptr(), B, c);
                    sgemm(
                        self.tmps[i].as_mut_ptr(),
                        self.ws[i].as_ptr(),
                        self.dts[i].as_ptr(),
                        r,
                        c,
                        B,
                    );
                    fast_transpose(self.deltas[i - 1].as_mut_ptr(), self.tmps[i].as_ptr(), r, B);
                }
            }

            sgd_fused_tn(
                self.ws[i].as_mut_ptr(),
                self.acts[i].as_ptr(),
                self.deltas[i].as_ptr(),
                r,
                B,
                c,
                lr,
            );

            if is_last {
                self.dbs[i].fill(0.0);
                for bi in 0..B {
                    for j in 0..c {
                        self.dbs[i][j] += self.deltas[i][bi * c + j];
                    }
                }
                for k in 0..c {
                    self.bs[i][k] -= lr * self.dbs[i][k];
                }
            }
        }
    }
}

fn main() {
    let x: Vec<f32> = (0..B * LAYERS[0])
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let y: Vec<f32> = (0..B * LAYERS[N]).map(|i| (i % 7) as f32 * 0.1).collect();
    let lr = 0.01f32;

    let mut s = State::new();
    for _ in 0..10 {
        s.step(&x, &y, lr);
    }

    const STEPS: usize = 50;
    const RUNS: usize = 7;
    let mut results = [0.0f64; RUNS];
    let mut checksum = 0.0f32;
    for r in 0..RUNS {
        let t0 = Instant::now();
        for _ in 0..STEPS {
            s.step(&x, &y, lr);
        }
        results[r] = t0.elapsed().as_nanos() as f64 / 1000.0 / STEPS as f64 / 1000.0;
        checksum += s.last_loss;
    }
    results.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = results[RUNS / 2];
    println!(
        "[TRAIN][VOLTA-RUST-V3] median={:.3} ms/step  all7={:?}",
        med,
        results.map(|x| format!("{:.3}", x))
    );
    println!("  checksum={:.6}", checksum);
    println!(
        "  batch={}  steps={}  in={}  out={}",
        B, STEPS, LAYERS[0], LAYERS[N]
    );

    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(code: u32);
        }
        ExitProcess(0);
    }
}
