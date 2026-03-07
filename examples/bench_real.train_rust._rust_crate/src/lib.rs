#![allow(non_snake_case, unused, private_interfaces)]
use std::alloc::{alloc_zeroed, dealloc, Layout};

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

fn sgemm(C: *mut f32, A: *const f32, B: *const f32, m: usize, k: usize, n: usize) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            C,
            1isize,
            n as isize,
            false,
            A,
            1isize,
            k as isize,
            B,
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

fn sgemm_tn(C: *mut f32, A: *const f32, B: *const f32, m: usize, k: usize, n: usize) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            C,
            1isize,
            n as isize,
            false,
            A,
            m as isize,
            1isize,
            B,
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

fn sgemm_nt(
    C: *mut f32,
    A: *const f32,
    B: *const f32,
    m: usize,
    k: usize,
    n: usize,
    b_cols: usize,
) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            C,
            1isize,
            n as isize,
            false,
            A,
            1isize,
            k as isize,
            B,
            b_cols as isize,
            1isize,
            0f32,
            1f32,
            false,
            false,
            false,
            par(m, k, n),
        );
    }
}

fn sgd_fused_tn(W: *mut f32, A: *const f32, B: *const f32, m: usize, k: usize, n: usize, lr: f32) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            W,
            1isize,
            n as isize,
            true,
            A,
            m as isize,
            1isize,
            B,
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

// delta_prev[batch×r] = delta[batch×c] @ W^T  (r > c: shrinking layers — faster than 2 transposes + sgemm)
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

// ── AVX2 fused bias + relu ────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bias_relu_avx2(pre: *mut f32, act: *mut f32, bias: *const f32, rows: usize, cols: usize) {
    let zero = _mm256_setzero_ps();
    for bi in 0..rows {
        let mut j = 0usize;
        while j + 8 <= cols {
            let b = _mm256_loadu_ps(bias.add(j));
            let p = _mm256_add_ps(_mm256_loadu_ps(pre.add(bi * cols + j)), b);
            _mm256_storeu_ps(pre.add(bi * cols + j), p);
            _mm256_storeu_ps(act.add(bi * cols + j), _mm256_max_ps(p, zero));
            j += 8;
        }
        while j < cols {
            let v = *pre.add(bi * cols + j) + *bias.add(j);
            *pre.add(bi * cols + j) = v;
            *act.add(bi * cols + j) = if v > 0.0 { v } else { 0.0 };
            j += 1;
        }
    }
}

// ── AVX2 relu mask + db accumulate ───────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_mask_db_avx2(
    delta: *mut f32,
    pre: *const f32,
    db: *mut f32,
    rows: usize,
    cols: usize,
) {
    std::ptr::write_bytes(db, 0, cols * std::mem::size_of::<f32>());
    let zero = _mm256_setzero_ps();
    for bi in 0..rows {
        let mut j = 0usize;
        while j + 8 <= cols {
            let p = _mm256_loadu_ps(pre.add(bi * cols + j));
            let d = _mm256_loadu_ps(delta.add(bi * cols + j));
            let mask = _mm256_cmp_ps(p, zero, _CMP_GT_OQ);
            let d2 = _mm256_and_ps(d, mask);
            _mm256_storeu_ps(delta.add(bi * cols + j), d2);
            let acc = _mm256_loadu_ps(db.add(j));
            _mm256_storeu_ps(db.add(j), _mm256_add_ps(acc, d2));
            j += 8;
        }
        while j < cols {
            let mask = if *pre.add(bi * cols + j) > 0.0 {
                1.0f32
            } else {
                0.0
            };
            let d = *delta.add(bi * cols + j) * mask;
            *delta.add(bi * cols + j) = d;
            *db.add(j) += d;
            j += 1;
        }
    }
}

use rayon::prelude::*;
#[allow(clippy::too_many_arguments)]
fn adam_update(
    w: *mut f32,
    mw: *mut f32,
    vw: *mut f32,
    dw: *const f32,
    n: usize,
    b1: f32,
    b2: f32,
    bc1: f32,
    bc2: f32,
    lr: f32,
    eps: f32,
    wd: f32,
) {
    const CHUNK: usize = 1 << 13;
    if n >= 1 << 17 {
        let w_s = w as usize;
        let mw_s = mw as usize;
        let vw_s = vw as usize;
        let dw_s = dw as usize;
        (0..n)
            .into_par_iter()
            .step_by(CHUNK)
            .for_each(|base: usize| {
                let end: usize = (base + CHUNK).min(n);
                unsafe {
                    adam_update_slice(
                        (w_s + base * 4) as *mut f32,
                        (mw_s + base * 4) as *mut f32,
                        (vw_s + base * 4) as *mut f32,
                        (dw_s + base * 4) as *const f32,
                        0,
                        end - base,
                        b1,
                        b2,
                        bc1,
                        bc2,
                        lr,
                        eps,
                        wd,
                    );
                }
            });
    } else {
        unsafe {
            adam_update_slice(w, mw, vw, dw, 0, n, b1, b2, bc1, bc2, lr, eps, wd);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn adam_update_slice(
    w: *mut f32,
    mw: *mut f32,
    vw: *mut f32,
    dw: *const f32,
    start: usize,
    end: usize,
    b1: f32,
    b2: f32,
    bc1: f32,
    bc2: f32,
    lr: f32,
    eps: f32,
    wd: f32,
) {
    use std::arch::x86_64::*;
    let vb1 = _mm256_set1_ps(b1);
    let v1mb1 = _mm256_set1_ps(1.0 - b1);
    let vb2 = _mm256_set1_ps(b2);
    let v1mb2 = _mm256_set1_ps(1.0 - b2);
    let vbc1 = _mm256_set1_ps(bc1);
    let vbc2 = _mm256_set1_ps(bc2);
    let vlr = _mm256_set1_ps(lr);
    let veps = _mm256_set1_ps(eps);
    let vwd = _mm256_set1_ps(wd);
    let mut i = start;
    while i + 8 <= end {
        let g = _mm256_loadu_ps(dw.add(i));
        let m0 = _mm256_loadu_ps(mw.add(i));
        let v0 = _mm256_loadu_ps(vw.add(i));
        let m1 = _mm256_add_ps(_mm256_mul_ps(vb1, m0), _mm256_mul_ps(v1mb1, g));
        let v1 = _mm256_add_ps(
            _mm256_mul_ps(vb2, v0),
            _mm256_mul_ps(v1mb2, _mm256_mul_ps(g, g)),
        );
        _mm256_storeu_ps(mw.add(i), m1);
        _mm256_storeu_ps(vw.add(i), v1);
        let mhat = _mm256_div_ps(m1, vbc1);
        let vhat = _mm256_div_ps(v1, vbc2);
        let denom = _mm256_add_ps(_mm256_sqrt_ps(vhat), veps);
        let w0 = _mm256_loadu_ps(w.add(i));
        // step = lr * (mhat/denom + wd*w)
        let step = _mm256_mul_ps(
            vlr,
            _mm256_add_ps(_mm256_div_ps(mhat, denom), _mm256_mul_ps(vwd, w0)),
        );
        _mm256_storeu_ps(w.add(i), _mm256_sub_ps(w0, step));
        i += 8;
    }
    while i < end {
        let g = *dw.add(i);
        let m0 = *mw.add(i);
        let v0 = *vw.add(i);
        let m1 = b1 * m0 + (1.0 - b1) * g;
        *mw.add(i) = m1;
        let v1 = b2 * v0 + (1.0 - b2) * g * g;
        *vw.add(i) = v1;
        let w0 = *w.add(i);
        *w.add(i) = w0 - lr * ((m1 / bc1) / ((v1 / bc2).sqrt() + eps) + wd * w0);
        i += 1;
    }
}
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn adam_update_slice(
    w: *mut f32,
    mw: *mut f32,
    vw: *mut f32,
    dw: *const f32,
    start: usize,
    end: usize,
    b1: f32,
    b2: f32,
    bc1: f32,
    bc2: f32,
    lr: f32,
    eps: f32,
    wd: f32,
) {
    for i in start..end {
        let g = *dw.add(i);
        let m0 = *mw.add(i);
        let v0 = *vw.add(i);
        let m1 = b1 * m0 + (1.0 - b1) * g;
        *mw.add(i) = m1;
        let v1 = b2 * v0 + (1.0 - b2) * g * g;
        *vw.add(i) = v1;
        let w0 = *w.add(i);
        *w.add(i) = w0 - lr * ((m1 / bc1) / ((v1 / bc2).sqrt() + eps) + wd * w0);
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_8x8_avx(
    dst: *mut f32,
    dst_rows: usize,
    src: *const f32,
    src_cols: usize,
    bi: usize,
    bj: usize,
) {
    let r0 = _mm256_loadu_ps(src.add((bi) * src_cols + bj));
    let r1 = _mm256_loadu_ps(src.add((bi + 1) * src_cols + bj));
    let r2 = _mm256_loadu_ps(src.add((bi + 2) * src_cols + bj));
    let r3 = _mm256_loadu_ps(src.add((bi + 3) * src_cols + bj));
    let r4 = _mm256_loadu_ps(src.add((bi + 4) * src_cols + bj));
    let r5 = _mm256_loadu_ps(src.add((bi + 5) * src_cols + bj));
    let r6 = _mm256_loadu_ps(src.add((bi + 6) * src_cols + bj));
    let r7 = _mm256_loadu_ps(src.add((bi + 7) * src_cols + bj));
    let t0 = _mm256_unpacklo_ps(r0, r1);
    let t1 = _mm256_unpackhi_ps(r0, r1);
    let t2 = _mm256_unpacklo_ps(r2, r3);
    let t3 = _mm256_unpackhi_ps(r2, r3);
    let t4 = _mm256_unpacklo_ps(r4, r5);
    let t5 = _mm256_unpackhi_ps(r4, r5);
    let t6 = _mm256_unpacklo_ps(r6, r7);
    let t7 = _mm256_unpackhi_ps(r6, r7);
    let s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    let s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    let s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    let s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    let s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    let s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    let s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    let s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    let o0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    let o1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    let o2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    let o3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    let o4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    let o5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    let o6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    let o7 = _mm256_permute2f128_ps(s3, s7, 0x31);
    _mm256_storeu_ps(dst.add((bj) * dst_rows + bi), o0);
    _mm256_storeu_ps(dst.add((bj + 1) * dst_rows + bi), o1);
    _mm256_storeu_ps(dst.add((bj + 2) * dst_rows + bi), o2);
    _mm256_storeu_ps(dst.add((bj + 3) * dst_rows + bi), o3);
    _mm256_storeu_ps(dst.add((bj + 4) * dst_rows + bi), o4);
    _mm256_storeu_ps(dst.add((bj + 5) * dst_rows + bi), o5);
    _mm256_storeu_ps(dst.add((bj + 6) * dst_rows + bi), o6);
    _mm256_storeu_ps(dst.add((bj + 7) * dst_rows + bi), o7);
}

fn fast_transpose_scalar(dst: *mut f32, src: *const f32, rows: usize, cols: usize) {
    const T: usize = 32;
    let mut i = 0usize;
    while i < rows {
        let imax = if i + T < rows { i + T } else { rows };
        let mut j = 0usize;
        while j < cols {
            let jmax = if j + T < cols { j + T } else { cols };
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

fn fast_transpose(dst: *mut f32, src: *const f32, rows: usize, cols: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if rows.is_multiple_of(8) && cols.is_multiple_of(8) {
            let mut i = 0usize;
            while i < rows {
                let mut j = 0usize;
                while j < cols {
                    unsafe {
                        transpose_8x8_avx(dst, rows, src, cols, i, j);
                    }
                    j += 8;
                }
                i += 8;
            }
            return;
        }
    }
    fast_transpose_scalar(dst, src, rows, cols);
}

#[repr(C)]
struct Handle {
    last_loss: f32,
    batch: usize,
    w0: *mut f32,
    b0: *mut f32,
    dw0: *mut f32,
    db0: *mut f32,
    tmp0: *mut f32,
    dt0: *mut f32,
    delta0: *mut f32,
    w1: *mut f32,
    b1: *mut f32,
    dw1: *mut f32,
    db1: *mut f32,
    tmp1: *mut f32,
    dt1: *mut f32,
    delta1: *mut f32,
    w2: *mut f32,
    b2: *mut f32,
    dw2: *mut f32,
    db2: *mut f32,
    tmp2: *mut f32,
    dt2: *mut f32,
    delta2: *mut f32,
    w3: *mut f32,
    b3: *mut f32,
    dw3: *mut f32,
    db3: *mut f32,
    tmp3: *mut f32,
    dt3: *mut f32,
    delta3: *mut f32,
    w4: *mut f32,
    b4: *mut f32,
    dw4: *mut f32,
    db4: *mut f32,
    tmp4: *mut f32,
    dt4: *mut f32,
    delta4: *mut f32,
    act0: *mut f32,
    act1: *mut f32,
    act2: *mut f32,
    act3: *mut f32,
    act4: *mut f32,
    act5: *mut f32,
    pre0: *mut f32,
    pre1: *mut f32,
    pre2: *mut f32,
    pre3: *mut f32,
}

unsafe fn alloc_f32(n: usize) -> *mut f32 {
    let layout = Layout::array::<f32>(n).unwrap();
    alloc_zeroed(layout) as *mut f32
}

unsafe fn free_f32(p: *mut f32, n: usize) {
    if !p.is_null() {
        dealloc(p as *mut u8, Layout::array::<f32>(n).unwrap());
    }
}

fn lcg_xavier_init(w: *mut f32, n: usize, r: usize, c: usize, rng: &mut u64) {
    let lim = (6.0f32 / (r as f32 + c as f32)).sqrt();
    unsafe {
        for i in 0..n {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = ((*rng >> 11) & ((1u64 << 53) - 1)) as f32 / (1u64 << 53) as f32;
            *w.add(i) = (f * 2.0 - 1.0) * lim;
        }
    }
}

#[no_mangle]
/// # Safety
/// `batch` must be non-negative and the returned handle must later be freed by `volta_train_free`.
pub unsafe extern "C" fn volta_train_init(batch: i32) -> *mut Handle {
    let batch = batch as usize;
    let h = Box::into_raw(Box::new(Handle {
        last_loss: 0.0,
        batch,
        w0: alloc_f32(512 * 1024),
        b0: alloc_f32(1024),
        dw0: alloc_f32(512 * 1024),
        db0: alloc_f32(1024),
        tmp0: alloc_f32(512 * batch),
        dt0: alloc_f32(1024 * batch),
        delta0: alloc_f32(batch * 1024),
        w1: alloc_f32(1024 * 1024),
        b1: alloc_f32(1024),
        dw1: alloc_f32(1024 * 1024),
        db1: alloc_f32(1024),
        tmp1: alloc_f32(1024 * batch),
        dt1: alloc_f32(1024 * batch),
        delta1: alloc_f32(batch * 1024),
        w2: alloc_f32(1024 * 512),
        b2: alloc_f32(512),
        dw2: alloc_f32(1024 * 512),
        db2: alloc_f32(512),
        tmp2: alloc_f32(1024 * batch),
        dt2: alloc_f32(512 * batch),
        delta2: alloc_f32(batch * 512),
        w3: alloc_f32(512 * 256),
        b3: alloc_f32(256),
        dw3: alloc_f32(512 * 256),
        db3: alloc_f32(256),
        tmp3: alloc_f32(512 * batch),
        dt3: alloc_f32(256 * batch),
        delta3: alloc_f32(batch * 256),
        w4: alloc_f32(256),
        b4: alloc_f32(1),
        dw4: alloc_f32(256),
        db4: alloc_f32(1),
        tmp4: alloc_f32(256 * batch),
        dt4: alloc_f32(batch),
        delta4: alloc_f32(batch),
        act0: alloc_f32(batch * 512),
        act1: alloc_f32(batch * 1024),
        act2: alloc_f32(batch * 1024),
        act3: alloc_f32(batch * 512),
        act4: alloc_f32(batch * 256),
        act5: alloc_f32(batch),
        pre0: alloc_f32(batch * 1024),
        pre1: alloc_f32(batch * 1024),
        pre2: alloc_f32(batch * 512),
        pre3: alloc_f32(batch * 256),
    }));
    let mut rng = 42u64;
    lcg_xavier_init((*h).w0, 512 * 1024, 512, 1024, &mut rng);
    lcg_xavier_init((*h).w1, 1024 * 1024, 1024, 1024, &mut rng);
    lcg_xavier_init((*h).w2, 1024 * 512, 1024, 512, &mut rng);
    lcg_xavier_init((*h).w3, 512 * 256, 512, 256, &mut rng);
    lcg_xavier_init((*h).w4, 256, 256, 1, &mut rng);
    h
}

#[no_mangle]
/// # Safety
/// `h` must be a valid handle and `w`/`b` must point to enough parameter data for the selected layer.
pub unsafe extern "C" fn volta_train_set_params(
    h: *mut Handle,
    li: i32,
    w: *const f32,
    b: *const f32,
) {
    if li == 0 {
        std::ptr::copy_nonoverlapping(w, (*h).w0, 512 * 1024);
        std::ptr::copy_nonoverlapping(b, (*h).b0, 1024);
        return;
    }
    if li == 1 {
        std::ptr::copy_nonoverlapping(w, (*h).w1, 1024 * 1024);
        std::ptr::copy_nonoverlapping(b, (*h).b1, 1024);
        return;
    }
    if li == 2 {
        std::ptr::copy_nonoverlapping(w, (*h).w2, 1024 * 512);
        std::ptr::copy_nonoverlapping(b, (*h).b2, 512);
        return;
    }
    if li == 3 {
        std::ptr::copy_nonoverlapping(w, (*h).w3, 512 * 256);
        std::ptr::copy_nonoverlapping(b, (*h).b3, 256);
        return;
    }
    if li == 4 {
        std::ptr::copy_nonoverlapping(w, (*h).w4, 256);
        std::ptr::copy_nonoverlapping(b, (*h).b4, 1);
    }
}

#[no_mangle]
/// # Safety
/// `h` must be a valid handle and `wo`/`bo` must each point to storage for at least five pointers.
pub unsafe extern "C" fn volta_train_get_params(
    h: *mut Handle,
    wo: *mut *mut f32,
    bo: *mut *mut f32,
) {
    *wo.add(0) = (*h).w0;
    *bo.add(0) = (*h).b0;
    *wo.add(1) = (*h).w1;
    *bo.add(1) = (*h).b1;
    *wo.add(2) = (*h).w2;
    *bo.add(2) = (*h).b2;
    *wo.add(3) = (*h).w3;
    *bo.add(3) = (*h).b3;
    *wo.add(4) = (*h).w4;
    *bo.add(4) = (*h).b4;
}

#[no_mangle]
/// # Safety
/// `h` must be a valid non-null handle allocated by `volta_train_init`.
pub unsafe extern "C" fn volta_train_loss(h: *mut Handle) -> f32 {
    (*h).last_loss
}

#[no_mangle]
/// # Safety
/// `h` must be either null or a valid handle previously returned by `volta_train_init`.
pub unsafe extern "C" fn volta_train_free(h: *mut Handle) {
    if h.is_null() {
        return;
    }
    free_f32((*h).w0, 512 * 1024);
    free_f32((*h).b0, 1024);
    free_f32((*h).dw0, 512 * 1024);
    free_f32((*h).db0, 1024);
    free_f32((*h).tmp0, 512 * (*h).batch);
    free_f32((*h).dt0, 1024 * (*h).batch);
    free_f32((*h).delta0, (*h).batch * 1024);
    free_f32((*h).w1, 1024 * 1024);
    free_f32((*h).b1, 1024);
    free_f32((*h).dw1, 1024 * 1024);
    free_f32((*h).db1, 1024);
    free_f32((*h).tmp1, 1024 * (*h).batch);
    free_f32((*h).dt1, 1024 * (*h).batch);
    free_f32((*h).delta1, (*h).batch * 1024);
    free_f32((*h).w2, 1024 * 512);
    free_f32((*h).b2, 512);
    free_f32((*h).dw2, 1024 * 512);
    free_f32((*h).db2, 512);
    free_f32((*h).tmp2, 1024 * (*h).batch);
    free_f32((*h).dt2, 512 * (*h).batch);
    free_f32((*h).delta2, (*h).batch * 512);
    free_f32((*h).w3, 512 * 256);
    free_f32((*h).b3, 256);
    free_f32((*h).dw3, 512 * 256);
    free_f32((*h).db3, 256);
    free_f32((*h).tmp3, 512 * (*h).batch);
    free_f32((*h).dt3, 256 * (*h).batch);
    free_f32((*h).delta3, (*h).batch * 256);
    free_f32((*h).w4, 256);
    free_f32((*h).b4, 1);
    free_f32((*h).dw4, 256);
    free_f32((*h).db4, 1);
    free_f32((*h).tmp4, 256 * (*h).batch);
    free_f32((*h).dt4, (*h).batch);
    free_f32((*h).delta4, (*h).batch);
    free_f32((*h).act0, (*h).batch * 512);
    free_f32((*h).act1, (*h).batch * 1024);
    free_f32((*h).act2, (*h).batch * 1024);
    free_f32((*h).act3, (*h).batch * 512);
    free_f32((*h).act4, (*h).batch * 256);
    free_f32((*h).act5, (*h).batch);
    free_f32((*h).pre0, (*h).batch * 1024);
    free_f32((*h).pre1, (*h).batch * 1024);
    free_f32((*h).pre2, (*h).batch * 512);
    free_f32((*h).pre3, (*h).batch * 256);
    let _ = Box::from_raw(h);
}

#[no_mangle]
/// # Safety
/// `h` must be a valid handle and `X`/`Y` must point to at least `batch*512` and `batch` floats respectively.
pub unsafe extern "C" fn volta_train_step(h: *mut Handle, X: *const f32, Y: *const f32, lr: f32) {
    let B = (*h).batch;

    // FORWARD
    std::ptr::copy_nonoverlapping(X, (*h).act0, B * 512);
    sgemm(
        (*h).pre0,
        (*h).act0 as *const f32,
        (*h).w0 as *const f32,
        B,
        512,
        1024,
    );
    bias_relu_avx2((*h).pre0, (*h).act1, (*h).b0 as *const f32, B, 1024);
    sgemm(
        (*h).pre1,
        (*h).act1 as *const f32,
        (*h).w1 as *const f32,
        B,
        1024,
        1024,
    );
    bias_relu_avx2((*h).pre1, (*h).act2, (*h).b1 as *const f32, B, 1024);
    sgemm(
        (*h).pre2,
        (*h).act2 as *const f32,
        (*h).w2 as *const f32,
        B,
        1024,
        512,
    );
    bias_relu_avx2((*h).pre2, (*h).act3, (*h).b2 as *const f32, B, 512);
    sgemm(
        (*h).pre3,
        (*h).act3 as *const f32,
        (*h).w3 as *const f32,
        B,
        512,
        256,
    );
    bias_relu_avx2((*h).pre3, (*h).act4, (*h).b3 as *const f32, B, 256);
    sgemm(
        (*h).act5,
        (*h).act4 as *const f32,
        (*h).w4 as *const f32,
        B,
        256,
        1,
    );
    for bi in 0..B {
        *(*h).act5.add(bi) += *(*h).b4;
    }

    // MSE loss
    let mut lacc = 0.0f32;
    let nt = B;
    for k in 0..nt {
        let d = *(*h).act5.add(k) - *Y.add(k);
        lacc += d * d;
        *(*h).delta4.add(k) = 2.0 * d / (nt as f32);
    }
    (*h).last_loss = lacc / (nt as f32);

    // BACKWARD
    // W4 (r=256 > c=1): use direct delta@W^T — avoids 2 transposes
    bwd_delta_Wt(
        (*h).delta3,
        (*h).w4 as *const f32,
        (*h).delta4 as *const f32,
        256,
        1,
        B,
    );
    std::ptr::write_bytes((*h).db4, 0, std::mem::size_of::<f32>());
    for bi in 0..B {
        *(*h).db4 += *(*h).delta4.add(bi);
    }
    relu_mask_db_avx2((*h).delta3, (*h).pre3 as *const f32, (*h).db3, B, 256);
    // W3 (r=512 > c=256): use direct delta@W^T
    bwd_delta_Wt(
        (*h).delta2,
        (*h).w3 as *const f32,
        (*h).delta3 as *const f32,
        512,
        256,
        B,
    );
    relu_mask_db_avx2((*h).delta2, (*h).pre2 as *const f32, (*h).db2, B, 512);
    // W2 (r=1024 > c=512): use direct delta@W^T
    bwd_delta_Wt(
        (*h).delta1,
        (*h).w2 as *const f32,
        (*h).delta2 as *const f32,
        1024,
        512,
        B,
    );
    relu_mask_db_avx2((*h).delta1, (*h).pre1 as *const f32, (*h).db1, B, 1024);
    fast_transpose((*h).dt1, (*h).delta1 as *const f32, B, 1024);
    sgemm(
        (*h).tmp1,
        (*h).w1 as *const f32,
        (*h).dt1 as *const f32,
        1024,
        1024,
        B,
    );
    fast_transpose((*h).delta0, (*h).tmp1 as *const f32, 1024, B);
    relu_mask_db_avx2((*h).delta0, (*h).pre0 as *const f32, (*h).db0, B, 1024);
    sgd_fused_tn(
        (*h).w4,
        (*h).act4 as *const f32,
        (*h).delta4 as *const f32,
        256,
        B,
        1,
        lr,
    );
    for k in 0..1 {
        *(*h).b4.add(k) -= lr * *(*h).db4.add(k);
    }
    sgd_fused_tn(
        (*h).w3,
        (*h).act3 as *const f32,
        (*h).delta3 as *const f32,
        512,
        B,
        256,
        lr,
    );
    for k in 0..256 {
        *(*h).b3.add(k) -= lr * *(*h).db3.add(k);
    }
    sgd_fused_tn(
        (*h).w2,
        (*h).act2 as *const f32,
        (*h).delta2 as *const f32,
        1024,
        B,
        512,
        lr,
    );
    for k in 0..512 {
        *(*h).b2.add(k) -= lr * *(*h).db2.add(k);
    }
    sgd_fused_tn(
        (*h).w1,
        (*h).act1 as *const f32,
        (*h).delta1 as *const f32,
        1024,
        B,
        1024,
        lr,
    );
    for k in 0..1024 {
        *(*h).b1.add(k) -= lr * *(*h).db1.add(k);
    }
    sgd_fused_tn(
        (*h).w0,
        (*h).act0 as *const f32,
        (*h).delta0 as *const f32,
        512,
        B,
        1024,
        lr,
    );
    for k in 0..1024 {
        *(*h).b0.add(k) -= lr * *(*h).db0.add(k);
    }
}
