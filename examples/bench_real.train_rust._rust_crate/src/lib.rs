#![allow(non_snake_case, unused, private_interfaces)]
use std::alloc::{alloc_zeroed, dealloc, Layout};

fn par(m: usize, k: usize, n: usize) -> gemm::Parallelism {
    let ops = 2*m*k*n;
    if ops < (1<<20) { gemm::Parallelism::None }
    else if ops < (1<<25) { gemm::Parallelism::Rayon(5) }
    else { gemm::Parallelism::Rayon(0) }
}

fn sgemm(C: *mut f32, A: *const f32, B: *const f32, m: usize, k: usize, n: usize) {
    unsafe { gemm::gemm(m,n,k, C,1isize,n as isize, false, A,1isize,k as isize, B,1isize,n as isize, 0f32,1f32, false,false,false, par(m,k,n)); }
}

fn sgemm_tn(C: *mut f32, A: *const f32, B: *const f32, m: usize, k: usize, n: usize) {
    unsafe { gemm::gemm(m,n,k, C,1isize,n as isize, false, A,m as isize,1isize, B,1isize,n as isize, 0f32,1f32, false,false,false, par(m,k,n)); }
}

// --- SGD backends ---

#[link(name = "mkl_rt", kind = "dylib")]
extern "C" {
    fn cblas_sgemm(
        layout: i32, transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

// MKL: W[m×n] += -lr * act^T @ delta
// act is k×m (B×m) row-major → transa=Trans, lda=m
// delta is k×n (B×n) row-major → transb=NoTrans, ldb=n
#[inline]
fn sgd_fused_tn_mkl(W: *mut f32, A: *const f32, B: *const f32, m: usize, k: usize, n: usize, lr: f32) {
    unsafe {
        cblas_sgemm(
            101, // CblasRowMajor
            112, // CblasTrans
            111, // CblasNoTrans
            m as i32, n as i32, k as i32,
            -lr, A, m as i32,
            B, n as i32,
            1.0f32, W, n as i32,
        );
    }
}

#[inline]
fn sgd_fused_tn_gemmcrate(W: *mut f32, A: *const f32, B: *const f32, m: usize, k: usize, n: usize, lr: f32) {
    unsafe { gemm::gemm(m,n,k, W,1isize,n as isize, true, A,m as isize,1isize, B,1isize,n as isize, 1f32,-lr, false,false,false, par(m,k,n)); }
}

// Runtime backend selection via VOLTA_SGD_BACKEND env var:
//   VOLTA_SGD_BACKEND=mkl   → MKL cblas (default when MKL available)
//   VOLTA_SGD_BACKEND=gemm  → gemm crate
// Default: MKL (since mkl_rt.lib is linked)
fn sgd_fused_tn(W: *mut f32, A: *const f32, B: *const f32, m: usize, k: usize, n: usize, lr: f32) {
    use std::sync::OnceLock;
    static USE_MKL: OnceLock<bool> = OnceLock::new();
    let use_mkl = USE_MKL.get_or_init(|| {
        std::env::var("VOLTA_SGD_BACKEND").map(|v| v != "gemm").unwrap_or(true)
    });
    if *use_mkl {
        sgd_fused_tn_mkl(W, A, B, m, k, n, lr);
    } else {
        sgd_fused_tn_gemmcrate(W, A, B, m, k, n, lr);
    }
}

fn sgemm_nt(C: *mut f32, A: *const f32, B: *const f32, m: usize, k: usize, n: usize, b_cols: usize) {
    unsafe { gemm::gemm(m,n,k, C,1isize,n as isize, false, A,1isize,k as isize, B,b_cols as isize,1isize, 0f32,1f32, false,false,false, par(m,k,n)); }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_8x8_avx(dst: *mut f32, dst_rows: usize, src: *const f32, src_cols: usize, bi: usize, bj: usize) {
    let r0=_mm256_loadu_ps(src.add((bi  )*src_cols+bj)); let r1=_mm256_loadu_ps(src.add((bi+1)*src_cols+bj));
    let r2=_mm256_loadu_ps(src.add((bi+2)*src_cols+bj)); let r3=_mm256_loadu_ps(src.add((bi+3)*src_cols+bj));
    let r4=_mm256_loadu_ps(src.add((bi+4)*src_cols+bj)); let r5=_mm256_loadu_ps(src.add((bi+5)*src_cols+bj));
    let r6=_mm256_loadu_ps(src.add((bi+6)*src_cols+bj)); let r7=_mm256_loadu_ps(src.add((bi+7)*src_cols+bj));
    let t0=_mm256_unpacklo_ps(r0,r1); let t1=_mm256_unpackhi_ps(r0,r1);
    let t2=_mm256_unpacklo_ps(r2,r3); let t3=_mm256_unpackhi_ps(r2,r3);
    let t4=_mm256_unpacklo_ps(r4,r5); let t5=_mm256_unpackhi_ps(r4,r5);
    let t6=_mm256_unpacklo_ps(r6,r7); let t7=_mm256_unpackhi_ps(r6,r7);
    let s0=_mm256_shuffle_ps(t0,t2,0x44); let s1=_mm256_shuffle_ps(t0,t2,0xEE);
    let s2=_mm256_shuffle_ps(t1,t3,0x44); let s3=_mm256_shuffle_ps(t1,t3,0xEE);
    let s4=_mm256_shuffle_ps(t4,t6,0x44); let s5=_mm256_shuffle_ps(t4,t6,0xEE);
    let s6=_mm256_shuffle_ps(t5,t7,0x44); let s7=_mm256_shuffle_ps(t5,t7,0xEE);
    let o0=_mm256_permute2f128_ps(s0,s4,0x20); let o1=_mm256_permute2f128_ps(s1,s5,0x20);
    let o2=_mm256_permute2f128_ps(s2,s6,0x20); let o3=_mm256_permute2f128_ps(s3,s7,0x20);
    let o4=_mm256_permute2f128_ps(s0,s4,0x31); let o5=_mm256_permute2f128_ps(s1,s5,0x31);
    let o6=_mm256_permute2f128_ps(s2,s6,0x31); let o7=_mm256_permute2f128_ps(s3,s7,0x31);
    _mm256_storeu_ps(dst.add((bj  )*dst_rows+bi),o0); _mm256_storeu_ps(dst.add((bj+1)*dst_rows+bi),o1);
    _mm256_storeu_ps(dst.add((bj+2)*dst_rows+bi),o2); _mm256_storeu_ps(dst.add((bj+3)*dst_rows+bi),o3);
    _mm256_storeu_ps(dst.add((bj+4)*dst_rows+bi),o4); _mm256_storeu_ps(dst.add((bj+5)*dst_rows+bi),o5);
    _mm256_storeu_ps(dst.add((bj+6)*dst_rows+bi),o6); _mm256_storeu_ps(dst.add((bj+7)*dst_rows+bi),o7);
}

fn fast_transpose_scalar(dst: *mut f32, src: *const f32, rows: usize, cols: usize) {
    const T: usize = 32;
    let mut i = 0usize; while i < rows {
        let imax = if i+T<rows{i+T}else{rows}; let mut j=0usize; while j<cols {
            let jmax=if j+T<cols{j+T}else{cols};
            unsafe{for ii in i..imax{for jj in j..jmax{ *dst.add(jj*rows+ii)=*src.add(ii*cols+jj); }}}
            j+=T; } i+=T; }
}

fn fast_transpose(dst: *mut f32, src: *const f32, rows: usize, cols: usize) {
    #[cfg(target_arch = "x86_64")] {
        if rows % 8 == 0 && cols % 8 == 0 {
            let mut i=0usize; while i<rows { let mut j=0usize; while j<cols {
                unsafe { transpose_8x8_avx(dst,rows,src,cols,i,j); }
                j+=8; } i+=8; }
            return;
        }
    }
    fast_transpose_scalar(dst, src, rows, cols);
}

#[repr(C)]
struct Handle {
    last_loss: f32, batch: usize,
    w0: *mut f32, b0: *mut f32, dw0: *mut f32, db0: *mut f32,
    tmp0: *mut f32, dt0: *mut f32, delta0: *mut f32,
    w1: *mut f32, b1: *mut f32, dw1: *mut f32, db1: *mut f32,
    tmp1: *mut f32, dt1: *mut f32, delta1: *mut f32,
    w2: *mut f32, b2: *mut f32, dw2: *mut f32, db2: *mut f32,
    tmp2: *mut f32, dt2: *mut f32, delta2: *mut f32,
    w3: *mut f32, b3: *mut f32, dw3: *mut f32, db3: *mut f32,
    tmp3: *mut f32, dt3: *mut f32, delta3: *mut f32,
    w4: *mut f32, b4: *mut f32, dw4: *mut f32, db4: *mut f32,
    tmp4: *mut f32, dt4: *mut f32, delta4: *mut f32,
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
    if !p.is_null() { dealloc(p as *mut u8, Layout::array::<f32>(n).unwrap()); }
}

fn lcg_xavier_init(w: *mut f32, n: usize, r: usize, c: usize, rng: &mut u64) {
    let lim = (6.0f32 / (r as f32 + c as f32)).sqrt();
    unsafe { for i in 0..n {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let f = ((*rng >> 11) & ((1u64<<53)-1)) as f32 / (1u64<<53) as f32;
        *w.add(i) = (f * 2.0 - 1.0) * lim;
    }}
}

#[no_mangle]
pub unsafe extern "C" fn volta_train_init(batch: i32) -> *mut Handle {
    let batch = batch as usize;
    let h = Box::into_raw(Box::new(Handle {
        last_loss: 0.0, batch,
        w0: alloc_f32(512*1024), b0: alloc_f32(1024),
        dw0: alloc_f32(512*1024), db0: alloc_f32(1024),
        tmp0: alloc_f32(512*batch), dt0: alloc_f32(1024*batch), delta0: alloc_f32(batch*1024),
        w1: alloc_f32(1024*1024), b1: alloc_f32(1024),
        dw1: alloc_f32(1024*1024), db1: alloc_f32(1024),
        tmp1: alloc_f32(1024*batch), dt1: alloc_f32(1024*batch), delta1: alloc_f32(batch*1024),
        w2: alloc_f32(1024*512), b2: alloc_f32(512),
        dw2: alloc_f32(1024*512), db2: alloc_f32(512),
        tmp2: alloc_f32(1024*batch), dt2: alloc_f32(512*batch), delta2: alloc_f32(batch*512),
        w3: alloc_f32(512*256), b3: alloc_f32(256),
        dw3: alloc_f32(512*256), db3: alloc_f32(256),
        tmp3: alloc_f32(512*batch), dt3: alloc_f32(256*batch), delta3: alloc_f32(batch*256),
        w4: alloc_f32(256*1), b4: alloc_f32(1),
        dw4: alloc_f32(256*1), db4: alloc_f32(1),
        tmp4: alloc_f32(256*batch), dt4: alloc_f32(1*batch), delta4: alloc_f32(batch*1),
        act0: alloc_f32(batch*512),
        act1: alloc_f32(batch*1024),
        act2: alloc_f32(batch*1024),
        act3: alloc_f32(batch*512),
        act4: alloc_f32(batch*256),
        act5: alloc_f32(batch*1),
        pre0: alloc_f32(batch*1024),
        pre1: alloc_f32(batch*1024),
        pre2: alloc_f32(batch*512),
        pre3: alloc_f32(batch*256),
    }));
    let mut rng = 42u64;
    lcg_xavier_init((*h).w0, 512*1024, 512, 1024, &mut rng);
    lcg_xavier_init((*h).w1, 1024*1024, 1024, 1024, &mut rng);
    lcg_xavier_init((*h).w2, 1024*512, 1024, 512, &mut rng);
    lcg_xavier_init((*h).w3, 512*256, 512, 256, &mut rng);
    lcg_xavier_init((*h).w4, 256*1, 256, 1, &mut rng);
    h
}

#[no_mangle]
pub unsafe extern "C" fn volta_train_set_params(h: *mut Handle, li: i32, w: *const f32, b: *const f32) {
    if li==0 { std::ptr::copy_nonoverlapping(w,(*h).w0,512*1024); std::ptr::copy_nonoverlapping(b,(*h).b0,1024); return; }
    if li==1 { std::ptr::copy_nonoverlapping(w,(*h).w1,1024*1024); std::ptr::copy_nonoverlapping(b,(*h).b1,1024); return; }
    if li==2 { std::ptr::copy_nonoverlapping(w,(*h).w2,1024*512); std::ptr::copy_nonoverlapping(b,(*h).b2,512); return; }
    if li==3 { std::ptr::copy_nonoverlapping(w,(*h).w3,512*256); std::ptr::copy_nonoverlapping(b,(*h).b3,256); return; }
    if li==4 { std::ptr::copy_nonoverlapping(w,(*h).w4,256*1); std::ptr::copy_nonoverlapping(b,(*h).b4,1); return; }
}

#[no_mangle]
pub unsafe extern "C" fn volta_train_get_params(h: *mut Handle, wo: *mut *mut f32, bo: *mut *mut f32) {
    *wo.add(0)=(*h).w0; *bo.add(0)=(*h).b0;
    *wo.add(1)=(*h).w1; *bo.add(1)=(*h).b1;
    *wo.add(2)=(*h).w2; *bo.add(2)=(*h).b2;
    *wo.add(3)=(*h).w3; *bo.add(3)=(*h).b3;
    *wo.add(4)=(*h).w4; *bo.add(4)=(*h).b4;
}

#[no_mangle]
pub unsafe extern "C" fn volta_train_loss(h: *mut Handle) -> f32 { (*h).last_loss }

#[no_mangle]
pub unsafe extern "C" fn volta_train_free(h: *mut Handle) {
    if h.is_null() { return; }
        free_f32((*h).w0,512*1024); free_f32((*h).b0,1024);
        free_f32((*h).dw0,512*1024); free_f32((*h).db0,1024);
        free_f32((*h).tmp0,512*(*h).batch); free_f32((*h).dt0,1024*(*h).batch); free_f32((*h).delta0,(*h).batch*1024);
        free_f32((*h).w1,1024*1024); free_f32((*h).b1,1024);
        free_f32((*h).dw1,1024*1024); free_f32((*h).db1,1024);
        free_f32((*h).tmp1,1024*(*h).batch); free_f32((*h).dt1,1024*(*h).batch); free_f32((*h).delta1,(*h).batch*1024);
        free_f32((*h).w2,1024*512); free_f32((*h).b2,512);
        free_f32((*h).dw2,1024*512); free_f32((*h).db2,512);
        free_f32((*h).tmp2,1024*(*h).batch); free_f32((*h).dt2,512*(*h).batch); free_f32((*h).delta2,(*h).batch*512);
        free_f32((*h).w3,512*256); free_f32((*h).b3,256);
        free_f32((*h).dw3,512*256); free_f32((*h).db3,256);
        free_f32((*h).tmp3,512*(*h).batch); free_f32((*h).dt3,256*(*h).batch); free_f32((*h).delta3,(*h).batch*256);
        free_f32((*h).w4,256*1); free_f32((*h).b4,1);
        free_f32((*h).dw4,256*1); free_f32((*h).db4,1);
        free_f32((*h).tmp4,256*(*h).batch); free_f32((*h).dt4,1*(*h).batch); free_f32((*h).delta4,(*h).batch*1);
        free_f32((*h).act0,(*h).batch*512);
        free_f32((*h).act1,(*h).batch*1024);
        free_f32((*h).act2,(*h).batch*1024);
        free_f32((*h).act3,(*h).batch*512);
        free_f32((*h).act4,(*h).batch*256);
        free_f32((*h).act5,(*h).batch*1);
        free_f32((*h).pre0,(*h).batch*1024);
        free_f32((*h).pre1,(*h).batch*1024);
        free_f32((*h).pre2,(*h).batch*512);
        free_f32((*h).pre3,(*h).batch*256);
    let _ = Box::from_raw(h);
}

#[no_mangle]
pub unsafe extern "C" fn volta_train_step(h: *mut Handle, X: *const f32, Y: *const f32, lr: f32) {
    let B = (*h).batch;

    // FORWARD — fused bias+relu per hidden layer
    std::ptr::copy_nonoverlapping(X, (*h).act0, B*512);
    sgemm((*h).pre0, (*h).act0 as *const f32, (*h).w0 as *const f32, B, 512, 1024);
    for bi in 0..B { for j in 0..1024 {
        let v = *(*h).pre0.add(bi*1024+j) + *(*h).b0.add(j);
        *(*h).pre0.add(bi*1024+j) = v;
        *(*h).act1.add(bi*1024+j) = if v > 0.0 { v } else { 0.0 };
    }}
    sgemm((*h).pre1, (*h).act1 as *const f32, (*h).w1 as *const f32, B, 1024, 1024);
    for bi in 0..B { for j in 0..1024 {
        let v = *(*h).pre1.add(bi*1024+j) + *(*h).b1.add(j);
        *(*h).pre1.add(bi*1024+j) = v;
        *(*h).act2.add(bi*1024+j) = if v > 0.0 { v } else { 0.0 };
    }}
    sgemm((*h).pre2, (*h).act2 as *const f32, (*h).w2 as *const f32, B, 1024, 512);
    for bi in 0..B { for j in 0..512 {
        let v = *(*h).pre2.add(bi*512+j) + *(*h).b2.add(j);
        *(*h).pre2.add(bi*512+j) = v;
        *(*h).act3.add(bi*512+j) = if v > 0.0 { v } else { 0.0 };
    }}
    sgemm((*h).pre3, (*h).act3 as *const f32, (*h).w3 as *const f32, B, 512, 256);
    for bi in 0..B { for j in 0..256 {
        let v = *(*h).pre3.add(bi*256+j) + *(*h).b3.add(j);
        *(*h).pre3.add(bi*256+j) = v;
        *(*h).act4.add(bi*256+j) = if v > 0.0 { v } else { 0.0 };
    }}
    sgemm((*h).act5, (*h).act4 as *const f32, (*h).w4 as *const f32, B, 256, 1);
    *(*h).act5 += *(*h).b4;

    // MSE loss
    let mut lacc = 0.0f32;
    let nt = B*1;
    for k in 0..nt { let d = *(*h).act5.add(k) - *Y.add(k); lacc += d*d; *(*h).delta4.add(k) = 2.0*d/(nt as f32); }
    (*h).last_loss = lacc / (nt as f32);

    // BACKWARD — fused relu_mask+db_reduce per non-last layer, correct order
    // Layer 4 (last, no relu): db4 separate
    std::ptr::write_bytes((*h).db4, 0, 1);
    for bi in 0..B { *(*h).db4 += *(*h).delta4.add(bi); }
    // dX4 → delta3
    fast_transpose((*h).dt4, (*h).delta4 as *const f32, B, 1);
    sgemm((*h).tmp4, (*h).w4 as *const f32, (*h).dt4 as *const f32, 256, 1, B);
    fast_transpose((*h).delta3, (*h).tmp4 as *const f32, 256, B);
    // SGD w4, b4
    sgd_fused_tn((*h).w4, (*h).act4 as *const f32, (*h).delta4 as *const f32, 256, B, 1, lr);
    *(*h).b4 -= lr * *(*h).db4;

    // Layer 3: fused relu_mask + db3 reduce; then dX3 → delta2; SGD w3,b3
    std::ptr::write_bytes((*h).db3, 0, 256);
    for bi in 0..B { for j in 0..256 {
        let mask = if *(*h).pre3.add(bi*256+j) > 0.0 { 1.0f32 } else { 0.0 };
        let d = *(*h).delta3.add(bi*256+j) * mask;
        *(*h).delta3.add(bi*256+j) = d;
        *(*h).db3.add(j) += d;
    }}
    fast_transpose((*h).dt3, (*h).delta3 as *const f32, B, 256);
    sgemm((*h).tmp3, (*h).w3 as *const f32, (*h).dt3 as *const f32, 512, 256, B);
    fast_transpose((*h).delta2, (*h).tmp3 as *const f32, 512, B);
    sgd_fused_tn((*h).w3, (*h).act3 as *const f32, (*h).delta3 as *const f32, 512, B, 256, lr);
    for k in 0..256 { *(*h).b3.add(k) -= lr * *(*h).db3.add(k); }

    // Layer 2: fused relu_mask + db2 reduce; then dX2 → delta1; SGD w2,b2
    std::ptr::write_bytes((*h).db2, 0, 512);
    for bi in 0..B { for j in 0..512 {
        let mask = if *(*h).pre2.add(bi*512+j) > 0.0 { 1.0f32 } else { 0.0 };
        let d = *(*h).delta2.add(bi*512+j) * mask;
        *(*h).delta2.add(bi*512+j) = d;
        *(*h).db2.add(j) += d;
    }}
    fast_transpose((*h).dt2, (*h).delta2 as *const f32, B, 512);
    sgemm((*h).tmp2, (*h).w2 as *const f32, (*h).dt2 as *const f32, 1024, 512, B);
    fast_transpose((*h).delta1, (*h).tmp2 as *const f32, 1024, B);
    sgd_fused_tn((*h).w2, (*h).act2 as *const f32, (*h).delta2 as *const f32, 1024, B, 512, lr);
    for k in 0..512 { *(*h).b2.add(k) -= lr * *(*h).db2.add(k); }

    // Layer 1: fused relu_mask + db1 reduce; then dX1 → delta0; SGD w1,b1
    std::ptr::write_bytes((*h).db1, 0, 1024);
    for bi in 0..B { for j in 0..1024 {
        let mask = if *(*h).pre1.add(bi*1024+j) > 0.0 { 1.0f32 } else { 0.0 };
        let d = *(*h).delta1.add(bi*1024+j) * mask;
        *(*h).delta1.add(bi*1024+j) = d;
        *(*h).db1.add(j) += d;
    }}
    fast_transpose((*h).dt1, (*h).delta1 as *const f32, B, 1024);
    sgemm((*h).tmp1, (*h).w1 as *const f32, (*h).dt1 as *const f32, 1024, 1024, B);
    fast_transpose((*h).delta0, (*h).tmp1 as *const f32, 1024, B);
    sgd_fused_tn((*h).w1, (*h).act1 as *const f32, (*h).delta1 as *const f32, 1024, B, 1024, lr);
    for k in 0..1024 { *(*h).b1.add(k) -= lr * *(*h).db1.add(k); }

    // Layer 0: fused relu_mask + db0 reduce; SGD w0,b0 (no dX needed for input)
    std::ptr::write_bytes((*h).db0, 0, 1024);
    for bi in 0..B { for j in 0..1024 {
        let mask = if *(*h).pre0.add(bi*1024+j) > 0.0 { 1.0f32 } else { 0.0 };
        let d = *(*h).delta0.add(bi*1024+j) * mask;
        *(*h).delta0.add(bi*1024+j) = d;
        *(*h).db0.add(j) += d;
    }}
    sgd_fused_tn((*h).w0, (*h).act0 as *const f32, (*h).delta0 as *const f32, 512, B, 1024, lr);
    for k in 0..1024 { *(*h).b0.add(k) -= lr * *(*h).db0.add(k); }
}
