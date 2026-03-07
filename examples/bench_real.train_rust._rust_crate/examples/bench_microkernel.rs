// bench_microkernel.rs — custom AVX-512 GEMM for M=64 (batch dimension)
// Strategy: C[64×N] = A[64×K] @ B[K×N]
//   - Tile N by 64 (4 zmm registers = 64 floats)
//   - For each N-tile: load B column strip into zmm, accumulate over K
//   - 64 rows of A processed sequentially (outer loop)
//   - All 64 output rows for current N-tile in registers simultaneously
//
// This avoids BLIS packing overhead and exploits the fixed M=64 structure.
//
// Run: cargo run --release --example bench_microkernel
#![allow(non_snake_case)]
use std::time::Instant;

const B: usize = 64;  // batch size = M dimension

/// C[M×N] += A[M×K] @ B_mat[K×N], row-major, M=64 fixed
/// Uses AVX-512: tile N by 16, accumulate K in zmm registers
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sgemm_m64_add(
    c: *mut f32,        // [M×N] row-major, beta=1 (accumulate)
    a: *const f32,      // [M×K] row-major
    b: *const f32,      // [K×N] row-major
    m: usize, k: usize, n: usize,
    alpha: f32,
) {
    use std::arch::x86_64::*;
    let valpha = _mm512_set1_ps(alpha);

    // Tile over N by 16
    let mut jj = 0usize;
    while jj + 16 <= n {
        // For each of the M=64 output rows, compute dot product with B columns jj..jj+16
        for i in 0..m {
            // acc[0..16] = sum over k of A[i,k] * B[k, jj..jj+16]
            let mut acc = _mm512_setzero_ps();
            for kk in 0..k {
                let a_ik = _mm512_set1_ps(*a.add(i*k + kk));
                let b_row = _mm512_loadu_ps(b.add(kk*n + jj));
                acc = _mm512_fmadd_ps(a_ik, b_row, acc);
            }
            // C[i, jj..jj+16] += alpha * acc
            let c_ptr = c.add(i*n + jj);
            let c_old = _mm512_loadu_ps(c_ptr);
            _mm512_storeu_ps(c_ptr, _mm512_fmadd_ps(valpha, acc, c_old));
        }
        jj += 16;
    }
    // Remainder columns
    if jj < n {
        let rem = (n - jj) as u16;
        let mask = (1u16 << rem) - 1;
        for i in 0..m {
            let mut acc = _mm512_setzero_ps();
            for kk in 0..k {
                let a_ik = _mm512_set1_ps(*a.add(i*k + kk));
                let b_row = _mm512_maskz_loadu_ps(mask, b.add(kk*n + jj));
                acc = _mm512_fmadd_ps(a_ik, b_row, acc);
            }
            let c_ptr = c.add(i*n + jj);
            let c_old = _mm512_maskz_loadu_ps(mask, c_ptr);
            _mm512_mask_storeu_ps(c_ptr, mask, _mm512_fmadd_ps(valpha, acc, c_old));
        }
    }
}

/// C[M×N] = A[M×K] @ B_mat[K×N], beta=0 (replace)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sgemm_m64(
    c: *mut f32, a: *const f32, b: *const f32,
    m: usize, k: usize, n: usize,
) {
    use std::arch::x86_64::*;
    // Zero output first
    std::ptr::write_bytes(c, 0, m * n);
    sgemm_m64_add(c, a, b, m, k, n, 1.0f32);
}

/// Better version: tile K as well to improve cache reuse
/// Process K in blocks of KT to keep B tiles in L1
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sgemm_m64_tiled(
    c: *mut f32, a: *const f32, b: *const f32,
    m: usize, k: usize, n: usize,
) {
    use std::arch::x86_64::*;
    // Zero output
    std::ptr::write_bytes(c, 0, m * n);

    // Tile sizes tuned for L1/L2 cache
    // K-tile: keep A[64×KT] + B[KT×NT] in L2
    // KT=32: A tile = 64×32×4 = 8KB, B tile = 32×NT×4
    // NT=64: B tile = 32×64×4 = 8KB — fits in L1 (32KB)
    const KT: usize = 32;
    const NT: usize = 64;  // must be multiple of 16 for AVX-512

    let mut kk = 0usize;
    while kk < k {
        let ke = (kk + KT).min(k);
        let klen = ke - kk;

        let mut jj = 0usize;
        while jj + NT <= n {
            // Inner: C[m × NT] += A[m × klen] @ B[klen × NT]
            for i in 0..m {
                // Load 4 zmm accumulators = 4×16 = 64 outputs for row i, cols jj..jj+NT
                let mut acc0 = _mm512_setzero_ps();
                let mut acc1 = _mm512_setzero_ps();
                let mut acc2 = _mm512_setzero_ps();
                let mut acc3 = _mm512_setzero_ps();
                for kp in kk..ke {
                    let a_val = _mm512_set1_ps(*a.add(i*k + kp));
                    acc0 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(b.add(kp*n + jj     )), acc0);
                    acc1 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(b.add(kp*n + jj + 16)), acc1);
                    acc2 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(b.add(kp*n + jj + 32)), acc2);
                    acc3 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(b.add(kp*n + jj + 48)), acc3);
                }
                let cp = c.add(i*n + jj);
                _mm512_storeu_ps(cp,      _mm512_add_ps(_mm512_loadu_ps(cp),      acc0));
                _mm512_storeu_ps(cp.add(16), _mm512_add_ps(_mm512_loadu_ps(cp.add(16)), acc1));
                _mm512_storeu_ps(cp.add(32), _mm512_add_ps(_mm512_loadu_ps(cp.add(32)), acc2));
                _mm512_storeu_ps(cp.add(48), _mm512_add_ps(_mm512_loadu_ps(cp.add(48)), acc3));
            }
            jj += NT;
        }
        // Handle remaining N columns (not multiple of NT)
        while jj + 16 <= n {
            for i in 0..m {
                let mut acc = _mm512_setzero_ps();
                for kp in kk..ke {
                    let a_val = _mm512_set1_ps(*a.add(i*k + kp));
                    acc = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(b.add(kp*n + jj)), acc);
                }
                let cp = c.add(i*n + jj);
                _mm512_storeu_ps(cp, _mm512_add_ps(_mm512_loadu_ps(cp), acc));
            }
            jj += 16;
        }
        if jj < n {
            let rem = (n - jj) as u16;
            let mask = (1u16 << rem) - 1;
            for i in 0..m {
                let mut acc = _mm512_setzero_ps();
                for kp in kk..ke {
                    let a_val = _mm512_set1_ps(*a.add(i*k + kp));
                    acc = _mm512_fmadd_ps(a_val, _mm512_maskz_loadu_ps(mask, b.add(kp*n + jj)), acc);
                }
                let cp = c.add(i*n + jj);
                _mm512_mask_storeu_ps(cp, mask, _mm512_add_ps(_mm512_maskz_loadu_ps(mask, cp), acc));
            }
        }
        kk += KT;
    }
}

/// SGD update: W[R×C] -= lr * A[M×R]^T @ delta[M×C]
/// = W[R×C] -= lr * sum_i(a[i,r] * delta[i,c]) for each r,c
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sgd_m64_tiled(
    w: *mut f32, a: *const f32, delta: *const f32,
    r: usize, m: usize, c: usize, lr: f32,
) {
    use std::arch::x86_64::*;
    let neg_lr = _mm512_set1_ps(-lr);
    const MT: usize = 16; // tile over M dimension
    const CT: usize = 64; // tile over C dimension

    for rr in 0..r {
        let mut cc = 0usize;
        while cc + CT <= c {
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            // sum over M: a[i,rr] * delta[i, cc..cc+CT]
            let mut mi = 0usize;
            while mi + MT <= m {
                for ii in mi..mi+MT {
                    let a_val = _mm512_set1_ps(*a.add(ii*r + rr));
                    acc0 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(delta.add(ii*c + cc     )), acc0);
                    acc1 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(delta.add(ii*c + cc + 16)), acc1);
                    acc2 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(delta.add(ii*c + cc + 32)), acc2);
                    acc3 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(delta.add(ii*c + cc + 48)), acc3);
                }
                mi += MT;
            }
            for ii in mi..m {
                let a_val = _mm512_set1_ps(*a.add(ii*r + rr));
                acc0 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(delta.add(ii*c + cc     )), acc0);
                acc1 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(delta.add(ii*c + cc + 16)), acc1);
                acc2 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(delta.add(ii*c + cc + 32)), acc2);
                acc3 = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(delta.add(ii*c + cc + 48)), acc3);
            }
            // W[rr, cc..cc+CT] += -lr * acc
            let wp = w.add(rr*c + cc);
            _mm512_storeu_ps(wp,      _mm512_fmadd_ps(neg_lr, acc0, _mm512_loadu_ps(wp     )));
            _mm512_storeu_ps(wp.add(16), _mm512_fmadd_ps(neg_lr, acc1, _mm512_loadu_ps(wp.add(16))));
            _mm512_storeu_ps(wp.add(32), _mm512_fmadd_ps(neg_lr, acc2, _mm512_loadu_ps(wp.add(32))));
            _mm512_storeu_ps(wp.add(48), _mm512_fmadd_ps(neg_lr, acc3, _mm512_loadu_ps(wp.add(48))));
            cc += CT;
        }
        // remainder C
        while cc + 16 <= c {
            let mut acc = _mm512_setzero_ps();
            for ii in 0..m {
                let a_val = _mm512_set1_ps(*a.add(ii*r + rr));
                acc = _mm512_fmadd_ps(a_val, _mm512_loadu_ps(delta.add(ii*c + cc)), acc);
            }
            let wp = w.add(rr*c + cc);
            _mm512_storeu_ps(wp, _mm512_fmadd_ps(neg_lr, acc, _mm512_loadu_ps(wp)));
            cc += 16;
        }
        if cc < c {
            let rem = (c - cc) as u16;
            let mask = (1u16 << rem) - 1;
            let mut acc = _mm512_setzero_ps();
            for ii in 0..m {
                let a_val = _mm512_set1_ps(*a.add(ii*r + rr));
                acc = _mm512_fmadd_ps(a_val, _mm512_maskz_loadu_ps(mask, delta.add(ii*c + cc)), acc);
            }
            let wp = w.add(rr*c + cc);
            _mm512_mask_storeu_ps(wp, mask, _mm512_fmadd_ps(neg_lr, acc, _mm512_maskz_loadu_ps(mask, wp)));
        }
    }
}

fn lcg(v: &mut [f32], rng: &mut u64) {
    for x in v.iter_mut() {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((*rng >> 11) as f32) / (1u64 << 53) as f32 * 2.0 - 1.0;
    }
}

fn gemm_ref(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    unsafe {
        gemm::gemm(m,n,k, c.as_mut_ptr(),1,n as isize,false,
            a.as_ptr(),1,k as isize, b.as_ptr(),1,n as isize,
            0f32,1f32,false,false,false, gemm::Parallelism::Rayon(0));
    }
}

fn bench_fn(label: &str, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 { f(); }
    let mut results = [0f64; 7];
    for run in 0..7 {
        let t0 = Instant::now();
        for _ in 0..iters { f(); }
        results[run] = t0.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
    }
    results.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let med = results[3];
    println!("  [{:<42}] p50={:.1} us", label, med);
    med
}

fn main() {
    let mut rng = 42u64;

    println!("=== Custom AVX-512 GEMM vs gemm-crate (M=64) ===\n");

    // Test forward shapes: C[64×N] = A[64×K] @ B[K×N]
    for &(k, n) in &[(512usize,1024usize),(1024,1024),(1024,512),(512,256),(256,1usize)] {
        println!("Forward [{}×{}] @ [{}×{}]:", B,k, k,n);
        let mut a = vec![0f32; B*k]; lcg(&mut a, &mut rng);
        let mut b = vec![0f32; k*n]; lcg(&mut b, &mut rng);
        let mut c_ref = vec![0f32; B*n];
        let mut c_ours = vec![0f32; B*n];

        let t_ref = bench_fn(&format!("gemm_crate [{}×{}]@[{}×{}]", B,k,k,n), 200, || {
            gemm_ref(&mut c_ref, &a, &b, B, k, n);
        });
        let t_tiled = bench_fn(&format!("sgemm_m64_tiled [{}×{}]@[{}×{}]", B,k,k,n), 200, || {
            unsafe { sgemm_m64_tiled(c_ours.as_mut_ptr(), a.as_ptr(), b.as_ptr(), B, k, n); }
        });

        // Verify correctness
        gemm_ref(&mut c_ref, &a, &b, B, k, n);
        unsafe { sgemm_m64_tiled(c_ours.as_mut_ptr(), a.as_ptr(), b.as_ptr(), B, k, n); }
        let max_err = c_ref.iter().zip(c_ours.iter()).map(|(r,o)| (r-o).abs()).fold(0f32, f32::max);
        println!("  max_err={:.2e}  speedup={:.2}x", max_err, t_ref/t_tiled);
        println!();
    }

    // Test SGD shapes: W[R×C] -= lr * A[64×R]^T @ delta[64×C]
    println!("=== SGD update ===\n");
    let lr = 0.01f32;
    for &(r, c) in &[(512usize,1024usize),(1024,1024),(1024,512),(512,256)] {
        println!("SGD [{}×{}]:", r,c);
        let mut act   = vec![0f32; B*r]; lcg(&mut act, &mut rng);
        let mut delta = vec![0f32; B*c]; lcg(&mut delta, &mut rng);
        let mut w1 = vec![0.1f32; r*c];
        let mut w2 = vec![0.1f32; r*c];

        let t_ref = bench_fn(&format!("gemm_crate SGD [{}×{}]", r,c), 200, || {
            unsafe { gemm::gemm(r,c,B, w1.as_mut_ptr(),1,c as isize,true,
                act.as_ptr(),r as isize,1, delta.as_ptr(),1,c as isize,
                1f32,-lr,false,false,false, gemm::Parallelism::Rayon(0)); }
        });
        let t_ours = bench_fn(&format!("sgd_m64_tiled   [{}×{}]", r,c), 200, || {
            unsafe { sgd_m64_tiled(w2.as_mut_ptr(), act.as_ptr(), delta.as_ptr(), r, B, c, lr); }
        });
        println!("  speedup={:.2}x\n", t_ref/t_ours);
    }

    #[cfg(windows)]
    unsafe { extern "system" { fn ExitProcess(c: u32); } ExitProcess(0); }
}
