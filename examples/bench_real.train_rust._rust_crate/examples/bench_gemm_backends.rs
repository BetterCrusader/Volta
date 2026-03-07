// bench_gemm_backends.rs — gemm crate vs faer for our specific matrix sizes
// Run: cargo run --release --example bench_gemm_backends
#![allow(non_snake_case)]
#[path = "common/mod.rs"]
mod common;

use faer::Mat;
use std::time::Instant;

fn par_gemm(m: usize, k: usize, n: usize) -> gemm::Parallelism {
    let ops = 2 * m * k * n;
    if ops < (1 << 20) {
        gemm::Parallelism::None
    } else if ops < (1 << 25) {
        gemm::Parallelism::Rayon(5)
    } else {
        gemm::Parallelism::Rayon(0)
    }
}

fn lcg_fill(v: &mut [f32], rng: &mut u64) {
    for x in v.iter_mut() {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = ((*rng >> 11) & ((1u64 << 53) - 1)) as f32 / (1u64 << 53) as f32;
        *x = f * 2.0 - 1.0;
    }
}

fn bench_fn(label: &str, mut f: impl FnMut()) -> f64 {
    for _ in 0..10 {
        f();
    }
    let mut results = [0.0f64; 7];
    for slot in &mut results {
        let t0 = Instant::now();
        for _ in 0..200 {
            f();
        }
        *slot = t0.elapsed().as_nanos() as f64 / 200.0 / 1000.0;
    }
    let med = common::median_f64_samples(&mut results);
    println!("  [{:<40}] median={:.1} us", label, med);
    med
}

fn main() {
    let mut rng = 42u64;
    let B = 64usize;

    // Test: C[B×n] = A[B×k] @ B_mat[k×n]  (forward pass shape)
    for &(k, n) in &[(512usize, 1024usize), (1024, 1024), (1024, 512), (512, 256)] {
        println!(
            "\n--- Forward: [{}x{}] @ [{}x{}] = [{}x{}] ---",
            B, k, k, n, B, n
        );
        let mut a = vec![0f32; B * k];
        lcg_fill(&mut a, &mut rng);
        let mut b = vec![0f32; k * n];
        lcg_fill(&mut b, &mut rng);
        let mut c = vec![0f32; B * n];

        // gemm crate
        bench_fn("gemm_crate C=A@B", || unsafe {
            gemm::gemm(
                B,
                n,
                k,
                c.as_mut_ptr(),
                1,
                n as isize,
                false,
                a.as_ptr(),
                1,
                k as isize,
                b.as_ptr(),
                1,
                n as isize,
                0f32,
                1f32,
                false,
                false,
                false,
                par_gemm(B, k, n),
            );
        });

        // faer via Mat views
        let fa = Mat::<f32>::from_fn(B, k, |i, j| a[i * k + j]);
        let fb = Mat::<f32>::from_fn(k, n, |i, j| b[i * n + j]);
        let mut fc = Mat::<f32>::zeros(B, n);
        bench_fn("faer matmul", || {
            faer::linalg::matmul::matmul(
                fc.as_mut(),
                faer::Accum::Replace,
                fa.as_ref(),
                fb.as_ref(),
                1.0f32,
                faer::get_global_parallelism(),
            );
        });
    }

    // Test: backward pass C[B×r] = A[B×c] @ B^T[c×r]  (bwd shape, r>c layers)
    for &(r, c) in &[(1024usize, 512usize), (512, 256), (256, 1)] {
        println!(
            "\n--- Backward (r>c): [{}x{}] @ [{}x{}]^T = [{}x{}] ---",
            B, c, r, c, B, r
        );
        let mut delta = vec![0f32; B * c];
        lcg_fill(&mut delta, &mut rng);
        let mut w = vec![0f32; r * c];
        lcg_fill(&mut w, &mut rng);
        let mut dp = vec![0f32; B * r];

        bench_fn("gemm_crate delta@W^T (stride)", || unsafe {
            gemm::gemm(
                B,
                r,
                c,
                dp.as_mut_ptr(),
                1,
                r as isize,
                false,
                delta.as_ptr(),
                1,
                c as isize,
                w.as_ptr(),
                c as isize,
                1isize,
                0f32,
                1f32,
                false,
                false,
                false,
                par_gemm(B, c, r),
            );
        });

        let fd = Mat::<f32>::from_fn(B, c, |i, j| delta[i * c + j]);
        let fw = Mat::<f32>::from_fn(r, c, |i, j| w[i * c + j]);
        let mut fdp = Mat::<f32>::zeros(B, r);
        bench_fn("faer delta@W^T (explicit transpose)", || {
            faer::linalg::matmul::matmul(
                fdp.as_mut(),
                faer::Accum::Replace,
                fd.as_ref(),
                fw.transpose(),
                1.0f32,
                faer::get_global_parallelism(),
            );
        });
    }

    // Test SGD: W[r×c] -= lr * A^T[r×B] @ B[B×c]
    println!("\n--- SGD update ---");
    for &(r, c) in &[(512usize, 1024usize), (1024, 1024), (1024, 512), (512, 256)] {
        println!("  SGD {}x{}", r, c);
        let act = vec![0f32; B * r];
        let delta = vec![0f32; B * c];
        let mut w1 = vec![0.1f32; r * c];
        let w2 = vec![0.1f32; r * c];
        let lr = 0.01f32;

        bench_fn(&format!("gemm_crate SGD {}x{}", r, c), || unsafe {
            gemm::gemm(
                r,
                c,
                B,
                w1.as_mut_ptr(),
                1,
                c as isize,
                true,
                act.as_ptr(),
                r as isize,
                1isize,
                delta.as_ptr(),
                1,
                c as isize,
                1f32,
                -lr,
                false,
                false,
                false,
                par_gemm(r, B, c),
            );
        });

        let fa2 = Mat::<f32>::from_fn(B, r, |i, j| act[i * r + j]);
        let fd2 = Mat::<f32>::from_fn(B, c, |i, j| delta[i * c + j]);
        let mut fw2 = Mat::<f32>::from_fn(r, c, |i, j| w2[i * c + j]);
        bench_fn(&format!("faer SGD {}x{}", r, c), || {
            faer::linalg::matmul::matmul(
                fw2.as_mut(),
                faer::Accum::Add,
                fa2.transpose(),
                fd2.as_ref(),
                -lr,
                faer::get_global_parallelism(),
            );
        });
    }

    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(code: u32);
        }
        ExitProcess(0);
    }
}
