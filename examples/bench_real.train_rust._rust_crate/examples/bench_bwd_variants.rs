// bench_bwd_variants.rs — compare backward pass strategies
// Tests different approaches to computing delta_prev = W @ delta^T without allocating dt
// Run: cargo run --release --example bench_bwd_variants
#![allow(
    non_snake_case,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]
#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

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

// Current approach: transpose delta, then sgemm(W, dt)
// C[r×B] = W[r×c] @ dt[c×B]
fn bwd_transpose_sgemm(
    delta_prev: *mut f32,
    tmp: *mut f32,
    dt: *mut f32,
    w: *const f32,
    delta: *const f32,
    r: usize,
    c: usize,
    batch: usize,
) {
    // transpose delta [B×c] → dt [c×B]
    const T: usize = 32;
    let mut i = 0;
    while i < batch {
        let imax = (i + T).min(batch);
        let mut j = 0;
        while j < c {
            let jmax = (j + T).min(c);
            unsafe {
                for ii in i..imax {
                    for jj in j..jmax {
                        *dt.add(jj * batch + ii) = *delta.add(ii * c + jj);
                    }
                }
            }
            j += T;
        }
        i += T;
    }
    // sgemm(tmp [r×B]) = W [r×c] @ dt [c×B]
    unsafe {
        gemm::gemm(
            r,
            batch,
            c,
            tmp,
            1isize,
            batch as isize,
            false,
            w,
            1isize,
            c as isize,
            dt,
            1isize,
            batch as isize,
            0f32,
            1f32,
            false,
            false,
            false,
            par(r, c, batch),
        );
    }
    // transpose tmp [r×B] → delta_prev [B×r]
    let mut i = 0;
    while i < r {
        let imax = (i + T).min(r);
        let mut j = 0;
        while j < batch {
            let jmax = (j + T).min(batch);
            unsafe {
                for ii in i..imax {
                    for jj in j..jmax {
                        *delta_prev.add(jj * r + ii) = *tmp.add(ii * batch + jj);
                    }
                }
            }
            j += T;
        }
        i += T;
    }
}

// Variant A: direct C = delta[B×c] @ W^T[c×r] — no transposes
// W is row-major [r×c]: W[i,j] = w[i*c+j]
// W^T is [c×r]: W^T[j,i] = W[i,j], so reading W with rhs_cs=c (col-stride), rhs_rs=1 (row-stride)
// Output: delta_prev[B×r] row-major
fn bwd_delta_Wt(
    delta_prev: *mut f32,
    w: *const f32,
    delta: *const f32,
    r: usize,
    c: usize,
    batch: usize,
) {
    // C[B×r] = delta[B×c] @ W^T
    // gemm args: m=B, n=r, k=c
    // dst=delta_prev[B×r], dst_cs=1, dst_rs=r
    // lhs=delta[B×c], lhs_cs=1, lhs_rs=c (row-major)
    // rhs=w (reading as W^T[c×r]): rhs_cs=c, rhs_rs=1
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

// Variant B: C = W[r×c] @ delta^T — pass delta with transposed strides, still need 1 final transpose
fn bwd_sgemm_trans_b(
    delta_prev: *mut f32,
    tmp: *mut f32,
    w: *const f32,
    delta: *const f32,
    r: usize,
    c: usize,
    batch: usize,
) {
    // tmp[r×B] = W[r×c] @ delta^T[c×B]
    // delta^T strides: col-stride=c (rhs_cs), row-stride=1 (rhs_rs)
    unsafe {
        gemm::gemm(
            r,
            batch,
            c,
            tmp,
            1isize,
            batch as isize,
            false,
            w,
            1isize,
            c as isize,
            delta,
            c as isize,
            1isize,
            0f32,
            1f32,
            false,
            false,
            false,
            par(r, c, batch),
        );
    }
    // transpose tmp [r×B] → delta_prev [B×r]
    const T: usize = 32;
    let mut i = 0;
    while i < r {
        let imax = (i + T).min(r);
        let mut j = 0;
        while j < batch {
            let jmax = (j + T).min(batch);
            unsafe {
                for ii in i..imax {
                    for jj in j..jmax {
                        *delta_prev.add(jj * r + ii) = *tmp.add(ii * batch + jj);
                    }
                }
            }
            j += T;
        }
        i += T;
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

fn bench_variant(label: &str, r: usize, c: usize, batch: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    let mut results = [0.0f64; 7];
    for run in 0..7 {
        let t0 = Instant::now();
        for _ in 0..200 {
            f();
        }
        results[run] = t0.elapsed().as_nanos() as f64 / 200.0 / 1000.0;
    }
    let med = common::median_f64_samples(&mut results);
    println!(
        "[{}] r={} c={} B={} median={:.1} us",
        label, r, c, batch, med
    );
    med
}

fn main() {
    let mut rng = 42u64;
    let batch = 64usize;

    // Test on the largest layer: W1 = 1024×1024
    for &(r, c) in &[(512usize, 1024usize), (1024, 1024), (1024, 512), (512, 256)] {
        let mut w = vec![0.0f32; r * c];
        lcg_fill(&mut w, &mut rng);
        let mut delta = vec![0.0f32; batch * c];
        lcg_fill(&mut delta, &mut rng);
        let mut delta_prev = vec![0.0f32; batch * r];
        let mut tmp = vec![0.0f32; r * batch];
        let mut dt = vec![0.0f32; c * batch];

        let (wp, dp, dpp, tmpp, dtp) = (
            w.as_ptr(),
            delta.as_ptr(),
            delta_prev.as_mut_ptr(),
            tmp.as_mut_ptr(),
            dt.as_mut_ptr(),
        );

        bench_variant("baseline (transpose+sgemm+transpose)", r, c, batch, || {
            bwd_transpose_sgemm(dpp, tmpp, dtp, wp, dp, r, c, batch);
        });
        bench_variant("varA (delta @ W^T direct)", r, c, batch, || {
            bwd_delta_Wt(dpp, wp, dp, r, c, batch);
        });
        bench_variant(
            "varB (sgemm_Wdt_no_dt_alloc + 1 transpose)",
            r,
            c,
            batch,
            || {
                bwd_sgemm_trans_b(dpp, tmpp, wp, dp, r, c, batch);
            },
        );
        println!();
    }

    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(code: u32);
        }
        ExitProcess(0);
    }
}
