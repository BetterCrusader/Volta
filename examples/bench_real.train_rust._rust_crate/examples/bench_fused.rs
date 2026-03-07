// bench_fused.rs — 3-in-1 fused GEMM: forward + backward + SGD in one W pass
// Run: cargo run --release --example bench_fused
#![allow(non_snake_case)]
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
fn bwd_delta_Wt(dp: *mut f32, w: *const f32, d: *const f32, r: usize, c: usize, batch: usize) {
    unsafe {
        gemm::gemm(
            batch,
            r,
            c,
            dp,
            1isize,
            r as isize,
            false,
            d,
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

// ── Baseline: separate fwd / bwd / sgd calls ─────────────────────────────────
fn baseline_step(
    w: &mut [f32],
    act_in: &[f32],
    act_out: &mut [f32],
    pre: &mut [f32],
    delta_in: &[f32],
    delta_prev: &mut [f32],
    r: usize,
    c: usize,
    batch: usize,
    lr: f32,
) {
    // fwd: act_out[B×c] = act_in[B×r] @ W[r×c]
    sgemm(
        act_out.as_mut_ptr(),
        act_in.as_ptr(),
        w.as_ptr(),
        batch,
        r,
        c,
    );
    // bwd: delta_prev[B×r] = delta_in[B×c] @ W^T
    if r > c {
        bwd_delta_Wt(
            delta_prev.as_mut_ptr(),
            w.as_ptr(),
            delta_in.as_ptr(),
            r,
            c,
            batch,
        );
    } else {
        // transpose + sgemm + transpose
        let mut dt = vec![0f32; c * batch];
        let mut tmp = vec![0f32; r * batch];
        const T: usize = 32;
        let mut i = 0;
        while i < batch {
            let im = (i + T).min(batch);
            let mut j = 0;
            while j < c {
                let jm = (j + T).min(c);
                unsafe {
                    for ii in i..im {
                        for jj in j..jm {
                            *dt.as_mut_ptr().add(jj * batch + ii) =
                                *delta_in.as_ptr().add(ii * c + jj);
                        }
                    }
                }
                j += T;
            }
            i += T;
        }
        sgemm(tmp.as_mut_ptr(), w.as_ptr(), dt.as_ptr(), r, c, batch);
        let mut i = 0;
        while i < r {
            let im = (i + T).min(r);
            let mut j = 0;
            while j < batch {
                let jm = (j + T).min(batch);
                unsafe {
                    for ii in i..im {
                        for jj in j..jm {
                            *delta_prev.as_mut_ptr().add(jj * r + ii) =
                                *tmp.as_ptr().add(ii * batch + jj);
                        }
                    }
                }
                j += T;
            }
            i += T;
        }
    }
    // sgd: W -= lr * act_in^T @ delta_in
    sgd_fused_tn(
        w.as_mut_ptr(),
        act_in.as_ptr(),
        delta_in.as_ptr(),
        r,
        batch,
        c,
        lr,
    );
    let _ = pre;
}

// ── Fused: tile W[R×C], do fwd+bwd+sgd in one pass ───────────────────────────
// Strategy: tile over rows of W (r-dim). For each r-tile:
//   1. act_out[:, c_start..c_end] += act_in[:, ri..ri+RT] @ W[ri..ri+RT, :]   (fwd, accumulate over r-tiles)
//   2. delta_prev[:, ri..ri+RT]   += delta_in @ W[ri..ri+RT, :]^T             (bwd, accumulate over r-tiles)
//   3. W[ri..ri+RT, :] -= lr * act_in[:, ri..ri+RT]^T @ delta_in             (sgd, independent per tile)
//
// This requires act_out and delta_prev to be pre-zeroed and accumulated.
fn fused_step(
    w: &mut [f32],
    act_in: &[f32],
    act_out: &mut [f32],
    delta_in: &[f32],
    delta_prev: &mut [f32],
    r: usize,
    c: usize,
    batch: usize,
    lr: f32,
) {
    // zero outputs first
    act_out.iter_mut().for_each(|x| *x = 0.0);
    delta_prev.iter_mut().for_each(|x| *x = 0.0);

    const RT: usize = 64; // r-tile size — tune for L2 fit
                          // W tile: RT × c floats. For c=1024, RT=64: 64*1024*4 = 256KB (fits in L2)

    let mut ri = 0usize;
    while ri < r {
        let rt = RT.min(r - ri);
        let w_tile = unsafe { w.as_mut_ptr().add(ri * c) };
        let a_tile = unsafe { act_in.as_ptr().add(ri) }; // act_in[batch, ri..ri+rt], stride r
        let ap_tile = act_out.as_mut_ptr(); // act_out[batch, 0..c]
        let dp_tile = unsafe { delta_prev.as_mut_ptr().add(ri) }; // delta_prev[batch, ri..ri+rt]
        let d_tile = delta_in.as_ptr(); // delta_in[batch, 0..c]

        // 1) Forward: act_out[B×c] += act_in[:,ri:ri+rt][B×rt] @ W[ri:ri+rt, :][rt×c]
        //    act_in column-stride = r (non-contiguous), need to pack or use stride
        //    Use gemm with lhs_cs=r (column stride = r, row stride = 1 → column-major read of act_in slice)
        unsafe {
            gemm::gemm(
                batch,
                c,
                rt,
                ap_tile,
                1isize,
                c as isize,
                ri > 0, // accumulate after first tile
                a_tile,
                r as isize,
                1isize, // act_in col-major: col-stride=r, row-stride=1
                w_tile,
                1isize,
                c as isize,
                if ri > 0 { 1f32 } else { 0f32 },
                1f32,
                false,
                false,
                false,
                par(batch, rt, c),
            );
        }

        // 2) Backward: delta_prev[:,ri:ri+rt][B×rt] += delta_in[B×c] @ W[ri:ri+rt,:]^T[c×rt]
        //    W^T strides: col-stride=c, row-stride=1
        //    delta_prev col-stride = r (non-contiguous)
        unsafe {
            gemm::gemm(
                batch,
                rt,
                c,
                dp_tile,
                r as isize,
                1isize,
                ri > 0, // col-major write, accumulate
                d_tile,
                1isize,
                c as isize,
                w_tile,
                c as isize,
                1isize, // W as W^T
                if ri > 0 { 1f32 } else { 0f32 },
                1f32,
                false,
                false,
                false,
                par(batch, c, rt),
            );
        }

        // 3) SGD: W[ri:ri+rt, :] -= lr * act_in[:,ri:ri+rt]^T @ delta_in
        //    act_in^T[rt×B]: row-stride=r (col-major act_in), col-stride=1
        unsafe {
            gemm::gemm(
                rt,
                c,
                batch,
                w_tile,
                1isize,
                c as isize,
                true,
                a_tile,
                r as isize,
                1isize, // act_in^T: reading act_in col-major
                d_tile,
                1isize,
                c as isize,
                1f32,
                -lr,
                false,
                false,
                false,
                par(rt, batch, c),
            );
        }

        ri += RT;
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

fn verify_close(a: &[f32], b: &[f32], label: &str) -> bool {
    let max_err = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    if max_err > 1e-3 {
        println!("  MISMATCH {}: max_err={:.2e}", label, max_err);
        false
    } else {
        println!("  OK {}: max_err={:.2e}", label, max_err);
        true
    }
}

fn bench(label: &str, mut f: impl FnMut()) -> f64 {
    for _ in 0..10 {
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
        "[{}] median={:.1} us  all={:?}",
        label,
        med,
        results.map(|x| format!("{:.0}", x))
    );
    med
}

fn main() {
    let mut rng = 42u64;
    let batch = 64usize;
    let lr = 0.01f32;

    println!("=== Correctness check ===");
    for &(r, c) in &[(512usize, 1024usize), (1024, 1024), (1024, 512), (512, 256)] {
        let mut w_base = vec![0f32; r * c];
        lcg_fill(&mut w_base, &mut rng);
        let mut w_fuse = w_base.clone();
        let act_in = {
            let mut v = vec![0f32; batch * r];
            lcg_fill(&mut v, &mut rng);
            v
        };
        let delta_in = {
            let mut v = vec![0f32; batch * c];
            lcg_fill(&mut v, &mut rng);
            v
        };
        let mut pre = vec![0f32; batch * c];

        let mut ao_base = vec![0f32; batch * c];
        let mut dp_base = vec![0f32; batch * r];
        let mut ao_fuse = vec![0f32; batch * c];
        let mut dp_fuse = vec![0f32; batch * r];

        baseline_step(
            &mut w_base,
            &act_in,
            &mut ao_base,
            &mut pre,
            &delta_in,
            &mut dp_base,
            r,
            c,
            batch,
            lr,
        );
        fused_step(
            &mut w_fuse,
            &act_in,
            &mut ao_fuse,
            &delta_in,
            &mut dp_fuse,
            r,
            c,
            batch,
            lr,
        );

        println!("Layer {}x{}:", r, c);
        verify_close(&ao_base, &ao_fuse, "act_out");
        verify_close(&dp_base, &dp_fuse, "delta_prev");
        verify_close(&w_base, &w_fuse, "W_updated");
    }

    println!("\n=== Performance (per-layer, us) ===");
    for &(r, c) in &[(512usize, 1024usize), (1024, 1024), (1024, 512), (512, 256)] {
        let mut rng2 = 99u64;
        let mut w = vec![0f32; r * c];
        lcg_fill(&mut w, &mut rng2);
        let act_in = {
            let mut v = vec![0f32; batch * r];
            lcg_fill(&mut v, &mut rng2);
            v
        };
        let delta_in = {
            let mut v = vec![0f32; batch * c];
            lcg_fill(&mut v, &mut rng2);
            v
        };
        let mut pre = vec![0f32; batch * c];
        let mut ao = vec![0f32; batch * c];
        let mut dp = vec![0f32; batch * r];

        let mut wb = w.clone();
        let mut wf = w.clone();

        println!("--- {}x{} ---", r, c);
        bench(&format!("baseline {}x{}", r, c), || {
            baseline_step(
                &mut wb, &act_in, &mut ao, &mut pre, &delta_in, &mut dp, r, c, batch, lr,
            );
        });
        bench(&format!("fused    {}x{}", r, c), || {
            fused_step(
                &mut wf, &act_in, &mut ao, &delta_in, &mut dp, r, c, batch, lr,
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
