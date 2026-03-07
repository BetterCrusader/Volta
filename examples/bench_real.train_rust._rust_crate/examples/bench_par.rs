// bench_par.rs — test different Rayon parallelism thresholds
#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

fn par_current(m: usize, k: usize, n: usize) -> gemm::Parallelism {
    let ops = 2 * m * k * n;
    if ops < (1 << 20) {
        gemm::Parallelism::None
    } else if ops < (1 << 25) {
        gemm::Parallelism::Rayon(5)
    } else {
        gemm::Parallelism::Rayon(0)
    }
}
fn par_all(m: usize, k: usize, n: usize) -> gemm::Parallelism {
    let ops = 2 * m * k * n;
    if ops < (1 << 18) {
        gemm::Parallelism::None
    } else {
        gemm::Parallelism::Rayon(0)
    }
}
fn par_none(_: usize, _: usize, _: usize) -> gemm::Parallelism {
    gemm::Parallelism::None
}

fn sgemm_with(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    k: usize,
    n: usize,
    par: gemm::Parallelism,
) {
    unsafe {
        gemm::gemm(
            m, n, k, c, 1isize, n as isize, false, a, 1isize, k as isize, b, 1isize, n as isize,
            0f32, 1f32, false, false, false, par,
        );
    }
}
fn sgd_with(
    w: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    k: usize,
    n: usize,
    lr: f32,
    par: gemm::Parallelism,
) {
    unsafe {
        gemm::gemm(
            m, n, k, w, 1isize, n as isize, true, a, m as isize, 1isize, b, 1isize, n as isize,
            1f32, -lr, false, false, false, par,
        );
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

fn bench<F: Fn(usize, usize, usize) -> gemm::Parallelism>(label: &str, par: F) {
    let mut rng = 42u64;
    let mut ws: [Vec<f32>; N] = std::array::from_fn(|i| vec![0.0f32; LAYERS[i] * LAYERS[i + 1]]);
    lcg_xavier(&mut ws[0], 512, 1024, &mut rng);
    lcg_xavier(&mut ws[1], 1024, 1024, &mut rng);
    lcg_xavier(&mut ws[2], 1024, 512, &mut rng);
    lcg_xavier(&mut ws[3], 512, 256, &mut rng);
    lcg_xavier(&mut ws[4], 256, 1, &mut rng);
    let mut bs: [Vec<f32>; N] = std::array::from_fn(|i| vec![0.0f32; LAYERS[i + 1]]);
    let mut pres: [Vec<f32>; 4] = std::array::from_fn(|i| vec![0.0f32; B * LAYERS[i + 1]]);
    let mut acts: [Vec<f32>; 6] = std::array::from_fn(|i| vec![0.0f32; B * LAYERS[i]]);
    let mut deltas: [Vec<f32>; N] = std::array::from_fn(|i| vec![0.0f32; B * LAYERS[i + 1]]);
    let mut dts: [Vec<f32>; N] = std::array::from_fn(|i| vec![0.0f32; B * LAYERS[i + 1]]);
    let mut tmps: [Vec<f32>; N] = std::array::from_fn(|i| vec![0.0f32; B * LAYERS[i]]);
    let x: Vec<f32> = (0..B * 512)
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let y: Vec<f32> = (0..B).map(|i| (i % 7) as f32 * 0.1).collect();
    let lr = 0.01f32;

    let step = |ws: &mut [Vec<f32>; N],
                bs: &mut [Vec<f32>; N],
                pres: &mut [Vec<f32>; 4],
                acts: &mut [Vec<f32>; 6],
                deltas: &mut [Vec<f32>; N],
                dts: &mut [Vec<f32>; N],
                tmps: &mut [Vec<f32>; N]| {
        acts[0].copy_from_slice(&x);
        for i in 0..N {
            let (r, c) = (LAYERS[i], LAYERS[i + 1]);
            let p = par(B, r, c);
            if i == N - 1 {
                sgemm_with(
                    acts[N].as_mut_ptr(),
                    acts[i].as_ptr(),
                    ws[i].as_ptr(),
                    B,
                    r,
                    c,
                    p,
                );
                for bi in 0..B {
                    for j in 0..c {
                        acts[N][bi * c + j] += bs[i][j];
                    }
                }
            } else {
                sgemm_with(
                    pres[i].as_mut_ptr(),
                    acts[i].as_ptr(),
                    ws[i].as_ptr(),
                    B,
                    r,
                    c,
                    p,
                );
                for bi in 0..B {
                    for j in 0..c {
                        let v = pres[i][bi * c + j] + bs[i][j];
                        pres[i][bi * c + j] = v;
                        acts[i + 1][bi * c + j] = if v > 0.0 { v } else { 0.0 };
                    }
                }
            }
        }
        let nt = B;
        for k in 0..nt {
            let d = acts[N][k] - y[k];
            deltas[N - 1][k] = 2.0 * d / (nt as f32);
        }
        for i in (0..N).rev() {
            let (r, c) = (LAYERS[i], LAYERS[i + 1]);
            let p = par(r, c, B);
            if i != N - 1 {
                for bi in 0..B {
                    for j in 0..c {
                        let m = if pres[i][bi * c + j] > 0.0 {
                            1.0f32
                        } else {
                            0.0
                        };
                        deltas[i][bi * c + j] *= m;
                    }
                }
                for k in 0..c {
                    bs[i][k] -= lr * (0..B).map(|bi| deltas[i][bi * c + k]).sum::<f32>();
                }
            }
            if i > 0 {
                let (tr, tc) = (B, c);
                for ii in 0..tr {
                    for jj in 0..tc {
                        dts[i][jj * tr + ii] = deltas[i][ii * tc + jj];
                    }
                }
                sgemm_with(
                    tmps[i].as_mut_ptr(),
                    ws[i].as_ptr(),
                    dts[i].as_ptr(),
                    r,
                    c,
                    B,
                    p,
                );
                for ii in 0..r {
                    for jj in 0..B {
                        deltas[i - 1][jj * r + ii] = tmps[i][ii * B + jj];
                    }
                }
            }
            sgd_with(
                ws[i].as_mut_ptr(),
                acts[i].as_ptr(),
                deltas[i].as_ptr(),
                r,
                B,
                c,
                lr,
                par(r, B, c),
            );
        }
    };

    for _ in 0..10 {
        step(
            &mut ws,
            &mut bs,
            &mut pres,
            &mut acts,
            &mut deltas,
            &mut dts,
            &mut tmps,
        );
    }
    let mut results = [0.0f64; 7];
    for run in 0..7 {
        let t0 = Instant::now();
        for _ in 0..50 {
            step(
                &mut ws,
                &mut bs,
                &mut pres,
                &mut acts,
                &mut deltas,
                &mut dts,
                &mut tmps,
            );
        }
        results[run] = t0.elapsed().as_nanos() as f64 / 1000.0 / 50.0 / 1000.0;
    }
    common::sort_f64_samples(&mut results);
    println!(
        "[{}] median={:.3} ms  all={:?}",
        label,
        results[3],
        results.map(|x| format!("{:.3}", x))
    );
}

fn main() {
    bench("par_current", par_current);
    bench("par_all_rayon0", par_all);
    bench("par_none", par_none);
    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(c: u32);
        }
        ExitProcess(0);
    }
}
