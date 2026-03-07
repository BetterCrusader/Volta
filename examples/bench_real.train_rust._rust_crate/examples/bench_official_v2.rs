// bench_official_v2.rs — official-style benchmark matching bench_train_volta_rust.c
// Uses same params: B=64, steps=50, runs=7, warmup=10, deterministic input
// Exits cleanly via ExitProcess to avoid Rayon DLL abort issue
#![allow(non_snake_case)]
#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

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
fn fast_transpose(dst: *mut f32, src: *const f32, rows: usize, cols: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if rows.is_multiple_of(8) && cols.is_multiple_of(8) {
            let mut i = 0;
            while i < rows {
                let mut j = 0;
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
        // FORWARD — fused bias+relu
        self.acts[0].copy_from_slice(x);
        for i in 0..N {
            let (r, c) = (LAYERS[i], LAYERS[i + 1]);
            let is_last = i == N - 1;
            if is_last {
                sgemm(
                    self.acts[N].as_mut_ptr(),
                    self.acts[i].as_ptr(),
                    self.ws[i].as_ptr(),
                    B,
                    r,
                    c,
                );
                for bi in 0..B {
                    for j in 0..c {
                        self.acts[N][bi * c + j] += self.bs[i][j];
                    }
                }
            } else {
                sgemm(
                    self.pres[i].as_mut_ptr(),
                    self.acts[i].as_ptr(),
                    self.ws[i].as_ptr(),
                    B,
                    r,
                    c,
                );
                for bi in 0..B {
                    for j in 0..c {
                        let v = self.pres[i][bi * c + j] + self.bs[i][j];
                        self.pres[i][bi * c + j] = v;
                        self.acts[i + 1][bi * c + j] = if v > 0.0 { v } else { 0.0 };
                    }
                }
            }
        }
        // MSE loss
        let nt = B * LAYERS[N];
        let mut lacc = 0.0f32;
        for k in 0..nt {
            let d = self.acts[N][k] - y[k];
            lacc += d * d;
            self.deltas[N - 1][k] = 2.0 * d / (nt as f32);
        }
        self.last_loss = lacc / (nt as f32);

        // BACKWARD — fused relu_mask+db_reduce
        for i in (0..N).rev() {
            let (r, c) = (LAYERS[i], LAYERS[i + 1]);
            let is_last = i == N - 1;
            if !is_last {
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
                for k in 0..c {
                    self.bs[i][k] -= lr * self.dbs[i][k];
                }
            }
            if i > 0 {
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
    // warmup
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
    let med = common::median_f64_samples(&mut results);
    println!(
        "[TRAIN][VOLTA-RUST-V2] median={:.3} ms/step  all7={:?}",
        med,
        results.map(|x| format!("{:.3}", x))
    );
    println!("  checksum={:.6}", checksum);
    println!(
        "  batch={}  steps={}  in={}  out={}",
        B, STEPS, LAYERS[0], LAYERS[N]
    );

    // Use ExitProcess to avoid Rayon abort on thread pool teardown
    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(code: u32);
        }
        ExitProcess(0);
    }
}
