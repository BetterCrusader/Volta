// bench_mlp2048.rs — MLP 512→2048→2048→512→1, B=64, heavy workload
// Build: cargo build --release --example bench_mlp2048
#![allow(non_snake_case, unused)]
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
        if rows % 8 == 0 && cols % 8 == 0 {
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

#[link(name = "mkl_rt", kind = "dylib")]
extern "C" {
    fn cblas_sgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

fn sgd_fused_tn(
    W: *mut f32,
    A: *const f32,
    B_ptr: *const f32,
    m: usize,
    k: usize,
    n: usize,
    lr: f32,
) {
    unsafe {
        cblas_sgemm(
            101, 112, 111, m as i32, n as i32, k as i32, -lr, A, m as i32, B_ptr, n as i32, 1.0f32,
            W, n as i32,
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

// MLP: 512→2048→2048→512→1, B=64
const L: [usize; 5] = [512, 2048, 2048, 512, 1];
const NL: usize = 4;
const B: usize = 64;

struct State {
    ws: [Vec<f32>; NL],
    bs: [Vec<f32>; NL],
    dbs: [Vec<f32>; NL],
    tmps: [Vec<f32>; NL],
    dts: [Vec<f32>; NL],
    acts: [Vec<f32>; 5],
    pres: [Vec<f32>; 3],
    deltas: [Vec<f32>; NL],
    last_loss: f32,
}

impl State {
    fn new() -> Self {
        let mut rng = 42u64;
        let mut ws: [Vec<f32>; NL] = std::array::from_fn(|i| vec![0.0f32; L[i] * L[i + 1]]);
        for i in 0..NL {
            lcg_xavier(&mut ws[i], L[i], L[i + 1], &mut rng);
        }
        State {
            ws,
            bs: std::array::from_fn(|i| vec![0.0f32; L[i + 1]]),
            dbs: std::array::from_fn(|i| vec![0.0f32; L[i + 1]]),
            tmps: std::array::from_fn(|i| vec![0.0f32; L[i] * B]),
            dts: std::array::from_fn(|i| vec![0.0f32; L[i + 1] * B]),
            acts: std::array::from_fn(|i| vec![0.0f32; B * L[i]]),
            pres: std::array::from_fn(|i| vec![0.0f32; B * L[i + 1]]),
            deltas: std::array::from_fn(|i| vec![0.0f32; B * L[i + 1]]),
            last_loss: 0.0,
        }
    }

    fn step(&mut self, x: &[f32], y: &[f32], lr: f32) {
        // FORWARD
        self.acts[0].copy_from_slice(x);
        for i in 0..NL {
            let (r, c) = (L[i], L[i + 1]);
            let is_last = i == NL - 1;
            if is_last {
                sgemm(
                    self.acts[NL].as_mut_ptr(),
                    self.acts[i].as_ptr(),
                    self.ws[i].as_ptr(),
                    B,
                    r,
                    c,
                );
                for bi in 0..B {
                    for j in 0..c {
                        self.acts[NL][bi * c + j] += self.bs[i][j];
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
        let nt = B * L[NL];
        let mut lacc = 0.0f32;
        for k in 0..nt {
            let d = self.acts[NL][k] - y[k];
            lacc += d * d;
            self.deltas[NL - 1][k] = 2.0 * d / (nt as f32);
        }
        self.last_loss = lacc / (nt as f32);
        // BACKWARD
        for i in (0..NL).rev() {
            let (r, c) = (L[i], L[i + 1]);
            let is_last = i == NL - 1;
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
    let args: Vec<String> = std::env::args().collect();
    let out_file = args
        .windows(2)
        .find(|w| w[0] == "--out")
        .map(|w| w[1].clone());

    let x: Vec<f32> = (0..B * L[0])
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let y: Vec<f32> = (0..B * L[NL]).map(|i| (i % 7) as f32 * 0.1).collect();
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
        results[r] = t0.elapsed().as_nanos() as f64 / 1e6 / STEPS as f64;
        checksum += s.last_loss;
    }
    results.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = results[RUNS / 2];
    eprintln!(
        "[TRAIN][VOLTA-MLP2048] median={:.3} ms/step  all7={:?}",
        med,
        results.map(|x| format!("{:.3}", x))
    );
    eprintln!(
        "  checksum={:.6}  B={}  arch=512->2048->2048->512->1",
        checksum, B
    );

    if let Some(path) = out_file {
        let _ = std::fs::write(&path, format!("median={:.3}\n", med));
    }

    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(code: u32);
        }
        ExitProcess(0);
    }
}
