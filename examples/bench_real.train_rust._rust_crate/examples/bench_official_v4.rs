// bench_official_v4.rs — faer forward+SGD, gemm-crate stride bwd for r>c layers
// Run: cargo run --release --example bench_official_v4
#![allow(non_snake_case, dead_code)]
use faer::linalg::matmul::matmul;
use faer::Accum;
use std::time::Instant;

// ── helpers ───────────────────────────────────────────────────────────────────

use faer::mat::{MatMut, MatRef};

/// row-major slice → faer MatRef (zero-copy)
#[inline(always)]
unsafe fn as_mat<'a>(ptr: *const f32, rows: usize, cols: usize) -> MatRef<'a, f32> {
    MatRef::from_raw_parts(ptr, rows, cols, cols as isize, 1isize)
}
#[inline(always)]
unsafe fn as_mat_mut<'a>(ptr: *mut f32, rows: usize, cols: usize) -> MatMut<'a, f32> {
    MatMut::from_raw_parts_mut(ptr, rows, cols, cols as isize, 1isize)
}

fn fpar() -> faer::Par {
    faer::get_global_parallelism()
}

/// C[m×n] = A[m×k] @ B[k×n]   (replace)
#[inline(always)]
fn fgemm(c: *mut f32, a: *const f32, b: *const f32, m: usize, k: usize, n: usize) {
    unsafe {
        matmul(
            as_mat_mut(c, m, n),
            Accum::Replace,
            as_mat(a, m, k),
            as_mat(b, k, n),
            1.0f32,
            fpar(),
        );
    }
}

/// W[r×c] -= lr * A[B×r]^T @ B[B×c]   (SGD fused)
#[inline(always)]
fn fsgd(w: *mut f32, a: *const f32, b: *const f32, r: usize, batch: usize, c: usize, lr: f32) {
    unsafe {
        matmul(
            as_mat_mut(w, r, c),
            Accum::Add,
            as_mat(a, batch, r).transpose(),
            as_mat(b, batch, c),
            -lr,
            fpar(),
        );
    }
}

/// delta_prev[B×r] = delta[B×c] @ W^T  (r>c, no transpose buffer)
/// Uses gemm crate with non-unit col-stride — faster than faer explicit transpose for these shapes
#[inline(always)]
fn bwd_wt(dp: *mut f32, w: *const f32, d: *const f32, r: usize, c: usize, batch: usize) {
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
    unsafe {
        gemm::gemm(
            batch,
            r,
            c,
            dp,
            1,
            r as isize,
            false,
            d,
            1,
            c as isize,
            w,
            c as isize,
            1,
            0f32,
            1f32,
            false,
            false,
            false,
            par(batch, c, r),
        );
    }
}

/// delta_prev[B×r] = delta[B×c] @ W^T  (r<=c, faer with explicit transpose — faster for square/wide)
#[inline(always)]
fn bwd_wt_faer(dp: *mut f32, w: *const f32, d: *const f32, r: usize, c: usize, batch: usize) {
    unsafe {
        matmul(
            as_mat_mut(dp, batch, r),
            Accum::Replace,
            as_mat(d, batch, c),
            as_mat(w, r, c).transpose(),
            1.0f32,
            fpar(),
        );
    }
}

// ── AVX-512 bias+relu ─────────────────────────────────────────────────────────
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
        if j < cols {
            let rem = (cols - j) as u16;
            let mask = (1u16 << rem) - 1;
            let b = _mm512_maskz_loadu_ps(mask, bias.add(j));
            let p = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, pre.add(base + j)), b);
            _mm512_mask_storeu_ps(pre.add(base + j), mask, p);
            _mm512_mask_storeu_ps(act.add(base + j), mask, _mm512_max_ps(p, zero));
        }
    }
}

// ── AVX-512 relu_mask + db accumulate ────────────────────────────────────────
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
    std::ptr::write_bytes(db, 0, cols * 4);
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
            _mm512_storeu_ps(db.add(j), _mm512_add_ps(_mm512_loadu_ps(db.add(j)), d2));
            j += 16;
        }
        if j < cols {
            let rem = (cols - j) as u16;
            let km = (1u16 << rem) - 1;
            let p = _mm512_maskz_loadu_ps(km, pre.add(base + j));
            let d = _mm512_maskz_loadu_ps(km, delta.add(base + j));
            let cm = _mm512_cmp_ps_mask(p, zero, _CMP_GT_OQ);
            let d2 = _mm512_maskz_mov_ps(cm & km, d);
            _mm512_mask_storeu_ps(delta.add(base + j), km, d2);
            let acc = _mm512_maskz_loadu_ps(km, db.add(j));
            _mm512_mask_storeu_ps(db.add(j), km, _mm512_add_ps(acc, d2));
        }
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ── AVX2 8×8 transpose ───────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_8x8(
    dst: *mut f32,
    drows: usize,
    src: *const f32,
    scols: usize,
    bi: usize,
    bj: usize,
) {
    let r0 = _mm256_loadu_ps(src.add(bi * scols + bj));
    let r1 = _mm256_loadu_ps(src.add((bi + 1) * scols + bj));
    let r2 = _mm256_loadu_ps(src.add((bi + 2) * scols + bj));
    let r3 = _mm256_loadu_ps(src.add((bi + 3) * scols + bj));
    let r4 = _mm256_loadu_ps(src.add((bi + 4) * scols + bj));
    let r5 = _mm256_loadu_ps(src.add((bi + 5) * scols + bj));
    let r6 = _mm256_loadu_ps(src.add((bi + 6) * scols + bj));
    let r7 = _mm256_loadu_ps(src.add((bi + 7) * scols + bj));
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
    _mm256_storeu_ps(dst.add(bj * drows + bi), o0);
    _mm256_storeu_ps(dst.add((bj + 1) * drows + bi), o1);
    _mm256_storeu_ps(dst.add((bj + 2) * drows + bi), o2);
    _mm256_storeu_ps(dst.add((bj + 3) * drows + bi), o3);
    _mm256_storeu_ps(dst.add((bj + 4) * drows + bi), o4);
    _mm256_storeu_ps(dst.add((bj + 5) * drows + bi), o5);
    _mm256_storeu_ps(dst.add((bj + 6) * drows + bi), o6);
    _mm256_storeu_ps(dst.add((bj + 7) * drows + bi), o7);
}

fn fast_transpose(dst: *mut f32, src: *const f32, rows: usize, cols: usize) {
    #[cfg(target_arch = "x86_64")]
    if rows.is_multiple_of(8) && cols.is_multiple_of(8) {
        let mut i = 0;
        while i < rows {
            let mut j = 0;
            while j < cols {
                unsafe {
                    transpose_8x8(dst, rows, src, cols, i, j);
                }
                j += 8;
            }
            i += 8;
        }
        return;
    }
    const T: usize = 32;
    let mut i = 0;
    while i < rows {
        let im = (i + T).min(rows);
        let mut j = 0;
        while j < cols {
            let jm = (j + T).min(cols);
            unsafe {
                for ii in i..im {
                    for jj in j..jm {
                        *dst.add(jj * rows + ii) = *src.add(ii * cols + jj);
                    }
                }
            }
            j += T;
        }
        i += T;
    }
}

// ── data ─────────────────────────────────────────────────────────────────────
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
    dts: [Vec<f32>; N],
    tmps: [Vec<f32>; N],
    acts: [Vec<f32>; 6],
    pres: [Vec<f32>; 4],
    deltas: [Vec<f32>; N],
    last_loss: f32,
}
impl State {
    fn new() -> Self {
        let mut rng = 42u64;
        let mut ws: [Vec<f32>; N] = std::array::from_fn(|i| vec![0f32; LAYERS[i] * LAYERS[i + 1]]);
        lcg_xavier(&mut ws[0], 512, 1024, &mut rng);
        lcg_xavier(&mut ws[1], 1024, 1024, &mut rng);
        lcg_xavier(&mut ws[2], 1024, 512, &mut rng);
        lcg_xavier(&mut ws[3], 512, 256, &mut rng);
        lcg_xavier(&mut ws[4], 256, 1, &mut rng);
        State {
            ws,
            bs: std::array::from_fn(|i| vec![0f32; LAYERS[i + 1]]),
            dbs: std::array::from_fn(|i| vec![0f32; LAYERS[i + 1]]),
            dts: std::array::from_fn(|i| vec![0f32; LAYERS[i + 1] * B]),
            tmps: std::array::from_fn(|i| vec![0f32; LAYERS[i] * B]),
            acts: std::array::from_fn(|i| vec![0f32; B * LAYERS[i]]),
            pres: std::array::from_fn(|i| vec![0f32; B * LAYERS[i + 1]]),
            deltas: std::array::from_fn(|i| vec![0f32; B * LAYERS[i + 1]]),
            last_loss: 0.0,
        }
    }

    fn step(&mut self, x: &[f32], y: &[f32], lr: f32) {
        // ── FORWARD (faer) ────────────────────────────────────────────────
        self.acts[0].copy_from_slice(x);
        for i in 0..N {
            let (r, c) = (LAYERS[i], LAYERS[i + 1]);
            let is_last = i == N - 1;
            if is_last {
                fgemm(
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
                fgemm(
                    self.pres[i].as_mut_ptr(),
                    self.acts[i].as_ptr(),
                    self.ws[i].as_ptr(),
                    B,
                    r,
                    c,
                );
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

        // ── LOSS ──────────────────────────────────────────────────────────
        let nt = B * LAYERS[N];
        let mut lacc = 0f32;
        for k in 0..nt {
            let d = self.acts[N][k] - y[k];
            lacc += d * d;
            self.deltas[N - 1][k] = 2.0 * d / (nt as f32);
        }
        self.last_loss = lacc / (nt as f32);

        // ── BACKWARD ─────────────────────────────────────────────────────
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
                            let m = if self.pres[i][bi * c + j] > 0.0 {
                                1f32
                            } else {
                                0f32
                            };
                            let d = self.deltas[i][bi * c + j] * m;
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
                if r > c {
                    // shrinking layer: gemm stride trick is fastest (no transpose buffer needed)
                    bwd_wt(
                        self.deltas[i - 1].as_mut_ptr(),
                        self.ws[i].as_ptr(),
                        self.deltas[i].as_ptr(),
                        r,
                        c,
                        B,
                    );
                } else {
                    // square/expanding: faer with explicit W^T is fastest
                    bwd_wt_faer(
                        self.deltas[i - 1].as_mut_ptr(),
                        self.ws[i].as_ptr(),
                        self.deltas[i].as_ptr(),
                        r,
                        c,
                        B,
                    );
                }
            }

            // SGD (faer)
            fsgd(
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
    let mut results = [0f64; RUNS];
    let mut checksum = 0f32;
    for r in 0..RUNS {
        let t0 = Instant::now();
        for _ in 0..STEPS {
            s.step(&x, &y, lr);
        }
        results[r] = t0.elapsed().as_nanos() as f64 / 1000.0 / STEPS as f64 / 1000.0;
        checksum += s.last_loss;
    }
    results.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "[TRAIN][VOLTA-RUST-V4] median={:.3} ms/step  all7={:?}",
        results[RUNS / 2],
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
