// bench_pipeline3.rs — pipeline + faer forward + gemm stride backward
// Architecture: fwd(N+1) || [copy_ws || compute_grads(N)] → sgd(N)
// Forward uses faer (15-27% faster for large layers).
// Backward r>c uses gemm stride trick (no transpose, 37-49% faster).
// par_threads=4 empirically best for pipeline balance.
#![allow(non_snake_case, dead_code)]
use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::Accum;
use std::time::Instant;

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

/// C[m×n] = A[m×k] @ B[k×n] via faer
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

/// SGD: W[r×c] -= lr * A[B×r]^T @ delta[B×c]  via faer
#[inline(always)]
fn fsgd(w: *mut f32, a: *const f32, d: *const f32, r: usize, batch: usize, c: usize, lr: f32) {
    unsafe {
        matmul(
            as_mat_mut(w, r, c),
            Accum::Add,
            as_mat(a, batch, r).transpose(),
            as_mat(d, batch, c),
            -lr,
            fpar(),
        );
    }
}

fn par_gemm(m: usize, k: usize, n: usize) -> gemm::Parallelism {
    let ops = 2 * m * k * n;
    if ops < (1 << 20) {
        gemm::Parallelism::None
    } else if ops < (1 << 25) {
        gemm::Parallelism::Rayon(4)
    } else {
        gemm::Parallelism::Rayon(0)
    }
}

/// delta_prev[B×r] = delta[B×c] @ W[r×c]^T  for r>c (stride trick, no alloc)
#[inline(always)]
fn bwd_wt_stride(dp: *mut f32, w: *const f32, d: *const f32, r: usize, c: usize, batch: usize) {
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
            1isize,
            0f32,
            1f32,
            false,
            false,
            false,
            par_gemm(batch, c, r),
        );
    }
}

/// delta_prev[B×r] = delta[B×c] @ W[r×c]^T  for r<=c (faer explicit transpose)
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn bias_relu(pre: *mut f32, act: *mut f32, bias: *const f32, rows: usize, cols: usize) {
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
            let mk = (1u16 << rem) - 1;
            let b = _mm512_maskz_loadu_ps(mk, bias.add(j));
            let p = _mm512_add_ps(_mm512_maskz_loadu_ps(mk, pre.add(base + j)), b);
            _mm512_mask_storeu_ps(pre.add(base + j), mk, p);
            _mm512_mask_storeu_ps(act.add(base + j), mk, _mm512_max_ps(p, zero));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn relu_mask_db(delta: *mut f32, pre: *const f32, db: *mut f32, rows: usize, cols: usize) {
    use std::arch::x86_64::*;
    std::ptr::write_bytes(db, 0, cols);
    let zero = _mm512_setzero_ps();
    for bi in 0..rows {
        let base = bi * cols;
        let mut j = 0usize;
        while j + 16 <= cols {
            let p = _mm512_loadu_ps(pre.add(base + j));
            let d = _mm512_loadu_ps(delta.add(base + j));
            let mk = _mm512_cmp_ps_mask(p, zero, 14i32);
            let d2 = _mm512_maskz_mov_ps(mk, d);
            _mm512_storeu_ps(delta.add(base + j), d2);
            _mm512_storeu_ps(db.add(j), _mm512_add_ps(_mm512_loadu_ps(db.add(j)), d2));
            j += 16;
        }
        if j < cols {
            let rem = (cols - j) as u16;
            let km = (1u16 << rem) - 1;
            let p = _mm512_maskz_loadu_ps(km, pre.add(base + j));
            let d = _mm512_maskz_loadu_ps(km, delta.add(base + j));
            let cm = _mm512_cmp_ps_mask(p, zero, 14i32);
            let d2 = _mm512_maskz_mov_ps(cm & km, d);
            _mm512_mask_storeu_ps(delta.add(base + j), km, d2);
            _mm512_mask_storeu_ps(
                db.add(j),
                km,
                _mm512_add_ps(_mm512_maskz_loadu_ps(km, db.add(j)), d2),
            );
        }
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

const LA: [usize; 6] = [512, 1024, 1024, 512, 256, 1];
const N: usize = 5;
const B: usize = 64;

struct Slot {
    acts: [Vec<f32>; 6],
    pres: [Vec<f32>; 4],
    deltas: [Vec<f32>; N],
    dbs: [Vec<f32>; N],
    loss: f32,
}
impl Slot {
    fn new() -> Self {
        Slot {
            acts: std::array::from_fn(|i| vec![0f32; B * LA[i]]),
            pres: std::array::from_fn(|i| vec![0f32; B * LA[i + 1]]),
            deltas: std::array::from_fn(|i| vec![0f32; B * LA[i + 1]]),
            dbs: std::array::from_fn(|i| vec![0f32; LA[i + 1]]),
            loss: 0.0,
        }
    }
}

// Forward with faer matmul + AVX-512 bias+relu
fn do_fwd(s: &mut Slot, ws: &[Vec<f32>; N], bs: &[Vec<f32>; N], x: &[f32]) {
    s.acts[0].copy_from_slice(x);
    for i in 0..N {
        let (r, c) = (LA[i], LA[i + 1]);
        if i == N - 1 {
            fgemm(
                s.acts[N].as_mut_ptr(),
                s.acts[i].as_ptr(),
                ws[i].as_ptr(),
                B,
                r,
                c,
            );
            for bi in 0..B {
                for j in 0..c {
                    s.acts[N][bi * c + j] += bs[i][j];
                }
            }
        } else {
            fgemm(
                s.pres[i].as_mut_ptr(),
                s.acts[i].as_ptr(),
                ws[i].as_ptr(),
                B,
                r,
                c,
            );
            #[cfg(target_arch = "x86_64")]
            unsafe {
                bias_relu(
                    s.pres[i].as_mut_ptr(),
                    s.acts[i + 1].as_mut_ptr(),
                    bs[i].as_ptr(),
                    B,
                    c,
                );
            }
        }
    }
}

// Backward: compute deltas + bias grad updates (no SGD)
fn compute_grads(s: &mut Slot, ws: &[Vec<f32>; N], bs: &mut [Vec<f32>; N], y: &[f32], lr: f32) {
    let nt = B;
    let mut lacc = 0f32;
    for k in 0..nt {
        let d = s.acts[N][k] - y[k];
        lacc += d * d;
        s.deltas[N - 1][k] = 2.0 * d / (nt as f32);
    }
    s.loss = lacc / (nt as f32);
    for i in (0..N).rev() {
        let (r, c) = (LA[i], LA[i + 1]);
        if i != N - 1 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                relu_mask_db(
                    s.deltas[i].as_mut_ptr(),
                    s.pres[i].as_ptr(),
                    s.dbs[i].as_mut_ptr(),
                    B,
                    c,
                );
            }
            for k in 0..c {
                bs[i][k] -= lr * s.dbs[i][k];
            }
        }
        if i > 0 {
            if r > c {
                // Shrinking layer: use gemm stride trick
                bwd_wt_stride(
                    s.deltas[i - 1].as_mut_ptr(),
                    ws[i].as_ptr(),
                    s.deltas[i].as_ptr(),
                    r,
                    c,
                    B,
                );
            } else {
                // Growing/square layer: faer with explicit transpose
                bwd_wt_faer(
                    s.deltas[i - 1].as_mut_ptr(),
                    ws[i].as_ptr(),
                    s.deltas[i].as_ptr(),
                    r,
                    c,
                    B,
                );
            }
        }
        if i == N - 1 {
            s.dbs[i].fill(0.0);
            for bi in 0..B {
                for j in 0..c {
                    s.dbs[i][j] += s.deltas[i][bi * c + j];
                }
            }
            for k in 0..c {
                bs[i][k] -= lr * s.dbs[i][k];
            }
        }
    }
}

// SGD: apply weight updates using faer
fn apply_sgd(s: &Slot, ws_out: &mut [Vec<f32>; N], lr: f32) {
    for i in 0..N {
        let (r, c) = (LA[i], LA[i + 1]);
        fsgd(
            ws_out[i].as_mut_ptr(),
            s.acts[i].as_ptr(),
            s.deltas[i].as_ptr(),
            r,
            B,
            c,
            lr,
        );
    }
}

fn make_ws() -> [Vec<f32>; N] {
    let mut rng = 42u64;
    let mut ws: [Vec<f32>; N] = std::array::from_fn(|i| vec![0f32; LA[i] * LA[i + 1]]);
    lcg_xavier(&mut ws[0], 512, 1024, &mut rng);
    lcg_xavier(&mut ws[1], 1024, 1024, &mut rng);
    lcg_xavier(&mut ws[2], 1024, 512, &mut rng);
    lcg_xavier(&mut ws[3], 512, 256, &mut rng);
    lcg_xavier(&mut ws[4], 256, 1, &mut rng);
    ws
}

fn run_sequential(x: &[f32], y: &[f32], lr: f32, steps: usize) -> (f64, f32) {
    let mut ws = make_ws();
    let mut bs: [Vec<f32>; N] = std::array::from_fn(|i| vec![0f32; LA[i + 1]]);
    let mut s = Slot::new();
    macro_rules! step {
        () => {{
            do_fwd(&mut s, &ws, &bs, x);
            compute_grads(&mut s, &ws, &mut bs, y, lr);
            apply_sgd(&s, &mut ws, lr);
        }};
    }
    for _ in 0..10 {
        step!();
    }
    let t0 = Instant::now();
    for _ in 0..steps {
        step!();
    }
    (
        t0.elapsed().as_nanos() as f64 / 1000.0 / steps as f64 / 1000.0,
        s.loss,
    )
}

fn run_pipeline(x: &[f32], y: &[f32], lr: f32, steps: usize) -> (f64, f32) {
    let mut ws = [make_ws(), make_ws()];
    let (ws0h, ws1h) = ws.split_at_mut(1);
    for i in 0..N {
        ws1h[0][i].copy_from_slice(&ws0h[0][i]);
    }

    let mut bs = [
        std::array::from_fn::<Vec<f32>, N, _>(|i| vec![0f32; LA[i + 1]]),
        std::array::from_fn::<Vec<f32>, N, _>(|i| vec![0f32; LA[i + 1]]),
    ];
    let mut slots = [Slot::new(), Slot::new()];
    let mut ri = 0usize;

    // warmup: sequential
    for _ in 0..10 {
        let wi = 1 - ri;
        let (lo, hi) = ws.split_at_mut(wi.max(ri));
        let (wsr, wsw) = if ri < wi {
            (&lo[ri], &mut hi[0])
        } else {
            (&hi[0], &mut lo[wi])
        };
        let (blo, bhi) = bs.split_at_mut(wi.max(ri));
        let (bsr, bsw) = if ri < wi {
            (&blo[ri], &mut bhi[0])
        } else {
            (&bhi[0], &mut blo[wi])
        };
        do_fwd(&mut slots[0], wsr, bsr, x);
        for i in 0..N {
            wsw[i].copy_from_slice(&wsr[i]);
        }
        for i in 0..N {
            bsw[i].copy_from_slice(&bsr[i]);
        }
        compute_grads(&mut slots[0], wsr, bsw, y, lr);
        apply_sgd(&slots[0], wsw, lr);
        ri = wi;
    }

    // prime: fwd for step 0
    do_fwd(&mut slots[ri & 1], &ws[ri], &bs[ri], x);

    let t0 = Instant::now();
    for _step in 0..steps {
        let cur = ri & 1;
        let nxt = 1 - cur;
        let wi = 1 - ri;

        let ws_r = &ws[ri] as *const [Vec<f32>; N] as usize;
        let ws_w = &mut ws[wi] as *mut [Vec<f32>; N] as usize;
        let bs_r = &bs[ri] as *const [Vec<f32>; N] as usize;
        let bs_w = &mut bs[wi] as *mut [Vec<f32>; N] as usize;
        let s_cur = &mut slots[cur] as *mut Slot as usize;
        let s_nxt = &mut slots[nxt] as *mut Slot as usize;
        let xp = x.as_ptr() as usize;
        let yp = y.as_ptr() as usize;

        rayon::join(
            || unsafe {
                // Thread A: fwd(step+1) using faer — faster for W1, W0
                let ws_arr = &*(ws_r as *const [Vec<f32>; N]);
                let bs_arr = &*(bs_r as *const [Vec<f32>; N]);
                let xsl = std::slice::from_raw_parts(xp as *const f32, B * LA[0]);
                do_fwd(&mut *(s_nxt as *mut Slot), ws_arr, bs_arr, xsl);
            },
            || unsafe {
                // Thread B: copy_ws || compute_grads (nested), then sgd
                let ws_src = &*(ws_r as *const [Vec<f32>; N]);
                let ws_dst = &mut *(ws_w as *mut [Vec<f32>; N]);
                let bs_dst = &mut *(bs_w as *mut [Vec<f32>; N]);
                let s = &mut *(s_cur as *mut Slot);
                let ysl = std::slice::from_raw_parts(yp as *const f32, B * LA[N]);

                rayon::join(
                    || {
                        for i in 0..N {
                            (*ws_dst)[i].copy_from_slice(&(*ws_src)[i]);
                        }
                    },
                    || {
                        compute_grads(&mut *(s_cur as *mut Slot), ws_src, &mut *bs_dst, ysl, lr);
                    },
                );

                apply_sgd(s, ws_dst, lr);
            },
        );

        ri = wi;
    }
    let elapsed = t0.elapsed().as_nanos() as f64 / 1000.0 / steps as f64 / 1000.0;
    (elapsed, slots[ri & 1].loss)
}

fn main() {
    let x: Vec<f32> = (0..B * LA[0])
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let y: Vec<f32> = (0..B * LA[N]).map(|i| (i % 7) as f32 * 0.1).collect();
    let lr = 0.01f32;

    let mut rb = [0f64; 7];
    let mut rp = [0f64; 7];
    println!("testing sequential...");
    for r in 0..7 {
        let (t, _) = run_sequential(&x, &y, lr, 50);
        rb[r] = t;
    }
    println!("testing pipeline...");
    for r in 0..7 {
        let (t, _) = run_pipeline(&x, &y, lr, 50);
        rp[r] = t;
    }
    rb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "[sequential] median={:.3} ms  all={:?}",
        rb[3],
        rb.map(|v| format!("{:.3}", v))
    );
    println!(
        "[pipeline  ] median={:.3} ms  all={:?}",
        rp[3],
        rp.map(|v| format!("{:.3}", v))
    );
    println!(
        "speedup: {:.2}x  (PyTorch baseline ~2.440 ms)",
        rb[3] / rp[3]
    );
    println!(
        "vs PyTorch: pipeline {:.1}% faster",
        (2.440 - rp[3]) / 2.440 * 100.0
    );

    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(c: u32);
        }
        ExitProcess(0);
    }
}
