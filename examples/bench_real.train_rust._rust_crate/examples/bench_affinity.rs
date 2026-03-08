#![allow(clippy::needless_range_loop)]
// bench_affinity.rs — pin rayon threads to cores, then benchmark pipeline
// Uses Windows SetThreadAffinityMask to reduce scheduler jitter
#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

#[cfg(windows)]
fn pin_current_thread(core: usize) {
    unsafe {
        extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn SetThreadAffinityMask(h: *mut std::ffi::c_void, mask: usize) -> usize;
        }
        SetThreadAffinityMask(GetCurrentThread(), 1usize << core);
    }
}

fn par(m: usize, k: usize, n: usize) -> gemm::Parallelism {
    let ops = 2 * m * k * n;
    if ops < (1 << 20) {
        gemm::Parallelism::None
    } else if ops < (1 << 25) {
        gemm::Parallelism::Rayon(4)
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
            1,
            n as isize,
            false,
            a,
            1,
            k as isize,
            b,
            1,
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
fn sgd_tn(w: *mut f32, a: *const f32, b: *const f32, m: usize, k: usize, n: usize, lr: f32) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            w,
            1,
            n as isize,
            true,
            a,
            m as isize,
            1,
            b,
            1,
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
fn bwd_wt(dp: *mut f32, w: *const f32, d: *const f32, r: usize, c: usize, b: usize) {
    unsafe {
        gemm::gemm(
            b,
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
            par(b, c, r),
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

fn do_fwd(s: &mut Slot, ws: &[Vec<f32>; N], bs: &[Vec<f32>; N], x: &[f32]) {
    s.acts[0].copy_from_slice(x);
    for i in 0..N {
        let (r, c) = (LA[i], LA[i + 1]);
        if i == N - 1 {
            sgemm(
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
            sgemm(
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
                bwd_wt(
                    s.deltas[i - 1].as_mut_ptr(),
                    ws[i].as_ptr(),
                    s.deltas[i].as_ptr(),
                    r,
                    c,
                    B,
                );
            } else {
                let mut dt = vec![0f32; c * B];
                let mut tmp = vec![0f32; r * B];
                const T: usize = 32;
                let mut a = 0;
                while a < B {
                    let am = (a + T).min(B);
                    let mut b2 = 0;
                    while b2 < c {
                        let bm = (b2 + T).min(c);
                        for aa in a..am {
                            for bb in b2..bm {
                                dt[bb * B + aa] = s.deltas[i][aa * c + bb];
                            }
                        }
                        b2 += T;
                    }
                    a += T;
                }
                sgemm(tmp.as_mut_ptr(), ws[i].as_ptr(), dt.as_ptr(), r, c, B);
                let mut a = 0;
                while a < r {
                    let am = (a + T).min(r);
                    let mut b2 = 0;
                    while b2 < B {
                        let bm = (b2 + T).min(B);
                        for aa in a..am {
                            for bb in b2..bm {
                                s.deltas[i - 1][bb * r + aa] = tmp[aa * B + bb];
                            }
                        }
                        b2 += T;
                    }
                    a += T;
                }
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

fn apply_sgd(s: &Slot, ws_out: &mut [Vec<f32>; N], lr: f32) {
    for i in 0..N {
        let (r, c) = (LA[i], LA[i + 1]);
        sgd_tn(
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

fn run_pipeline(x: &[f32], y: &[f32], lr: f32, steps: usize) -> (f64, f32) {
    let mut ws = [make_ws(), make_ws()];
    let (w0, w1) = ws.split_at_mut(1);
    for i in 0..N {
        w1[0][i].copy_from_slice(&w0[0][i]);
    }
    let mut bs = [
        std::array::from_fn::<Vec<f32>, N, _>(|i| vec![0f32; LA[i + 1]]),
        std::array::from_fn::<Vec<f32>, N, _>(|i| vec![0f32; LA[i + 1]]),
    ];
    let mut slots = [Slot::new(), Slot::new()];
    let mut ri = 0usize;

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
            bsw[i].copy_from_slice(&bsr[i]);
        }
        compute_grads(&mut slots[0], wsr, bsw, y, lr);
        apply_sgd(&slots[0], wsw, lr);
        ri = wi;
    }
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
                let ws_arr = &*(ws_r as *const [Vec<f32>; N]);
                let bs_arr = &*(bs_r as *const [Vec<f32>; N]);
                let xsl = std::slice::from_raw_parts(xp as *const f32, B * LA[0]);
                do_fwd(&mut *(s_nxt as *mut Slot), ws_arr, bs_arr, xsl);
            },
            || unsafe {
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

fn main() {
    // Pin main thread to core 0
    #[cfg(windows)]
    pin_current_thread(0);

    // Build rayon pool with thread pinning
    rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .start_handler(|i| {
            #[cfg(windows)]
            pin_current_thread(i + 1); // cores 1..12
        })
        .build_global()
        .ok();

    let x: Vec<f32> = (0..B * LA[0])
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let y: Vec<f32> = (0..B * LA[N]).map(|i| (i % 7) as f32 * 0.1).collect();
    let lr = 0.01f32;
    const RUNS: usize = 15;
    const STEPS: usize = 50;

    let mut rv = [0f64; RUNS];
    let mut rp = [0f64; RUNS];
    for r in 0..RUNS {
        let (t, _) = run_sequential(&x, &y, lr, STEPS);
        rv[r] = t;
    }
    for r in 0..RUNS {
        let (t, _) = run_pipeline(&x, &y, lr, STEPS);
        rp[r] = t;
    }
    common::sort_f64_samples(&mut rv);
    common::sort_f64_samples(&mut rp);
    let pytorch = 2.440f64;
    println!(
        "[sequential] p25={:.3} p50={:.3}  vs PyTorch {:+.1}%",
        rv[3],
        rv[7],
        (pytorch - rv[7]) / pytorch * 100.0
    );
    println!(
        "[pipeline  ] p25={:.3} p50={:.3}  vs PyTorch {:+.1}%",
        rp[3],
        rp[7],
        (pytorch - rp[7]) / pytorch * 100.0
    );
    println!("pipeline all: {:?}", rp.map(|v| format!("{:.3}", v)));

    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(c: u32);
        }
        ExitProcess(0);
    }
}
