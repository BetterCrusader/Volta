// bench_final2.rs — layer-pipelined backward+SGD
// For each layer i (N→0): sgd(i) runs PARALLEL with bwd_delta(i-1)
// This hides SGD cost behind backward compute.
#![allow(non_snake_case, dead_code)]
use std::time::Instant;

fn par(m: usize, k: usize, n: usize) -> gemm::Parallelism {
    let ops = 2 * m * k * n;
    if ops < (1 << 20) {
        gemm::Parallelism::None
    } else if ops < (1 << 25) {
        gemm::Parallelism::Rayon(2)
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

// Layer-pipelined: for each layer i, SGD(i) runs parallel with bwd_delta(i-1)
// ws_out must already be a copy of ws (done externally)
fn compute_grads_pipelined_sgd(
    s: &mut Slot,
    ws: &[Vec<f32>; N],
    bs: &mut [Vec<f32>; N],
    ws_out: &mut [Vec<f32>; N],
    y: &[f32],
    lr: f32,
) {
    let nt = B;
    let mut lacc = 0f32;
    for k in 0..nt {
        let d = s.acts[N][k] - y[k];
        lacc += d * d;
        s.deltas[N - 1][k] = 2.0 * d / (nt as f32);
    }
    s.loss = lacc / (nt as f32);

    // Process layer N-1 first (no delta_prev needed, just SGD)
    {
        let i = N - 1;
        let (_r, c) = (LA[i], LA[i + 1]);
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

    // For i = N-1 down to 1: compute delta_{i-1} || SGD(i) in parallel
    for i in (1..N).rev() {
        let (r, c) = (LA[i], LA[i + 1]);

        // Pointers for rayon::join
        let dp_ptr = s.deltas[i - 1].as_mut_ptr() as usize;
        let w_ptr = ws[i].as_ptr() as usize;
        let d_ptr = s.deltas[i].as_ptr() as usize;
        let wout_ptr = ws_out[i].as_mut_ptr() as usize;
        let act_ptr = s.acts[i].as_ptr() as usize;
        let delta_ptr = s.deltas[i].as_ptr() as usize;

        rayon::join(
            || unsafe {
                // compute delta_{i-1}
                let dp = dp_ptr as *mut f32;
                let w = w_ptr as *const f32;
                let d = d_ptr as *const f32;
                if r > c {
                    bwd_wt(dp, w, d, r, c, B);
                } else {
                    let mut dt = vec![0f32; c * B];
                    let mut tmp = vec![0f32; r * B];
                    const T: usize = 32;
                    let dslice = std::slice::from_raw_parts(d, B * c);
                    let mut a = 0;
                    while a < B {
                        let am = (a + T).min(B);
                        let mut b2 = 0;
                        while b2 < c {
                            let bm = (b2 + T).min(c);
                            for aa in a..am {
                                for bb in b2..bm {
                                    dt[bb * B + aa] = dslice[aa * c + bb];
                                }
                            }
                            b2 += T;
                        }
                        a += T;
                    }
                    sgemm(tmp.as_mut_ptr(), w, dt.as_ptr(), r, c, B);
                    let mut a = 0;
                    while a < r {
                        let am = (a + T).min(r);
                        let mut b2 = 0;
                        while b2 < B {
                            let bm = (b2 + T).min(B);
                            let dpslice = std::slice::from_raw_parts_mut(dp, B * r);
                            for aa in a..am {
                                for bb in b2..bm {
                                    dpslice[bb * r + aa] = tmp[aa * B + bb];
                                }
                            }
                            b2 += T;
                        }
                        a += T;
                    }
                }
            },
            || {
                // SGD for layer i (ws_out[i] already copied from ws[i])
                sgd_tn(
                    wout_ptr as *mut f32,
                    act_ptr as *const f32,
                    delta_ptr as *const f32,
                    r,
                    B,
                    c,
                    lr,
                );
            },
        );

        // After getting delta_i-1, apply relu_mask + bias grad for layer i-1 (if not input)
        if i - 1 != N - 1 && i > 1 {
            let ci = LA[i]; // cols of layer i-1 output = rows of layer i
            unsafe {
                relu_mask_db(
                    s.deltas[i - 1].as_mut_ptr(),
                    s.pres[i - 1].as_ptr(),
                    s.dbs[i - 1].as_mut_ptr(),
                    B,
                    ci,
                );
            }
            for k in 0..ci {
                bs[i - 1][k] -= lr * s.dbs[i - 1][k];
            }
        }
    }

    // relu_mask for layer 0 output (delta into layer 0)
    if N > 1 {
        let c0 = LA[1];
        unsafe {
            relu_mask_db(
                s.deltas[0].as_mut_ptr(),
                s.pres[0].as_ptr(),
                s.dbs[0].as_mut_ptr(),
                B,
                c0,
            );
        }
        for k in 0..c0 {
            bs[0][k] -= lr * s.dbs[0][k];
        }
    }

    // SGD for layer 0 (last, no parallelism possible here)
    {
        let (r, c) = (LA[0], LA[1]);
        sgd_tn(
            ws_out[0].as_mut_ptr(),
            s.acts[0].as_ptr(),
            s.deltas[0].as_ptr(),
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

fn run_pipeline_layerpipe(x: &[f32], y: &[f32], lr: f32, steps: usize) -> (f64, f32) {
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

    // warmup
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
        compute_grads_pipelined_sgd(&mut slots[0], wsr, bsw, wsw, y, lr);
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
                // copy_ws || (compute layer deltas + relu/bias) pipelined with SGD
                rayon::join(
                    || {
                        for i in 0..N {
                            (*ws_dst)[i].copy_from_slice(&(*ws_src)[i]);
                        }
                    },
                    || {
                        // Just compute relu_mask for layers N-2 down to 0 (bias grad)
                        // Full grads + pipelined sgd
                        let nt = B;
                        let mut lacc = 0f32;
                        for k in 0..nt {
                            let d = s.acts[N][k] - (*ysl.as_ptr().add(k));
                            lacc += d * d;
                            s.deltas[N - 1][k] = 2.0 * d / (nt as f32);
                        }
                        s.loss = lacc / (nt as f32);
                        // last layer bias
                        let cl = LA[N];
                        s.dbs[N - 1].fill(0.0);
                        for bi in 0..B {
                            for j in 0..cl {
                                s.dbs[N - 1][j] += s.deltas[N - 1][bi * cl + j];
                            }
                        }
                        for k in 0..cl {
                            (*bs_dst)[N - 1][k] -= lr * s.dbs[N - 1][k];
                        }
                        // backward layers
                        for i in (1..N).rev() {
                            let (r, c) = (LA[i], LA[i + 1]);
                            if r > c {
                                bwd_wt(
                                    s.deltas[i - 1].as_mut_ptr(),
                                    ws_src[i].as_ptr(),
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
                                sgemm(tmp.as_mut_ptr(), ws_src[i].as_ptr(), dt.as_ptr(), r, c, B);
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
                            if i > 1 {
                                let ci = LA[i];
                                relu_mask_db(
                                    s.deltas[i - 1].as_mut_ptr(),
                                    s.pres[i - 1].as_ptr(),
                                    s.dbs[i - 1].as_mut_ptr(),
                                    B,
                                    ci,
                                );
                                for k in 0..ci {
                                    (*bs_dst)[i - 1][k] -= lr * s.dbs[i - 1][k];
                                }
                            }
                        }
                        let c0 = LA[1];
                        relu_mask_db(
                            s.deltas[0].as_mut_ptr(),
                            s.pres[0].as_ptr(),
                            s.dbs[0].as_mut_ptr(),
                            B,
                            c0,
                        );
                        for k in 0..c0 {
                            (*bs_dst)[0][k] -= lr * s.dbs[0][k];
                        }
                    },
                );
                // Now ws_dst is copied AND grads computed — apply SGD
                for i in 0..N {
                    let (r, c) = (LA[i], LA[i + 1]);
                    sgd_tn(
                        (*ws_dst)[i].as_mut_ptr(),
                        s.acts[i].as_ptr(),
                        s.deltas[i].as_ptr(),
                        r,
                        B,
                        c,
                        lr,
                    );
                }
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
    const RUNS: usize = 15;
    const STEPS: usize = 50;

    let mut rp = [0f64; RUNS];
    println!(
        "measuring layer-pipelined bwd+sgd pipeline ({} runs)...",
        RUNS
    );
    for r in 0..RUNS {
        let (t, _) = run_pipeline_layerpipe(&x, &y, lr, STEPS);
        rp[r] = t;
    }
    rp.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pytorch = 2.440f64;
    let p25 = rp[3];
    let p50 = rp[7];
    println!(
        "[layer-pipe] p25={:.3} ms  p50={:.3} ms  vs PyTorch: {:+.1}%",
        p25,
        p50,
        (pytorch - p50) / pytorch * 100.0
    );
    println!("all: {:?}", rp.map(|v| format!("{:.3}", v)));

    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(c: u32);
        }
        ExitProcess(0);
    }
}
