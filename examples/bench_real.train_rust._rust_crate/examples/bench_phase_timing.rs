// bench_phase_timing.rs — measure individual phase costs to find bottleneck
#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

fn par(m: usize, k: usize, n: usize) -> gemm::Parallelism {
    let ops = 2 * m * k * n;
    if ops < (1 << 20) {
        gemm::Parallelism::None
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

fn bench_us(label: &str, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    let mut res = [0f64; 7];
    for slot in &mut res {
        let t0 = Instant::now();
        for _ in 0..100 {
            f();
        }
        *slot = t0.elapsed().as_nanos() as f64 / 100.0 / 1000.0;
    }
    common::sort_f64_samples(&mut res);
    println!("  [{:<30}] p50={:.1} us", label, res[3]);
    res[3]
}

fn main() {
    let x: Vec<f32> = (0..B * LA[0])
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let y: Vec<f32> = (0..B * LA[N]).map(|i| (i % 7) as f32 * 0.1).collect();
    let lr = 0.01f32;
    let ws = make_ws();
    let bs: [Vec<f32>; N] = std::array::from_fn(|i| vec![0f32; LA[i + 1]]);
    let mut ws2 = make_ws();
    let mut bs2: [Vec<f32>; N] = std::array::from_fn(|i| vec![0f32; LA[i + 1]]);
    let mut s = Slot::new();
    // warm up
    do_fwd(&mut s, &ws, &bs, &x);

    println!("=== Phase costs (single-threaded, Rayon(0)) ===");

    let t_fwd = bench_us("fwd pass", || {
        do_fwd(&mut s, &ws, &bs, &x);
    });

    let t_grads = bench_us("compute_grads (bwd)", || {
        compute_grads(&mut s, &ws, &mut bs2, &y, lr);
    });

    let t_sgd = bench_us("apply_sgd", || {
        apply_sgd(&s, &mut ws2, lr);
    });

    // ws copy cost
    let t_copy = bench_us("copy_ws (~8.5MB)", || {
        for i in 0..N {
            ws2[i].copy_from_slice(&ws[i]);
        }
    });

    // measure fwd+grads parallel (simulates Thread A + Thread B2)
    println!("\n=== Parallel phase costs ===");
    // fwd || compute_grads (theoretical min if perfectly parallel)
    let t_fwd_grads_seq = bench_us("fwd + grads (sequential)", || {
        do_fwd(&mut s, &ws, &bs, &x);
        compute_grads(&mut s, &ws, &mut bs2, &y, lr);
    });

    let t_fwd_grads_par = bench_us("fwd || grads (rayon::join)", || {
        let sp = &mut s as *mut Slot as usize;
        let wsp = &ws as *const [Vec<f32>; N] as usize;
        let bsp = &bs as *const [Vec<f32>; N] as usize;
        let bs2p = &mut bs2 as *mut [Vec<f32>; N] as usize;
        let xp = x.as_ptr() as usize;
        let yp = y.as_ptr() as usize;
        rayon::join(
            || unsafe {
                do_fwd(
                    &mut *(sp as *mut Slot),
                    &*(wsp as *const [Vec<f32>; N]),
                    &*(bsp as *const [Vec<f32>; N]),
                    std::slice::from_raw_parts(xp as *const f32, B * LA[0]),
                );
            },
            || unsafe {
                compute_grads(
                    &mut *(sp as *mut Slot),
                    &*(wsp as *const [Vec<f32>; N]),
                    &mut *(bs2p as *mut [Vec<f32>; N]),
                    std::slice::from_raw_parts(yp as *const f32, B * LA[N]),
                    lr,
                );
            },
        );
    });

    // 3-way: fwd || (copy_ws || grads)
    let mut s2 = Slot::new();
    do_fwd(&mut s2, &ws, &bs, &x);
    let t_3way = bench_us("fwd || (copy_ws || grads)", || {
        let sp = &mut s as *mut Slot as usize;
        let sp2 = &mut s2 as *mut Slot as usize;
        let wsp = &ws as *const [Vec<f32>; N] as usize;
        let ws2p = &mut ws2 as *mut [Vec<f32>; N] as usize;
        let bsp = &bs as *const [Vec<f32>; N] as usize;
        let bs2p = &mut bs2 as *mut [Vec<f32>; N] as usize;
        let xp = x.as_ptr() as usize;
        let yp = y.as_ptr() as usize;
        rayon::join(
            || unsafe {
                do_fwd(
                    &mut *(sp2 as *mut Slot),
                    &*(wsp as *const [Vec<f32>; N]),
                    &*(bsp as *const [Vec<f32>; N]),
                    std::slice::from_raw_parts(xp as *const f32, B * LA[0]),
                );
            },
            || unsafe {
                rayon::join(
                    || {
                        let ws_src = &*(wsp as *const [Vec<f32>; N]);
                        let ws_dst = &mut *(ws2p as *mut [Vec<f32>; N]);
                        for i in 0..N {
                            (*ws_dst)[i].copy_from_slice(&(*ws_src)[i]);
                        }
                    },
                    || {
                        compute_grads(
                            &mut *(sp as *mut Slot),
                            &*(wsp as *const [Vec<f32>; N]),
                            &mut *(bs2p as *mut [Vec<f32>; N]),
                            std::slice::from_raw_parts(yp as *const f32, B * LA[N]),
                            lr,
                        );
                    },
                );
            },
        );
    });

    let _ = (
        t_fwd,
        t_grads,
        t_sgd,
        t_copy,
        t_fwd_grads_seq,
        t_fwd_grads_par,
        t_3way,
    );

    println!("\n=== Summary ===");
    println!("  fwd:         {:.1} us", t_fwd);
    println!("  grads(bwd):  {:.1} us", t_grads);
    println!("  sgd:         {:.1} us", t_sgd);
    println!("  copy_ws:     {:.1} us", t_copy);
    println!("  fwd+grads seq: {:.1} us", t_fwd_grads_seq);
    println!(
        "  fwd||grads:    {:.1} us  (efficiency {:.0}%)",
        t_fwd_grads_par,
        (t_fwd + t_grads) / 2.0 / t_fwd_grads_par * 100.0
    );
    println!(
        "  fwd||(copy||grads): {:.1} us  vs pipeline result ~2110 us",
        t_3way
    );
    println!(
        "  theoretical min (fwd||grads+sgd): {:.1} us",
        t_fwd.max(t_grads + t_sgd)
    );

    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(c: u32);
        }
        ExitProcess(0);
    }
}
