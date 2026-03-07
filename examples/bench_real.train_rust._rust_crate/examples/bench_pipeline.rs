// bench_pipeline.rs — double-buffered pipeline: fwd(step N+1) || bwd+sgd(step N)
// Stale-W by 1 step — mathematically different but converges, checksum drifts slightly
// Run: cargo run --release --example bench_pipeline
#![allow(non_snake_case)]
use std::time::Instant;

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
fn bwd_wt(dp: *mut f32, w: *const f32, d: *const f32, r: usize, c: usize, batch: usize) {
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
            let mask = (1u16 << rem) - 1;
            let b = _mm512_maskz_loadu_ps(mask, bias.add(j));
            let p = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, pre.add(base + j)), b);
            _mm512_mask_storeu_ps(pre.add(base + j), mask, p);
            _mm512_mask_storeu_ps(act.add(base + j), mask, _mm512_max_ps(p, zero));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn relu_mask_db(delta: *mut f32, pre: *const f32, db: *mut f32, rows: usize, cols: usize) {
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

const LAYERS: [usize; 6] = [512, 1024, 1024, 512, 256, 1];
const N: usize = 5;
const B: usize = 64;

// All buffers for one "slot" (fwd activations + bwd deltas)
struct Slot {
    acts: [Vec<f32>; 6],
    pres: [Vec<f32>; 4],
    deltas: [Vec<f32>; N],
    dbs: [Vec<f32>; N],
    last_loss: f32,
}
impl Slot {
    fn new() -> Self {
        Slot {
            acts: std::array::from_fn(|i| vec![0f32; B * LAYERS[i]]),
            pres: std::array::from_fn(|i| vec![0f32; B * LAYERS[i + 1]]),
            deltas: std::array::from_fn(|i| vec![0f32; B * LAYERS[i + 1]]),
            dbs: std::array::from_fn(|i| vec![0f32; LAYERS[i + 1]]),
            last_loss: 0.0,
        }
    }
}

// Shared W double buffer + biases
struct Weights {
    ws: [[Vec<f32>; N]; 2], // ws[0]=read, ws[1]=write (swapped each step)
    bs: [Vec<f32>; N],
    read: usize, // index of current read buffer
}
impl Weights {
    fn new() -> Self {
        let mut rng = 42u64;
        let mut ws0: [Vec<f32>; N] = std::array::from_fn(|i| vec![0f32; LAYERS[i] * LAYERS[i + 1]]);
        lcg_xavier(&mut ws0[0], 512, 1024, &mut rng);
        lcg_xavier(&mut ws0[1], 1024, 1024, &mut rng);
        lcg_xavier(&mut ws0[2], 1024, 512, &mut rng);
        lcg_xavier(&mut ws0[3], 512, 256, &mut rng);
        lcg_xavier(&mut ws0[4], 256, 1, &mut rng);
        let ws1 = ws0.clone(); // write buffer starts as copy of read
        Weights {
            ws: [ws0, ws1],
            bs: std::array::from_fn(|i| vec![0f32; LAYERS[i + 1]]),
            read: 0,
        }
    }
    fn write(&self) -> usize {
        1 - self.read
    }
}

fn do_fwd(slot: &mut Slot, ws: &[Vec<f32>; N], bs: &[Vec<f32>; N], x: &[f32]) {
    slot.acts[0].copy_from_slice(x);
    for i in 0..N {
        let (r, c) = (LAYERS[i], LAYERS[i + 1]);
        if i == N - 1 {
            sgemm(
                slot.acts[N].as_mut_ptr(),
                slot.acts[i].as_ptr(),
                ws[i].as_ptr(),
                B,
                r,
                c,
            );
            for bi in 0..B {
                for j in 0..c {
                    slot.acts[N][bi * c + j] += bs[i][j];
                }
            }
        } else {
            sgemm(
                slot.pres[i].as_mut_ptr(),
                slot.acts[i].as_ptr(),
                ws[i].as_ptr(),
                B,
                r,
                c,
            );
            #[cfg(target_arch = "x86_64")]
            unsafe {
                bias_relu(
                    slot.pres[i].as_mut_ptr(),
                    slot.acts[i + 1].as_mut_ptr(),
                    bs[i].as_ptr(),
                    B,
                    c,
                );
            }
        }
    }
}

fn do_bwd_sgd(
    slot: &mut Slot,
    ws_write: &mut [Vec<f32>; N],
    bs: &mut [Vec<f32>; N],
    ws_read: &[Vec<f32>; N],
    y: &[f32],
    lr: f32,
) {
    // copy read→write before updating
    for i in 0..N {
        ws_write[i].copy_from_slice(&ws_read[i]);
    }

    let nt = B;
    let mut lacc = 0f32;
    for k in 0..nt {
        let d = slot.acts[N][k] - y[k];
        lacc += d * d;
        slot.deltas[N - 1][k] = 2.0 * d / (nt as f32);
    }
    slot.last_loss = lacc / (nt as f32);

    for i in (0..N).rev() {
        let (r, c) = (LAYERS[i], LAYERS[i + 1]);
        let is_last = i == N - 1;
        if !is_last {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                relu_mask_db(
                    slot.deltas[i].as_mut_ptr(),
                    slot.pres[i].as_ptr(),
                    slot.dbs[i].as_mut_ptr(),
                    B,
                    c,
                );
            }
            for k in 0..c {
                bs[i][k] -= lr * slot.dbs[i][k];
            }
        }
        if i > 0 {
            if r > c {
                bwd_wt(
                    slot.deltas[i - 1].as_mut_ptr(),
                    ws_write[i].as_ptr(),
                    slot.deltas[i].as_ptr(),
                    r,
                    c,
                    B,
                );
            } else {
                // transpose + sgemm + transpose for r<=c
                let mut dt = vec![0f32; c * B];
                let mut tmp = vec![0f32; r * B];
                const T: usize = 32;
                let mut ii = 0;
                while ii < B {
                    let im = (ii + T).min(B);
                    let mut jj = 0;
                    while jj < c {
                        let jm = (jj + T).min(c);
                        unsafe {
                            for a in ii..im {
                                for b in jj..jm {
                                    *dt.as_mut_ptr().add(b * B + a) = slot.deltas[i][a * c + b];
                                }
                            }
                        }
                        jj += T;
                    }
                    ii += T;
                }
                sgemm(tmp.as_mut_ptr(), ws_write[i].as_ptr(), dt.as_ptr(), r, c, B);
                let mut ii = 0;
                while ii < r {
                    let im = (ii + T).min(r);
                    let mut jj = 0;
                    while jj < B {
                        let jm = (jj + T).min(B);
                        unsafe {
                            for a in ii..im {
                                for b in jj..jm {
                                    slot.deltas[i - 1][b * r + a] = *tmp.as_ptr().add(a * B + b);
                                }
                            }
                        }
                        jj += T;
                    }
                    ii += T;
                }
            }
        }
        sgd_tn(
            ws_write[i].as_mut_ptr(),
            slot.acts[i].as_ptr(),
            slot.deltas[i].as_ptr(),
            r,
            B,
            c,
            lr,
        );
        if is_last {
            slot.dbs[i].fill(0.0);
            for bi in 0..B {
                for j in 0..c {
                    slot.dbs[i][j] += slot.deltas[i][bi * c + j];
                }
            }
            for k in 0..c {
                bs[i][k] -= lr * slot.dbs[i][k];
            }
        }
    }
}

fn bench_pipeline(x: &[f32], y: &[f32], lr: f32, steps: usize) -> (f64, f32) {
    // Two slots: slot[0] for even steps fwd, slot[1] for odd
    let mut slots = [Slot::new(), Slot::new()];
    let mut wts = Weights::new();

    // Warmup: run 10 steps non-pipelined to stabilize weights
    for _ in 0..10 {
        let r = wts.read;
        let w = wts.write();
        do_fwd(&mut slots[0], &wts.ws[r], &wts.bs, x);
        // need mut access to both ws[r] and ws[w] and bs simultaneously
        // safe: r != w, and we only read ws[r], write ws[w]
        let bs_ptr = wts.bs.as_mut_ptr();
        let ws_r = &wts.ws[r] as *const [Vec<f32>; N];
        let ws_w = &mut wts.ws[w] as *mut [Vec<f32>; N];
        unsafe {
            do_bwd_sgd(
                &mut slots[0],
                &mut *ws_w,
                &mut *std::ptr::addr_of_mut!(wts.bs),
                &*ws_r,
                y,
                lr,
            );
        }
        let _ = bs_ptr;
        wts.read = w;
    }

    let t0 = Instant::now();

    // Pipelined execution using 2 threads
    // Main thread: FWD for step i+1
    // Spawned thread: BWD+SGD for step i
    // Barrier sync between steps

    // We'll simulate pipeline with explicit thread::scope
    // Since Rayon already uses all cores for GEMM, adding another thread
    // may hurt. Instead do interleaved on same thread but measure separately.
    // True pipeline: use 2 threads but limit Rayon to N/2 cores each.

    // Simple approach: just run sequentially but measure if bwd can overlap with next fwd
    // For now: sequential but optimized (same as v3)
    // TODO: true pipeline with rayon::join

    // ACTUAL PIPELINE: rayon::join fwd(step N+1) with bwd+sgd(step N)
    // Both use Rayon internally — limit each to 3 cores
    // Ryzen 7500F has 6 cores, 2 threads × 3 cores = full utilization

    // For benchmark: pre-compute step 0 fwd, then pipeline steps 1..N
    let r0 = wts.read;
    do_fwd(&mut slots[0], &wts.ws[r0], &wts.bs, x);

    let mut last_loss = 0f32;

    for step in 0..steps {
        let cur_slot = step & 1;
        let nxt_slot = 1 - cur_slot;
        let r = wts.read;
        let w = wts.write();

        // Pipeline: fwd(next) || bwd_sgd(cur)
        // Use raw pointers to satisfy borrow checker across rayon::join
        let ws_r_ptr = wts.ws[r].as_ptr() as usize;
        let ws_w_ptr = wts.ws[w].as_mut_ptr() as usize;
        let bs_ptr = wts.bs.as_mut_ptr() as usize;
        let x_ptr = x.as_ptr() as usize;
        let y_ptr = y.as_ptr() as usize;
        let s_cur = &mut slots[cur_slot] as *mut Slot as usize;
        let s_nxt = &mut slots[nxt_slot] as *mut Slot as usize;

        rayon::join(
            || unsafe {
                // Thread A: FWD for next step using W_read
                let ws_r = &*(ws_r_ptr as *const [Vec<f32>; N]);
                let bs = &*(bs_ptr as *const [Vec<f32>; N]);
                let x2 = std::slice::from_raw_parts(x_ptr as *const f32, B * LAYERS[0]);
                do_fwd(&mut *(s_nxt as *mut Slot), ws_r, bs, x2);
            },
            || unsafe {
                // Thread B: BWD+SGD for current step, writes W_write
                let ws_r = &*(ws_r_ptr as *const [Vec<f32>; N]);
                let ws_w = &mut *(ws_w_ptr as *mut [Vec<f32>; N]);
                let bs = &mut *(bs_ptr as *mut [Vec<f32>; N]);
                let y2 = std::slice::from_raw_parts(y_ptr as *const f32, B * LAYERS[N]);
                do_bwd_sgd(&mut *(s_cur as *mut Slot), ws_w, bs, ws_r, y2, lr);
            },
        );

        last_loss = slots[cur_slot].last_loss;
        wts.read = w; // swap buffers
    }

    let elapsed = t0.elapsed().as_nanos() as f64 / 1000.0 / steps as f64 / 1000.0;
    (elapsed, last_loss)
}

fn bench_baseline(x: &[f32], y: &[f32], lr: f32, steps: usize) -> (f64, f32) {
    let mut wts = Weights::new();
    let mut slot = Slot::new();
    for _ in 0..10 {
        let r = wts.read;
        let w = wts.write();
        do_fwd(&mut slot, &wts.ws[r], &wts.bs, x);
        let ws_r = &wts.ws[r] as *const [Vec<f32>; N];
        let ws_w = &mut wts.ws[w] as *mut [Vec<f32>; N];
        unsafe {
            do_bwd_sgd(&mut slot, &mut *ws_w, &mut wts.bs, &*ws_r, y, lr);
        }
        wts.read = w;
    }
    let t0 = Instant::now();
    for _ in 0..steps {
        let r = wts.read;
        let w = wts.write();
        do_fwd(&mut slot, &wts.ws[r], &wts.bs, x);
        let ws_r = &wts.ws[r] as *const [Vec<f32>; N];
        let ws_w = &mut wts.ws[w] as *mut [Vec<f32>; N];
        unsafe {
            do_bwd_sgd(&mut slot, &mut *ws_w, &mut wts.bs, &*ws_r, y, lr);
        }
        wts.read = w;
    }
    let elapsed = t0.elapsed().as_nanos() as f64 / 1000.0 / steps as f64 / 1000.0;
    (elapsed, slot.last_loss)
}

fn main() {
    let x: Vec<f32> = (0..B * LAYERS[0])
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let y: Vec<f32> = (0..B * LAYERS[N]).map(|i| (i % 7) as f32 * 0.1).collect();
    let lr = 0.01f32;

    println!("=== Baseline (sequential, double-buffered W) ===");
    let mut results_b = [0f64; 7];
    for r in 0..7 {
        let (t, _) = bench_baseline(&x, &y, lr, 50);
        results_b[r] = t;
    }
    results_b.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "[baseline] median={:.3} ms  all={:?}",
        results_b[3],
        results_b.map(|x| format!("{:.3}", x))
    );

    println!("\n=== Pipeline (fwd || bwd+sgd, stale W by 1 step) ===");
    let mut results_p = [0f64; 7];
    let mut cs = 0f32;
    for r in 0..7 {
        let (t, loss) = bench_pipeline(&x, &y, lr, 50);
        results_p[r] = t;
        cs += loss;
    }
    results_p.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "[pipeline] median={:.3} ms  all={:?}",
        results_p[3],
        results_p.map(|x| format!("{:.3}", x))
    );
    println!(
        "  checksum={:.6} (differs from v3 due to stale W — expected)",
        cs
    );
    println!("\nSpeedup: {:.2}x", results_b[3] / results_p[3]);

    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(code: u32);
        }
        ExitProcess(0);
    }
}
