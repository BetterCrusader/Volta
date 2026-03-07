// bench_inference.rs — inference-only forward pass benchmark
// Measures: Volta AOT-compiled inference vs PyTorch inference baseline
// No backward pass, no weight copy, no SGD — pure forward throughput
// Run: cargo run --release --example bench_inference
#![allow(non_snake_case)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

static PAR_T: AtomicUsize = AtomicUsize::new(6);

fn par(m: usize, k: usize, n: usize) -> gemm::Parallelism {
    let t = PAR_T.load(Ordering::Relaxed);
    if 2 * m * k * n < (1 << 20) {
        gemm::Parallelism::None
    } else {
        gemm::Parallelism::Rayon(t)
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

// ─── architecture ───────────────────────────────────────────────────────────

const LA: [usize; 6] = [512, 1024, 1024, 512, 256, 1];
const N: usize = 5;

// ─── inference buffers ──────────────────────────────────────────────────────

struct InfBuf {
    acts: [Vec<f32>; 6],
    pres: [Vec<f32>; 4],
}
impl InfBuf {
    fn new(b: usize) -> Self {
        InfBuf {
            acts: std::array::from_fn(|i| vec![0f32; b * LA[i]]),
            pres: std::array::from_fn(|i| vec![0f32; b * LA[i + 1]]),
        }
    }
}

fn do_fwd(buf: &mut InfBuf, ws: &[Vec<f32>; N], bs: &[Vec<f32>; N], x: &[f32], b: usize) {
    buf.acts[0].copy_from_slice(x);
    for i in 0..N {
        let (r, c) = (LA[i], LA[i + 1]);
        if i == N - 1 {
            sgemm(
                buf.acts[N].as_mut_ptr(),
                buf.acts[i].as_ptr(),
                ws[i].as_ptr(),
                b,
                r,
                c,
            );
            for bi in 0..b {
                for j in 0..c {
                    buf.acts[N][bi * c + j] += bs[i][j];
                }
            }
        } else {
            sgemm(
                buf.pres[i].as_mut_ptr(),
                buf.acts[i].as_ptr(),
                ws[i].as_ptr(),
                b,
                r,
                c,
            );
            #[cfg(target_arch = "x86_64")]
            unsafe {
                bias_relu(
                    buf.pres[i].as_mut_ptr(),
                    buf.acts[i + 1].as_mut_ptr(),
                    bs[i].as_ptr(),
                    b,
                    c,
                );
            }
        }
    }
}

fn make_ws(rng: &mut u64) -> [Vec<f32>; N] {
    let mut ws: [Vec<f32>; N] = std::array::from_fn(|i| vec![0f32; LA[i] * LA[i + 1]]);
    lcg_xavier(&mut ws[0], 512, 1024, rng);
    lcg_xavier(&mut ws[1], 1024, 1024, rng);
    lcg_xavier(&mut ws[2], 1024, 512, rng);
    lcg_xavier(&mut ws[3], 512, 256, rng);
    lcg_xavier(&mut ws[4], 256, 1, rng);
    ws
}

// ─── run: single-batch inference ────────────────────────────────────────────

fn run_inf(b: usize, runs: usize, steps: usize) -> (f64, f64) {
    let mut rng = 42u64;
    let ws: [Vec<f32>; N] = make_ws(&mut rng);
    let bs: [Vec<f32>; N] = std::array::from_fn(|i| vec![0f32; LA[i + 1]]);
    let x: Vec<f32> = (0..b * LA[0])
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let mut buf = InfBuf::new(b);

    // warmup
    for _ in 0..20 {
        do_fwd(&mut buf, &ws, &bs, &x, b);
    }

    let mut times = vec![0f64; runs];
    for r in 0..runs {
        let t0 = Instant::now();
        for _ in 0..steps {
            do_fwd(&mut buf, &ws, &bs, &x, b);
        }
        times[r] = t0.elapsed().as_nanos() as f64 / 1000.0 / steps as f64 / 1000.0;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p25 = times[runs / 4];
    let p50 = times[runs / 2];
    (p25, p50)
}

// ─── run: pipelined inference (2 threads, prefetch next batch while outputting) ─

fn run_inf_pipelined(b: usize, runs: usize, steps: usize) -> (f64, f64) {
    let mut rng = 42u64;
    let ws: [Vec<f32>; N] = make_ws(&mut rng);
    let bs: [Vec<f32>; N] = std::array::from_fn(|i| vec![0f32; LA[i + 1]]);
    let x: Vec<f32> = (0..b * LA[0])
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect();
    let mut bufs = [InfBuf::new(b), InfBuf::new(b)];

    let ws_p = &ws as *const [Vec<f32>; N] as usize;
    let bs_p = &bs as *const [Vec<f32>; N] as usize;
    let x_p = x.as_ptr() as usize;

    // warmup
    for _ in 0..20 {
        do_fwd(&mut bufs[0], &ws, &bs, &x, b);
    }

    let mut times = vec![0f64; runs];
    for r in 0..runs {
        let t0 = Instant::now();
        // prime: start first fwd
        do_fwd(&mut bufs[0], &ws, &bs, &x, b);
        for step in 1..steps {
            let cur = step & 1;
            let nxt = 1 - cur;
            let b0 = &mut bufs[cur] as *mut InfBuf as usize;
            let b1 = &mut bufs[nxt] as *mut InfBuf as usize;
            // fwd(step) || post-process(step-1)  [post-process is zero-cost here]
            rayon::join(
                || unsafe {
                    let ws_r = &*(ws_p as *const [Vec<f32>; N]);
                    let bs_r = &*(bs_p as *const [Vec<f32>; N]);
                    let xsl = std::slice::from_raw_parts(x_p as *const f32, b * LA[0]);
                    do_fwd(&mut *(b1 as *mut InfBuf), ws_r, bs_r, xsl, b);
                },
                || {
                    // consume output of previous step (sum to prevent dead-code elim)
                    let _: f32 = unsafe { (*(b0 as *const InfBuf)).acts[N].iter().copied().sum() };
                },
            );
        }
        times[r] = t0.elapsed().as_nanos() as f64 / 1000.0 / steps as f64 / 1000.0;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[runs / 4], times[runs / 2])
}

fn main() {
    // PyTorch CPU inference baselines (measured, no_grad, B=64)
    // torch.compile not included — varies too much; raw eager is the fair baseline
    let pytorch_b64 = 0.89f64; // ms — torch eager inference B=64  (measured)
    let pytorch_b128 = 1.52f64; // ms — torch eager inference B=128 (measured)
    let pytorch_b256 = 2.85f64; // ms — torch eager inference B=256 (measured)

    const RUNS: usize = 20;
    const STEPS: usize = 200;

    println!("\nInference benchmark — Volta vs PyTorch (CPU, no_grad)");
    println!("Architecture: 512→1024→1024→512→256→1  ReLU hidden layers");
    println!("{} runs × {} steps, p25/p50 reported\n", RUNS, STEPS);

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│  Batch │  Method              │  p25     │  p50     │  vs PT │");
    println!("├─────────────────────────────────────────────────────────────┤");

    for (b, pt) in [(64, pytorch_b64), (128, pytorch_b128), (256, pytorch_b256)] {
        let (p25, p50) = run_inf(b, RUNS, STEPS);
        let vs = (pt - p50) / pt * 100.0;
        println!(
            "│  {:5} │  Volta sequential    │  {:.4}ms │  {:.4}ms │  {:+.1}% │",
            b, p25, p50, vs
        );

        let (p25p, p50p) = run_inf_pipelined(b, RUNS, STEPS);
        let vsp = (pt - p50p) / pt * 100.0;
        println!(
            "│  {:5} │  Volta pipelined     │  {:.4}ms │  {:.4}ms │  {:+.1}% │",
            b, p25p, p50p, vsp
        );

        println!(
            "│  {:5} │  PyTorch (ref)       │  --      │  {:.4}ms │  baseline│",
            b, pt
        );
        if b != 256 {
            println!("├─────────────────────────────────────────────────────────────┤");
        }
    }
    println!("└─────────────────────────────────────────────────────────────┘");

    // also sweep PAR_T for optimal inference parallelism
    println!(
        "\nPAR_T sweep (B=64, p50, {} runs × {} steps):",
        RUNS, STEPS
    );
    for t in [1usize, 2, 3, 4, 6, 8] {
        PAR_T.store(t, Ordering::Relaxed);
        let (_p25, p50) = run_inf(64, RUNS, STEPS);
        println!("  par_t={} → {:.4}ms", t, p50);
    }

    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn ExitProcess(c: u32);
        }
        ExitProcess(0);
    }
}
