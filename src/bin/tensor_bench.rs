//! Tensor operation benchmarks.
//!
//! Run with:
//! ```sh
//! cargo run --release --bin tensor_bench
//! ```
//!
//! These benchmarks measure the performance of core tensor operations
//! to guide optimization decisions (e.g., TILE size selection).

use volta::ir::tensor::Tensor;

fn timeit(label: &str, iters: u32, mut f: impl FnMut()) {
    let start = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let avg = elapsed / iters;
    println!("{} ×{}: {} ns/iter", label, iters, avg.as_nanos());
}

fn bench_matmul_64x64() {
    let a = Tensor::new(vec![64, 64], vec![1.0_f32; 64 * 64]).unwrap();
    let b = Tensor::new(vec![64, 64], vec![1.0_f32; 64 * 64]).unwrap();
    timeit("matmul 64x64", 1000, || {
        let _ = a.matmul(&b).unwrap();
    });
}

fn bench_matmul_128x128() {
    let a = Tensor::new(vec![128, 128], vec![1.0_f32; 128 * 128]).unwrap();
    let b = Tensor::new(vec![128, 128], vec![1.0_f32; 128 * 128]).unwrap();
    timeit("matmul 128x128", 1000, || {
        let _ = a.matmul(&b).unwrap();
    });
}

fn bench_matmul_256x256() {
    let a = Tensor::new(vec![256, 256], vec![1.0_f32; 256 * 256]).unwrap();
    let b = Tensor::new(vec![256, 256], vec![1.0_f32; 256 * 256]).unwrap();
    timeit("matmul 256x256", 100, || {
        let _ = a.matmul(&b).unwrap();
    });
}

fn bench_softmax_64() {
    let x = Tensor::new(vec![64], vec![1.0_f32; 64]).unwrap();
    timeit("softmax 64", 1000, || {
        let _ = x.softmax().unwrap();
    });
}

fn bench_softmax_1024() {
    let x = Tensor::new(vec![1024], vec![1.0_f32; 1024]).unwrap();
    timeit("softmax 1024", 1000, || {
        let _ = x.softmax().unwrap();
    });
}

fn bench_sigmoid_1024() {
    let x = Tensor::new(vec![1024], vec![1.0_f32; 1024]).unwrap();
    timeit("sigmoid 1024", 1000, || {
        let _ = x.sigmoid().unwrap();
    });
}

fn bench_gelu_1024() {
    let x = Tensor::new(vec![1024], vec![1.0_f32; 1024]).unwrap();
    timeit("gelu 1024", 1000, || {
        let _ = x.gelu().unwrap();
    });
}

fn bench_add_same_shape_1024() {
    let a = Tensor::new(vec![1024], vec![1.0_f32; 1024]).unwrap();
    let b = Tensor::new(vec![1024], vec![1.0_f32; 1024]).unwrap();
    timeit("add same shape 1024", 1000, || {
        let _ = a.add(&b).unwrap();
    });
}

fn bench_reduce_sum_1024() {
    let x = Tensor::new(vec![1024], vec![1.0_f32; 1024]).unwrap();
    timeit("reduce_sum 1024", 1000, || {
        let _ = x.reduce_sum(None);
    });
}

fn main() {
    println!("=== Tensor Operation Benchmarks ===\n");

    println!("--- MatMul ---");
    bench_matmul_64x64();
    bench_matmul_128x128();
    bench_matmul_256x256();

    println!("\n--- Softmax ---");
    bench_softmax_64();
    bench_softmax_1024();

    println!("\n--- Unary Ops ---");
    bench_sigmoid_1024();
    bench_gelu_1024();

    println!("\n--- Binary Ops ---");
    bench_add_same_shape_1024();

    println!("\n--- Reductions ---");
    bench_reduce_sum_1024();

    println!("\nDone.");
}
