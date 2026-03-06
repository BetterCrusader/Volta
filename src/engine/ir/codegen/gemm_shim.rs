/// C-callable gemm shim — bridges LLVM-generated code to the gemm crate.
///
/// volta_gemm_f32(C, A, B, m, k, n):  C[m×n] = A[m×k] @ B[k×n]
/// This is compiled into the DLL alongside the generated IR object.
/// Exported with #[no_mangle] so the linker resolves the IR declaration.

#[no_mangle]
pub unsafe extern "C" fn volta_gemm_f32(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    m: i64,
    k: i64,
    n: i64,
) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    // gemm crate: dst = alpha*dst + beta*(lhs @ rhs)
    // alpha=0, beta=1  →  dst = lhs @ rhs
    let parallelism = if 2 * m * k * n >= (1 << 17) {
        gemm::Parallelism::Rayon(0)
    } else {
        gemm::Parallelism::None
    };
    gemm::gemm(
        m, n, k,
        c,
        1,          // dst col stride
        n as isize, // dst row stride
        false,
        a,
        1,          // lhs col stride (row-major)
        k as isize, // lhs row stride
        b,
        1,          // rhs col stride
        n as isize, // rhs row stride
        0.0_f32,    // alpha
        1.0_f32,    // beta
        false, false, false,
        parallelism,
    );
}
