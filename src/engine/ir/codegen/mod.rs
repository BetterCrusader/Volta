/// LLVM IR codegen for Volta graphs.
///
/// Compiles a forward-inference Graph into a native shared object / executable
/// via inkwell (safe Rust LLVM bindings).  The generated function signature is:
///
///   void volta_infer(const float* input, usize input_len,
///                    const float** params, const usize* param_lens, usize n_params,
///                    float* output, usize output_len)
///
/// Static weight tensors are embedded as LLVM global constants so the
/// compiled binary carries its own parameters — no runtime loading needed.

#[cfg(feature = "llvm-codegen")]
mod inner;

#[cfg(feature = "llvm-codegen")]
pub use inner::{CodegenError, compile_graph_to_object};

#[cfg(feature = "llvm-codegen")]
pub use inner::link_object_to_exe;

// MLP training codegen — generates C source + compiles via clang.
// Does NOT require llvm-codegen feature (clang only, no inkwell).
pub mod mlp_train_codegen;
pub use mlp_train_codegen::{MlpTopology, MlpTrainCodegenError, compile_mlp_train_dll};

// Rust-based MLP training codegen — uses gemm crate (Rayon parallel GEMM).
pub mod mlp_train_rust_codegen;
pub use mlp_train_rust_codegen::{RustTrainCodegenError, compile_mlp_train_rust_dll};
