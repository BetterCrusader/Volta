fn main() {
    // MKL from conda mkl env
    let mkl_lib = "C:/Users/User/miniforge3/envs/mkl/Library/lib";
    println!("cargo:rustc-link-search=native={mkl_lib}");
    // mkl_rt is the single dynamic dispatch lib (links to mkl_rt.2.dll at runtime)
    println!("cargo:rustc-link-lib=dylib=mkl_rt");
}
