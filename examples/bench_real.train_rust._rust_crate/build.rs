use std::path::{Path, PathBuf};

const MKL_MISSING_MESSAGE: &str = "MKL feature enabled, but mkl_rt was not found.\n\
     Set MKL_LIB_DIR, MKLROOT, or CONDA_PREFIX so Cargo can locate mkl_rt.lib.";

fn main() {
    if std::env::var_os("CARGO_FEATURE_MKL").is_none() {
        return;
    }

    let mkl_lib_path = resolve_mkl_lib_path().unwrap_or_else(|| panic!("{MKL_MISSING_MESSAGE}"));

    println!("cargo:rustc-link-search=native={}", mkl_lib_path.display());
    println!("cargo:rustc-link-lib=dylib=mkl_rt");
}

fn resolve_mkl_lib_path() -> Option<PathBuf> {
    [
        std::env::var_os("MKL_LIB_DIR").map(PathBuf::from),
        std::env::var_os("MKLROOT")
            .map(PathBuf::from)
            .map(|root| root.join("lib")),
        std::env::var_os("CONDA_PREFIX")
            .map(PathBuf::from)
            .map(|prefix| prefix.join("Library").join("lib")),
    ]
    .into_iter()
    .flatten()
    .find(|path| has_mkl(path))
}

fn has_mkl(path: &Path) -> bool {
    let lib_name = if cfg!(windows) {
        "mkl_rt.lib"
    } else {
        "libmkl_rt.so"
    };
    path.join(lib_name).exists()
}
