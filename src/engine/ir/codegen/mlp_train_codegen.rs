//! MLP training codegen — generates C source for a specific MLP topology,
//! compiles with clang -O3 -march=native.
//!
//! Strategy: ALL matrix multiplications (forward + backward dW + backward dX)
//! use volta_gemm_f32 (the existing tiled GEMM that is already fast).
//!
//! To do this without scalar fallbacks:
//!   - Forward:    GEMM(act{i}, w{i})              — standard
//!   - Backward dW: GEMM(xt{i}, delta{i})           — xt{i} = act{i}^T, pre-computed after forward
//!   - Backward dX: GEMM(delta{i}, wtb{i})          — wtb{i} = w{i}^T, updated after SGD
//!
//! Transpose cost per step: one xt{i} transpose after forward (act{i} known)
//! + one wtb{i} transpose after SGD update. Both use cache-blocked volta_transpose_f32.
//!
//! Zero heap allocs per step; all buffers pre-allocated in init.
//! volta_transpose_f32 is exported from gemm_shim.c.
use std::path::Path;

#[derive(Debug)]
pub struct MlpTrainCodegenError {
    pub message: String,
}

impl std::fmt::Display for MlpTrainCodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MlpTrainCodegenError: {}", self.message)
    }
}

pub struct MlpTopology {
    pub layers: Vec<usize>,
    pub activation: String,
}

pub fn compile_mlp_train_dll(
    topology: &MlpTopology,
    init_weights: Option<&std::collections::HashMap<String, Vec<f32>>>,
    out_dll: &Path,
) -> Result<(), MlpTrainCodegenError> {
    let c_src = out_dll.with_extension("train.c");
    let c_obj = out_dll.with_extension("train.o");
    let shim_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/engine/ir/codegen/gemm_shim.c");
    let shim_obj = out_dll.with_extension("shim.o");

    let code = generate_c_source(topology, init_weights)?;
    std::fs::write(&c_src, &code).map_err(|e| MlpTrainCodegenError {
        message: format!("write C source: {e}"),
    })?;

    let clang = find_clang();

    let s1 = std::process::Command::new(&clang)
        .args(["-O3", "-march=native", "-ffast-math", "-funroll-loops"])
        .arg("-c")
        .arg(&c_src)
        .arg("-o")
        .arg(&c_obj)
        .status()
        .map_err(|e| MlpTrainCodegenError {
            message: format!("clang: {e}"),
        })?;
    if !s1.success() {
        return Err(MlpTrainCodegenError {
            message: format!("compile failed; see {}", c_src.display()),
        });
    }

    let omp_lib = find_omp_lib();

    let mut shim_args = vec!["-O3", "-march=native", "-ffast-math", "-funroll-loops"];
    if omp_lib.is_some() {
        shim_args.push("-fopenmp");
    }
    let s2 = std::process::Command::new(&clang)
        .args(&shim_args)
        .arg("-c")
        .arg(&shim_src)
        .arg("-o")
        .arg(&shim_obj)
        .status()
        .map_err(|e| MlpTrainCodegenError {
            message: format!("shim: {e}"),
        })?;
    if !s2.success() {
        return Err(MlpTrainCodegenError {
            message: "gemm_shim compile failed".into(),
        });
    }

    let mut cmd = std::process::Command::new(&clang);
    cmd.arg(&c_obj)
        .arg(&shim_obj)
        .arg("-shared")
        .arg("-O3")
        .arg("-march=native")
        .arg("-o")
        .arg(out_dll);
    if let Some(ref omp) = omp_lib {
        cmd.arg("-fopenmp");
        // Add libomp.lib directory to linker search path
        if let Some(parent) = omp.parent() {
            cmd.arg(format!("-L{}", parent.display()));
        }
    }
    #[cfg(target_os = "windows")]
    {
        for sym in &[
            "volta_train_init",
            "volta_train_step",
            "volta_train_loss",
            "volta_train_get_params",
            "volta_train_set_params",
            "volta_train_free",
        ] {
            cmd.arg(format!("-Wl,/EXPORT:{sym}"));
        }
    }
    #[cfg(not(target_os = "windows"))]
    cmd.arg("-lm");

    let s3 = cmd.status().map_err(|e| MlpTrainCodegenError {
        message: format!("link: {e}"),
    })?;
    if !s3.success() {
        return Err(MlpTrainCodegenError {
            message: "link failed".into(),
        });
    }
    Ok(())
}

fn find_clang() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("clang.exe")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("clang"))
}

fn find_omp_lib() -> Option<std::path::PathBuf> {
    // Try to find libomp.lib next to the clang executable
    let clang = find_clang();
    let lib_candidate = clang
        .parent()
        .and_then(|d| d.parent())
        .map(|d| d.join("lib").join("libomp.lib"));
    if let Some(p) = lib_candidate {
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn generate_c_source(
    topology: &MlpTopology,
    init_weights: Option<&std::collections::HashMap<String, Vec<f32>>>,
) -> Result<String, MlpTrainCodegenError> {
    let layers = &topology.layers;
    let nl = layers.len();
    if nl < 2 {
        return Err(MlpTrainCodegenError {
            message: "need >= 2 layer sizes".into(),
        });
    }
    let n = nl - 1; // number of weight matrices

    let mut s = String::new();
    s.push_str(
        "#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n#include <stdint.h>\n\n",
    );
    s.push_str(
        "extern void volta_gemm_f32(float*,const float*,const float*,int64_t,int64_t,int64_t);\n",
    );
    s.push_str("extern void volta_transpose_f32(float*,const float*,int64_t,int64_t);\n\n");

    // Struct: store W, W^T (wtb), act^T (xt) — all pre-allocated
    s.push_str("typedef struct VoltaTrainHandle {\n");
    for i in 0..n {
        let (r, c) = (layers[i], layers[i + 1]);
        s.push_str(&format!("  float *w{i}, *b{i}, *dw{i}, *db{i};\n"));
        s.push_str(&format!(
            "  float *wtb{i}; /* w{i}^T [{c}x{r}] — updated after SGD */\n"
        ));
        s.push_str(&format!(
            "  float *xt{i};  /* act{i}^T [{r}xbatch] — computed after forward */\n"
        ));
        s.push_str(&format!("  float *delta{i}; /* batch x {c} */\n"));
    }
    for i in 0..=n {
        s.push_str(&format!("  float *act{i}; /* batch x {} */\n", layers[i]));
    }
    for i in 0..(n - 1) {
        s.push_str(&format!(
            "  float *pre{i}; /* batch x {} — pre-relu */\n",
            layers[i + 1]
        ));
    }
    s.push_str("  float last_loss;\n  int batch;\n} VoltaTrainHandle;\n\n");

    // init
    s.push_str("VoltaTrainHandle* volta_train_init(int batch) {\n");
    s.push_str("  VoltaTrainHandle* h=(VoltaTrainHandle*)calloc(1,sizeof(VoltaTrainHandle));\n");
    s.push_str("  if(!h) return 0;\n  h->batch=batch;\n");
    for i in 0..n {
        let (r, c) = (layers[i], layers[i + 1]);
        s.push_str(&format!(
            "  h->w{i}=(float*)calloc({r}*{c},4); h->b{i}=(float*)calloc({c},4);\n"
        ));
        s.push_str(&format!(
            "  h->dw{i}=(float*)calloc({r}*{c},4); h->db{i}=(float*)calloc({c},4);\n"
        ));
        s.push_str(&format!("  h->wtb{i}=(float*)calloc({c}*{r},4);\n"));
        s.push_str(&format!("  h->xt{i}=(float*)calloc({r}*batch,4);\n"));
        s.push_str(&format!("  h->delta{i}=(float*)calloc(batch*{c},4);\n"));
    }
    for i in 0..=n {
        s.push_str(&format!(
            "  h->act{i}=(float*)calloc(batch*{},4);\n",
            layers[i]
        ));
    }
    for i in 0..(n - 1) {
        s.push_str(&format!(
            "  h->pre{i}=(float*)calloc(batch*{},4);\n",
            layers[i + 1]
        ));
    }
    // Xavier init
    if init_weights.is_none() {
        s.push_str("  uint64_t rng=42;\n");
        s.push_str("  #define LCG(r) ((r)=(r)*6364136223846793005ULL+1442695040888963407ULL)\n");
        s.push_str("  #define RF(r)  ((float)((LCG(r)>>11)&((1ULL<<53)-1))/(float)(1ULL<<53))\n");
        for i in 0..n {
            let (r, c) = (layers[i], layers[i + 1]);
            s.push_str(&format!("  {{float lim=sqrtf(6.0f/({r}.0f+{c}.0f));for(int j=0;j<{r}*{c};j++) h->w{i}[j]=(RF(rng)*2.0f-1.0f)*lim;}}\n"));
        }
        s.push_str("  #undef LCG\n  #undef RF\n");
        // init wtb from w
        for i in 0..n {
            let (r, c) = (layers[i], layers[i + 1]);
            s.push_str(&format!(
                "  volta_transpose_f32(h->wtb{i},h->w{i},{r},{c});\n"
            ));
        }
    }
    s.push_str("  return h;\n}\n\n");

    // set_params
    s.push_str("void volta_train_set_params(VoltaTrainHandle* h, int li, const float* w, const float* b) {\n");
    for i in 0..n {
        let (r, c) = (layers[i], layers[i + 1]);
        s.push_str(&format!("  if(li=={i}){{memcpy(h->w{i},w,{r}*{c}*4);memcpy(h->b{i},b,{c}*4);volta_transpose_f32(h->wtb{i},h->w{i},{r},{c});return;}}\n"));
    }
    s.push_str("}\n\n");

    // get_params
    s.push_str("void volta_train_get_params(VoltaTrainHandle* h, float** wo, float** bo) {\n");
    for i in 0..n {
        s.push_str(&format!("  wo[{i}]=h->w{i}; bo[{i}]=h->b{i};\n"));
    }
    s.push_str("}\n\n");

    s.push_str("float volta_train_loss(VoltaTrainHandle* h){return h->last_loss;}\n\n");

    // free
    s.push_str("void volta_train_free(VoltaTrainHandle* h) {\n");
    for i in 0..n {
        s.push_str(&format!(
            "  free(h->w{i});free(h->b{i});free(h->dw{i});free(h->db{i});\n"
        ));
        s.push_str(&format!(
            "  free(h->wtb{i});free(h->xt{i});free(h->delta{i});\n"
        ));
    }
    for i in 0..=n {
        s.push_str(&format!("  free(h->act{i});\n"));
    }
    for i in 0..(n - 1) {
        s.push_str(&format!("  free(h->pre{i});\n"));
    }
    s.push_str("  free(h);\n}\n\n");

    // train_step
    s.push_str(
        "void volta_train_step(VoltaTrainHandle* h, const float* X, const float* Y, float lr) {\n",
    );
    s.push_str("  int B=h->batch;\n\n");

    // FORWARD
    s.push_str("  /* FORWARD */\n");
    s.push_str(&format!("  memcpy(h->act0,X,B*{}*4);\n", layers[0]));
    for i in 0..n {
        let (r, c) = (layers[i], layers[i + 1]);
        let is_last = i == n - 1;
        let dst = if is_last {
            format!("h->act{n}")
        } else {
            format!("h->pre{i}")
        };
        // Standard GEMM: act{i}[B×r] @ w{i}[r×c] -> dst[B×c]
        s.push_str(&format!(
            "  volta_gemm_f32({dst},h->act{i},h->w{i},(int64_t)B,(int64_t){r},(int64_t){c});\n"
        ));
        // Bias add: loop is vectorised by clang (inner j over contiguous dst/b)
        s.push_str(&format!(
            "  for(int bi=0;bi<B;bi++) for(int j=0;j<{c};j++) {dst}[bi*{c}+j]+=h->b{i}[j];\n"
        ));
        if !is_last {
            // ReLU activation
            s.push_str(&format!(
                "  for(int k=0;k<B*{c};k++) h->act{}[k]=h->pre{i}[k]>0.0f?h->pre{i}[k]:0.0f;\n",
                i + 1
            ));
        }
        // NOTE: xt{i} transpose is done in backward, not here.
    }

    // MSE loss + seed gradient
    let out_sz = layers[n];
    s.push_str(&format!(
        "\n  /* MSE loss */\n  float lacc=0.0f; int Nt=B*{out_sz};\n"
    ));
    s.push_str(&format!("  for(int k=0;k<Nt;k++){{float d=h->act{n}[k]-Y[k];lacc+=d*d;h->delta{}[k]=2.0f*d/(float)Nt;}}\n", n-1));
    s.push_str("  h->last_loss=lacc/(float)Nt;\n\n");

    // BACKWARD
    s.push_str("  /* BACKWARD */\n");
    for i in (0..n).rev() {
        let (r, c) = (layers[i], layers[i + 1]);
        let is_last = i == n - 1;

        // ReLU backward
        if !is_last {
            s.push_str(&format!(
                "  for(int k=0;k<B*{c};k++) h->delta{i}[k]*=(h->pre{i}[k]>0.0f?1.0f:0.0f);\n"
            ));
        }

        // Compute xt{i} = act{i}^T  [r×B] in backward (act{i} is stable here)
        s.push_str(&format!(
            "  volta_transpose_f32(h->xt{i},h->act{i},(int64_t)B,(int64_t){r});\n"
        ));

        // dW{i} = xt{i} @ delta{i}
        // xt{i}: [r×B], delta{i}: [B×c] → dw{i}: [r×c]
        s.push_str(&format!("  volta_gemm_f32(h->dw{i},h->xt{i},h->delta{i},(int64_t){r},(int64_t)B,(int64_t){c});\n"));

        // db{i} = sum_batch(delta{i}): vectorised inner j-loop
        s.push_str(&format!("  memset(h->db{i},0,{c}*4);\n"));
        s.push_str(&format!(
            "  for(int bi=0;bi<B;bi++) for(int j=0;j<{c};j++) h->db{i}[j]+=h->delta{i}[bi*{c}+j];\n"
        ));

        // Propagate: delta{i-1} = delta{i} @ wtb{i}
        // delta{i}: [B×c], wtb{i}: [c×r] → delta{i-1}: [B×r]
        if i > 0 {
            s.push_str(&format!("  volta_gemm_f32(h->delta{},h->delta{i},h->wtb{i},(int64_t)B,(int64_t){c},(int64_t){r});\n", i-1));
        }

        // SGD update
        let wn = r * c;
        s.push_str(&format!(
            "  for(int k=0;k<{wn};k++) h->w{i}[k]-=lr*h->dw{i}[k];\n"
        ));
        s.push_str(&format!(
            "  for(int k=0;k<{c};k++) h->b{i}[k]-=lr*h->db{i}[k];\n"
        ));

        // Update wtb{i} = w{i}^T after SGD (needed for next step's dX)
        if i > 0 {
            s.push_str(&format!(
                "  volta_transpose_f32(h->wtb{i},h->w{i},{r},{c});\n"
            ));
        }
    }
    s.push_str("}\n");

    // Pre-trained weight loader (optional)
    if let Some(weights) = init_weights {
        s.push_str("\nvoid volta_train_load_pretrained(VoltaTrainHandle* h) {\n");
        for i in 0..n {
            let (r, c) = (layers[i], layers[i + 1]);
            let wname = format!("weight_{}_{}", layers[i], layers[i + 1]);
            let bname = format!("bias_{}", layers[i + 1]);
            if let Some(wd) = weights.get(&wname) {
                let vals: String = wd
                    .iter()
                    .map(|v| format!("{v:.8}f"))
                    .collect::<Vec<_>>()
                    .join(",");
                s.push_str(&format!("  {{static const float _w[]={{{vals}}};memcpy(h->w{i},_w,{r}*{c}*4);volta_transpose_f32(h->wtb{i},h->w{i},{r},{c});}}\n"));
            }
            if let Some(bd) = weights.get(&bname) {
                let vals: String = bd
                    .iter()
                    .map(|v| format!("{v:.8}f"))
                    .collect::<Vec<_>>()
                    .join(",");
                s.push_str(&format!(
                    "  {{static const float _b[]={{{vals}}};memcpy(h->b{i},_b,{c}*4);}}\n"
                ));
            }
        }
        s.push_str("}\n");
    }

    Ok(s)
}
