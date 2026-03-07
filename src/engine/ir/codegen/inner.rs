#![allow(unsafe_code)]
/// LLVM IR codegen — forward inference only (phase 1).
///
/// Graph → inkwell Module → object file → native exe via clang.
///
/// All loops over tensor elements are emitted as proper LLVM loops (not
/// unrolled), so even 1024×1024 matrices work without IR bloat.
use std::collections::HashMap;
use std::path::Path;

use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::OptimizationLevel;
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::values::PointerValue;

use crate::ir::{Graph, NodeId, Op, ValueId, build_schedule};

const GEMM_SHIM_C: &[u8] = include_bytes!("gemm_shim.c");

#[derive(Debug)]
pub struct CodegenError {
    pub message: String,
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CodegenError: {}", self.message)
    }
}

impl std::error::Error for CodegenError {}

macro_rules! cg_err {
    ($($t:tt)*) => { CodegenError { message: format!($($t)*) } };
}

// ─── public API ──────────────────────────────────────────────────────────────

pub fn compile_graph_to_object(
    graph: &Graph,
    params: &HashMap<String, Vec<f32>>,
    out_obj: &Path,
) -> Result<(), CodegenError> {
    Target::initialize_x86(&InitializationConfig::default());
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple)
        .map_err(|e| cg_err!("Target from triple: {}", e.to_string()))?;
    let cpu = TargetMachine::get_host_cpu_name();
    let features = TargetMachine::get_host_cpu_features();
    let cpu_str = cpu.to_str().unwrap_or("").to_string();
    let feat_str = features.to_str().unwrap_or("").to_string();
    std::mem::forget(cpu);
    std::mem::forget(features);
    let machine = target
        .create_target_machine(
            &triple,
            &cpu_str,
            &feat_str,
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or_else(|| cg_err!("Failed to create target machine"))?;

    // Leak the context — LLVM-C.dll destructor on Windows can segfault on drop.
    // The process exits shortly after, so the leak is harmless.
    // Leak the context — LLVM-C.dll destructor on Windows segfaults on drop.
    // The process exits shortly after compile, so the leak is harmless.
    let context = Box::leak(Box::new(Context::create()));
    let module = context.create_module("volta_model");
    let builder = context.create_builder();
    emit_infer_fn(context, &module, &builder, graph, params)?;
    drop(builder);

    // Run the full LLVM O3 pass pipeline — enables loop vectorization,
    // SLP vectorization, loop unrolling, inlining, and all scalar opts.
    let pass_opts = PassBuilderOptions::create();
    pass_opts.set_loop_vectorization(true);
    pass_opts.set_loop_slp_vectorization(true);
    pass_opts.set_loop_unrolling(true);
    pass_opts.set_merge_functions(true);
    // Run vectorization + loop optimization passes without the IPO passes
    // that would infer wrong parameter attributes (readnone) on volta_infer.
    // We run individual passes: mem2reg, loop-vectorize, slp-vectorizer,
    // instcombine — avoiding argumentpromotion and function-attrs inference
    // that would mark the output parameter as readnone.
    let passes = "mem2reg,loop-vectorize,slp-vectorizer,loop-unroll,instcombine";
    if let Err(e) = module.run_passes(passes, &machine, pass_opts) {
        eprintln!("[volta codegen] LLVM pass note: {}", e.to_string());
    }

    // Save optimized IR for debugging
    let _ = module.print_to_file(out_obj.with_extension("ll"));
    machine
        .write_to_file(&module, FileType::Object, out_obj)
        .map_err(|e| cg_err!("write_to_file: {}", e.to_string()))?;
    // Leak LLVM objects — drop ordering with LLVM-C on Windows can segfault
    std::mem::forget(module);
    std::mem::forget(machine);
    std::mem::forget(triple);
    Ok(())
}

pub fn link_object_to_exe(obj: &Path, exe: &Path) -> Result<(), CodegenError> {
    let clang = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("clang.exe")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("clang"));

    // Compile gemm_shim.c → gemm_shim.o next to the IR object
    let shim_src = obj.with_file_name("gemm_shim.c");
    std::fs::write(&shim_src, GEMM_SHIM_C).map_err(|e| cg_err!("write gemm_shim.c: {}", e))?;
    let shim_obj = obj.with_file_name("gemm_shim.o");

    let shim_status = std::process::Command::new(&clang)
        .arg("-O3")
        .arg("-march=native")
        .arg("-ffast-math")
        .arg("-funroll-loops")
        .arg("-c")
        .arg(&shim_src)
        .arg("-o")
        .arg(&shim_obj)
        .status()
        .map_err(|e| cg_err!("clang shim compile failed: {}", e))?;
    if !shim_status.success() {
        return Err(cg_err!("gemm_shim.c compile failed"));
    }

    // Link IR object + shim → shared library
    let mut cmd = std::process::Command::new(&clang);
    cmd.arg(obj)
        .arg(&shim_obj)
        .arg("-shared")
        .arg("-O3")
        .arg("-march=native")
        .arg("-o")
        .arg(exe);
    #[cfg(target_os = "windows")]
    {
        cmd.arg("-Wl,/EXPORT:volta_infer");
        cmd.arg("-Wl,/EXPORT:volta_gemm_f32");
    }
    #[cfg(not(target_os = "windows"))]
    cmd.arg("-lm");

    let status = cmd
        .status()
        .map_err(|e| cg_err!("clang link failed: {}", e))?;
    if !status.success() {
        return Err(cg_err!("link failed with status {}", status));
    }
    Ok(())
}

// ─── codegen core ────────────────────────────────────────────────────────────

fn emit_infer_fn<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    graph: &Graph,
    params: &HashMap<String, Vec<f32>>,
) -> Result<(), CodegenError> {
    let f32t = ctx.f32_type();
    let i64t = ctx.i64_type();
    let ptr_t = f32t.ptr_type(AddressSpace::default());

    let fn_type = ctx.void_type().fn_type(
        &[ptr_t.into(), i64t.into(), ptr_t.into(), i64t.into()],
        false,
    );
    let function = module.add_function("volta_infer", fn_type, None);
    // Mark output pointer (param 2) as writeonly+nocapture so O3 cannot
    // infer it readnone and eliminate the output stores.
    let writeonly_id = Attribute::get_named_enum_kind_id("writeonly");
    let nocapture_id = Attribute::get_named_enum_kind_id("nocapture");
    if writeonly_id != 0 {
        let attr = ctx.create_enum_attribute(writeonly_id, 0);
        function.add_attribute(AttributeLoc::Param(2), attr);
    }
    if nocapture_id != 0 {
        let attr = ctx.create_enum_attribute(nocapture_id, 0);
        function.add_attribute(AttributeLoc::Param(2), attr);
    }
    let entry = ctx.append_basic_block(function, "entry");
    builder.position_at_end(entry);

    let schedule = build_schedule(graph).map_err(|e| cg_err!("schedule: {}", e.message))?;

    // Build consumer map: value_id → list of node_ids that consume it
    // Used to detect fuseable Add+ReLU patterns.
    let mut consumers: HashMap<usize, Vec<NodeId>> = HashMap::new();
    for node in &graph.nodes {
        match &node.op {
            Op::Relu(v) | Op::Identity(v) | Op::Output(v) | Op::Transpose(v) => {
                consumers.entry(v.0).or_default().push(node.id);
            }
            Op::Add(l, r) | Op::Sub(l, r) | Op::Mul(l, r) | Op::MatMul(l, r) => {
                consumers.entry(l.0).or_default().push(node.id);
                consumers.entry(r.0).or_default().push(node.id);
            }
            _ => {}
        }
    }
    // Set of Add node output ids that are fused into their downstream Relu
    let mut fused_adds: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for node in &graph.nodes {
        if let Op::Relu(v) = &node.op {
            // Check if v is produced by an Add with a single consumer (this Relu)
            if let Some(add_node) = graph.nodes.get(v.0) {
                if let Op::Add(_, _) = &add_node.op {
                    let uses = consumers.get(&v.0).map(|c| c.len()).unwrap_or(0);
                    if uses == 1 {
                        fused_adds.insert(v.0);
                    }
                }
            }
        }
    }

    let mut value_map: HashMap<usize, PointerValue<'ctx>> = HashMap::new();
    // For fused Add+ReLU: maps Add output id → (lhs ValueId, rhs ValueId)
    let mut fused_add_ops: HashMap<usize, (ValueId, ValueId)> = HashMap::new();

    // Embed weight globals
    for node in &graph.nodes {
        if let Op::Parameter(name) = &node.op {
            let data = params
                .get(name)
                .ok_or_else(|| cg_err!("missing param '{}'", name))?;
            let gptr = embed_f32_global(ctx, module, name, data);
            value_map.insert(node.output.0, gptr);
        }
    }

    let input_ptr = function.get_nth_param(0).unwrap().into_pointer_value();

    for &node_id in schedule.nodes() {
        let node = graph
            .node(node_id)
            .ok_or_else(|| cg_err!("missing node {:?}", node_id))?;

        match &node.op {
            Op::Parameter(_) => {}

            Op::Input(_) => {
                value_map.insert(node.output.0, input_ptr);
            }

            Op::ConstTensor { data, .. } => {
                let name = format!("const_{}", node.output.0);
                let gptr = embed_f32_global(ctx, module, &name, data);
                value_map.insert(node.output.0, gptr);
            }

            Op::MatMul(lhs_id, rhs_id) => {
                let lptr = get_ptr(&value_map, *lhs_id)?;
                let rptr = get_ptr(&value_map, *rhs_id)?;
                let (m, k, n) = infer_matmul_shape(graph, *lhs_id, *rhs_id)?;
                let out = global_f32(ctx, module, m * n, &format!("mm_{}", node.output.0));
                call_volta_gemm(ctx, module, builder, lptr, rptr, out, m, k, n)?;
                value_map.insert(node.output.0, out);
            }

            Op::Add(l, r) => {
                if fused_adds.contains(&node.output.0) {
                    // Fused into downstream Relu — defer emission, record operands.
                    fused_add_ops.insert(node.output.0, (*l, *r));
                    // Insert a placeholder; the Relu will overwrite this entry
                    // with the actual fused output ptr.
                    let placeholder = get_ptr(&value_map, *l)?;
                    value_map.insert(node.output.0, placeholder);
                } else {
                    let n = infer_numel(graph, *l);
                    let out = global_f32(ctx, module, n, &format!("ew_{}", node.output.0));
                    emit_loop_binop(
                        ctx,
                        module,
                        builder,
                        &value_map,
                        function,
                        *l,
                        *r,
                        out,
                        n,
                        |b, a, bv| b.build_float_add(a, bv, "r").unwrap(),
                    )?;
                    value_map.insert(node.output.0, out);
                }
            }

            Op::Sub(l, r) => {
                let n = infer_numel(graph, *l);
                let out = global_f32(ctx, module, n, &format!("ew_{}", node.output.0));
                emit_loop_binop(
                    ctx,
                    module,
                    builder,
                    &value_map,
                    function,
                    *l,
                    *r,
                    out,
                    n,
                    |b, a, bv| b.build_float_sub(a, bv, "r").unwrap(),
                )?;
                value_map.insert(node.output.0, out);
            }

            Op::Mul(l, r) => {
                let n = infer_numel(graph, *l);
                let out = global_f32(ctx, module, n, &format!("ew_{}", node.output.0));
                emit_loop_binop(
                    ctx,
                    module,
                    builder,
                    &value_map,
                    function,
                    *l,
                    *r,
                    out,
                    n,
                    |b, a, bv| b.build_float_mul(a, bv, "r").unwrap(),
                )?;
                value_map.insert(node.output.0, out);
            }

            Op::Relu(v) => {
                let n = infer_numel(graph, *v);
                let out = global_f32(ctx, module, n, &format!("rl_{}", node.output.0));
                if let Some((add_l, add_r)) = fused_add_ops.get(&v.0) {
                    // Fused Add+ReLU: emit single pass out[i] = max(0, l[i] + r[i])
                    let lptr = get_ptr(&value_map, *add_l)?;
                    let rptr = get_ptr(&value_map, *add_r)?;
                    emit_loop_add_relu(ctx, builder, function, lptr, rptr, out, n);
                } else {
                    let src = get_ptr(&value_map, *v)?;
                    emit_loop_relu(ctx, module, builder, function, src, out, n);
                }
                value_map.insert(node.output.0, out);
            }

            Op::Transpose(v) => {
                let (rows, cols) = infer_2d_shape(graph, *v);
                let src = get_ptr(&value_map, *v)?;
                let out = global_f32(ctx, module, rows * cols, &format!("tr_{}", node.output.0));
                emit_transpose_loops(ctx, module, builder, function, src, out, rows, cols);
                value_map.insert(node.output.0, out);
            }

            Op::Identity(v) => {
                let ptr = get_ptr(&value_map, *v)?;
                value_map.insert(node.output.0, ptr);
            }

            Op::Output(v) => {
                let src = get_ptr(&value_map, *v)?;
                let dst = function.get_nth_param(2).unwrap().into_pointer_value();
                let n = infer_numel(graph, *v);
                // Emit a volatile memcpy loop — volatile stores cannot be
                // eliminated by any LLVM optimization pass, ensuring the result
                // is always written to the caller-provided output buffer even
                // when the function is fully optimized.
                emit_volatile_memcpy(ctx, builder, function, dst, src, n);
            }

            Op::ConstInt(_) | Op::ConstFloat(_) | Op::Removed => {}

            op => {
                return Err(cg_err!(
                    "Unsupported op in codegen: {:?}",
                    std::mem::discriminant(op)
                ));
            }
        }
    }

    builder.build_return(None).unwrap();
    Ok(())
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn embed_f32_global<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    name: &str,
    data: &[f32],
) -> PointerValue<'ctx> {
    // Store as [N x i8] with raw bytes — single IR blob, no per-element constants.
    // GEP callers must use i8 type with byte offset, or we bitcast on return.
    // We use i8 type in GEP helpers for weight globals.
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let i8t = ctx.i8_type();
    let arr_t = i8t.array_type(bytes.len() as u32);
    let global = module.add_global(arr_t, Some(AddressSpace::default()), name);
    global.set_constant(true);
    global.set_initializer(&ctx.const_string(&bytes, false));
    // Return pointer — callers treat as f32* (correct on little-endian x86)
    global.as_pointer_value()
}

/// Mutable zero-filled global buffer (used for intermediate computation results).
/// Uses zeroinitializer (null initializer) to avoid building huge LLVM IR arrays.
fn global_f32<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    n: usize,
    name: &str,
) -> PointerValue<'ctx> {
    let f32t = ctx.f32_type();
    let arr_t = f32t.array_type(n as u32);
    let global = module.add_global(arr_t, Some(AddressSpace::default()), name);
    global.set_constant(false);
    // zeroinitializer — no per-element IR constants needed
    global.set_initializer(&arr_t.const_zero());
    global.as_pointer_value()
}

fn get_ptr<'ctx>(
    map: &HashMap<usize, PointerValue<'ctx>>,
    id: ValueId,
) -> Result<PointerValue<'ctx>, CodegenError> {
    map.get(&id.0)
        .copied()
        .ok_or_else(|| cg_err!("value {} not in map", id.0))
}

fn call_volta_gemm<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    a: PointerValue<'ctx>,
    b: PointerValue<'ctx>,
    c: PointerValue<'ctx>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), CodegenError> {
    let f32p = ctx.f32_type().ptr_type(AddressSpace::default());
    let i64t = ctx.i64_type();
    let fn_type = ctx.void_type().fn_type(
        &[
            f32p.into(),
            f32p.into(),
            f32p.into(),
            i64t.into(),
            i64t.into(),
            i64t.into(),
        ],
        false,
    );
    let func = module
        .get_function("volta_gemm_f32")
        .unwrap_or_else(|| module.add_function("volta_gemm_f32", fn_type, None));
    builder
        .build_call(
            func,
            &[
                c.into(),
                a.into(),
                b.into(),
                i64t.const_int(m as u64, false).into(),
                i64t.const_int(k as u64, false).into(),
                i64t.const_int(n as u64, false).into(),
            ],
            "gemm",
        )
        .unwrap();
    Ok(())
}

/// Naive triple-loop matmul — kept for reference, not used when shim is present.
fn emit_matmul_loops<'ctx>(
    ctx: &'ctx Context,
    _module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    function: inkwell::values::FunctionValue<'ctx>,
    a: PointerValue<'ctx>, // m×k row-major
    b: PointerValue<'ctx>, // k×n row-major
    c: PointerValue<'ctx>, // m×n output (pre-zeroed)
    m: usize,
    k: usize,
    n: usize,
) {
    let f32t = ctx.f32_type();
    let i64t = ctx.i64_type();
    let zero_i = i64t.const_int(0, false);
    let zero_f = f32t.const_float(0.0);
    let one = i64t.const_int(1, false);
    let mv = i64t.const_int(m as u64, false);
    let kv = i64t.const_int(k as u64, false);
    let nv = i64t.const_int(n as u64, false);

    // i loop (rows of A/C)
    let pre = builder.get_insert_block().unwrap();
    let i_hdr = ctx.append_basic_block(function, "mm_i");
    let j_hdr = ctx.append_basic_block(function, "mm_j");
    let p_hdr = ctx.append_basic_block(function, "mm_p");
    let p_exit = ctx.append_basic_block(function, "mm_pex");
    let j_exit = ctx.append_basic_block(function, "mm_jex");
    let i_exit = ctx.append_basic_block(function, "mm_iex");

    builder.build_unconditional_branch(i_hdr).unwrap();
    builder.position_at_end(i_hdr);
    let i_phi = builder.build_phi(i64t, "i").unwrap();
    i_phi.add_incoming(&[(&zero_i, pre)]);
    let i = i_phi.as_basic_value().into_int_value();

    builder.build_unconditional_branch(j_hdr).unwrap();
    builder.position_at_end(j_hdr);
    let j_phi = builder.build_phi(i64t, "j").unwrap();
    j_phi.add_incoming(&[(&zero_i, i_hdr)]);
    let j = j_phi.as_basic_value().into_int_value();

    // acc = 0.0 phi across p loop
    builder.build_unconditional_branch(p_hdr).unwrap();
    builder.position_at_end(p_hdr);
    let p_phi = builder.build_phi(i64t, "p").unwrap();
    p_phi.add_incoming(&[(&zero_i, j_hdr)]);
    let acc_phi = builder.build_phi(f32t, "acc").unwrap();
    acc_phi.add_incoming(&[(&zero_f, j_hdr)]);
    let p = p_phi.as_basic_value().into_int_value();
    let acc = acc_phi.as_basic_value().into_float_value();

    // a_idx = i*k + p
    let ik = builder.build_int_mul(i, kv, "ik").unwrap();
    let a_idx = builder.build_int_add(ik, p, "ai").unwrap();
    let ap = unsafe { builder.build_gep(f32t, a, &[a_idx], "ap").unwrap() };
    let av = builder
        .build_load(f32t, ap, "av")
        .unwrap()
        .into_float_value();

    // b_idx = p*n + j
    let pn = builder.build_int_mul(p, nv, "pn").unwrap();
    let b_idx = builder.build_int_add(pn, j, "bi").unwrap();
    let bp = unsafe { builder.build_gep(f32t, b, &[b_idx], "bp").unwrap() };
    let bv = builder
        .build_load(f32t, bp, "bv")
        .unwrap()
        .into_float_value();

    let prod = builder.build_float_mul(av, bv, "prod").unwrap();
    let new_acc = builder.build_float_add(acc, prod, "nacc").unwrap();

    let p_next = builder.build_int_add(p, one, "pnxt").unwrap();
    let p_cond = builder
        .build_int_compare(IntPredicate::SLT, p_next, kv, "pc")
        .unwrap();
    let p_latch = builder.get_insert_block().unwrap();
    p_phi.add_incoming(&[(&p_next, p_latch)]);
    acc_phi.add_incoming(&[(&new_acc, p_latch)]);
    builder
        .build_conditional_branch(p_cond, p_hdr, p_exit)
        .unwrap();
    builder.position_at_end(p_exit);

    // c_idx = i*n + j
    let c_i = builder.build_int_mul(i, nv, "ci").unwrap();
    let c_idx = builder.build_int_add(c_i, j, "cij").unwrap();
    let cp = unsafe { builder.build_gep(f32t, c, &[c_idx], "cp").unwrap() };
    builder.build_store(cp, new_acc).unwrap();

    let j_next = builder.build_int_add(j, one, "jnxt").unwrap();
    let j_cond = builder
        .build_int_compare(IntPredicate::SLT, j_next, nv, "jc")
        .unwrap();
    let j_latch = builder.get_insert_block().unwrap();
    j_phi.add_incoming(&[(&j_next, j_latch)]);
    builder
        .build_conditional_branch(j_cond, j_hdr, j_exit)
        .unwrap();
    builder.position_at_end(j_exit);

    let i_next = builder.build_int_add(i, one, "inxt").unwrap();
    let i_cond = builder
        .build_int_compare(IntPredicate::SLT, i_next, mv, "ic")
        .unwrap();
    let i_latch = builder.get_insert_block().unwrap();
    i_phi.add_incoming(&[(&i_next, i_latch)]);
    builder
        .build_conditional_branch(i_cond, i_hdr, i_exit)
        .unwrap();
    builder.position_at_end(i_exit);
}

// ─── loop-based emitters ─────────────────────────────────────────────────────

/// Fused Add+ReLU: for i in 0..n { out[i] = max(0, lhs[i] + rhs[i]); }
/// Single pass over data — avoids separate Add and ReLU loops.
fn emit_loop_add_relu<'ctx>(
    ctx: &'ctx Context,
    builder: &Builder<'ctx>,
    function: inkwell::values::FunctionValue<'ctx>,
    lptr: PointerValue<'ctx>,
    rptr: PointerValue<'ctx>,
    out: PointerValue<'ctx>,
    n: usize,
) {
    let f32t = ctx.f32_type();
    let i64t = ctx.i64_type();
    let zero_i = i64t.const_int(0, false);
    let zero_f = f32t.const_float(0.0);
    let count = i64t.const_int(n as u64, false);
    let one = i64t.const_int(1, false);

    let pre_bb = builder.get_insert_block().unwrap();
    let loop_bb = ctx.append_basic_block(function, "ar_loop");
    let exit_bb = ctx.append_basic_block(function, "ar_exit");
    builder.build_unconditional_branch(loop_bb).unwrap();
    builder.position_at_end(loop_bb);

    let phi = builder.build_phi(i64t, "i").unwrap();
    phi.add_incoming(&[(&zero_i, pre_bb)]);
    let i = phi.as_basic_value().into_int_value();

    let lp = unsafe { builder.build_gep(f32t, lptr, &[i], "lp").unwrap() };
    let rp = unsafe { builder.build_gep(f32t, rptr, &[i], "rp").unwrap() };
    let lv = builder
        .build_load(f32t, lp, "lv")
        .unwrap()
        .into_float_value();
    let rv = builder
        .build_load(f32t, rp, "rv")
        .unwrap()
        .into_float_value();
    let sum = builder.build_float_add(lv, rv, "sum").unwrap();
    let cmp = builder
        .build_float_compare(inkwell::FloatPredicate::OGT, sum, zero_f, "c")
        .unwrap();
    let res = builder
        .build_select(cmp, sum, zero_f, "r")
        .unwrap()
        .into_float_value();
    let dp = unsafe { builder.build_gep(f32t, out, &[i], "dp").unwrap() };
    builder.build_store(dp, res).unwrap();

    let next = builder.build_int_add(i, one, "next").unwrap();
    let cond = builder
        .build_int_compare(IntPredicate::SLT, next, count, "cond")
        .unwrap();
    let latch = builder.get_insert_block().unwrap();
    phi.add_incoming(&[(&next, latch)]);
    builder
        .build_conditional_branch(cond, loop_bb, exit_bb)
        .unwrap();
    builder.position_at_end(exit_bb);
}

/// Emit:  for i in 0..n { out[i] = op(lhs[i], rhs[i]); }
fn emit_loop_binop<'ctx>(
    ctx: &'ctx Context,
    _module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    map: &HashMap<usize, PointerValue<'ctx>>,
    function: inkwell::values::FunctionValue<'ctx>,
    l: ValueId,
    r: ValueId,
    out: PointerValue<'ctx>,
    n: usize,
    op: impl Fn(
        &Builder<'ctx>,
        inkwell::values::FloatValue<'ctx>,
        inkwell::values::FloatValue<'ctx>,
    ) -> inkwell::values::FloatValue<'ctx>,
) -> Result<(), CodegenError> {
    let lptr = get_ptr(map, l)?;
    let rptr = get_ptr(map, r)?;
    let f32t = ctx.f32_type();
    let i64t = ctx.i64_type();
    let zero = i64t.const_int(0, false);
    let count = i64t.const_int(n as u64, false);
    let one = i64t.const_int(1, false);

    let pre_bb = builder.get_insert_block().unwrap(); // block before loop
    let loop_bb = ctx.append_basic_block(function, "bop_loop");
    let exit_bb = ctx.append_basic_block(function, "bop_exit");
    builder.build_unconditional_branch(loop_bb).unwrap();
    builder.position_at_end(loop_bb);

    let phi = builder.build_phi(i64t, "i").unwrap();
    // First incoming: i=0 from pre_bb
    phi.add_incoming(&[(&zero, pre_bb)]);

    let i = phi.as_basic_value().into_int_value();
    let lp = unsafe { builder.build_gep(f32t, lptr, &[i], "lp").unwrap() };
    let rp = unsafe { builder.build_gep(f32t, rptr, &[i], "rp").unwrap() };
    let lv = builder
        .build_load(f32t, lp, "lv")
        .unwrap()
        .into_float_value();
    let rv = builder
        .build_load(f32t, rp, "rv")
        .unwrap()
        .into_float_value();
    let res = op(builder, lv, rv);
    let dp = unsafe { builder.build_gep(f32t, out, &[i], "dp").unwrap() };
    builder.build_store(dp, res).unwrap();

    let next = builder.build_int_add(i, one, "next").unwrap();
    let cond = builder
        .build_int_compare(IntPredicate::SLT, next, count, "cond")
        .unwrap();
    // Second incoming: i=next from loop_bb
    let latch_bb = builder.get_insert_block().unwrap();
    phi.add_incoming(&[(&next, latch_bb)]);
    builder
        .build_conditional_branch(cond, loop_bb, exit_bb)
        .unwrap();
    builder.position_at_end(exit_bb);
    Ok(())
}

fn emit_loop_relu<'ctx>(
    ctx: &'ctx Context,
    _module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    function: inkwell::values::FunctionValue<'ctx>,
    src: PointerValue<'ctx>,
    out: PointerValue<'ctx>,
    n: usize,
) {
    let f32t = ctx.f32_type();
    let i64t = ctx.i64_type();
    let zero_i = i64t.const_int(0, false);
    let zero_f = f32t.const_float(0.0);
    let count = i64t.const_int(n as u64, false);
    let one = i64t.const_int(1, false);

    let pre_bb = builder.get_insert_block().unwrap();
    let loop_bb = ctx.append_basic_block(function, "relu_loop");
    let exit_bb = ctx.append_basic_block(function, "relu_exit");
    builder.build_unconditional_branch(loop_bb).unwrap();
    builder.position_at_end(loop_bb);

    let phi = builder.build_phi(i64t, "i").unwrap();
    phi.add_incoming(&[(&zero_i, pre_bb)]);
    let i = phi.as_basic_value().into_int_value();

    let sp = unsafe { builder.build_gep(f32t, src, &[i], "sp").unwrap() };
    let sv = builder
        .build_load(f32t, sp, "sv")
        .unwrap()
        .into_float_value();
    let cmp = builder
        .build_float_compare(inkwell::FloatPredicate::OGT, sv, zero_f, "c")
        .unwrap();
    let res = builder
        .build_select(cmp, sv, zero_f, "r")
        .unwrap()
        .into_float_value();
    let dp = unsafe { builder.build_gep(f32t, out, &[i], "dp").unwrap() };
    builder.build_store(dp, res).unwrap();

    let next = builder.build_int_add(i, one, "next").unwrap();
    let cond = builder
        .build_int_compare(IntPredicate::SLT, next, count, "cond")
        .unwrap();
    let latch = builder.get_insert_block().unwrap();
    phi.add_incoming(&[(&next, latch)]);
    builder
        .build_conditional_branch(cond, loop_bb, exit_bb)
        .unwrap();
    builder.position_at_end(exit_bb);
}

fn emit_transpose_loops<'ctx>(
    ctx: &'ctx Context,
    _module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    function: inkwell::values::FunctionValue<'ctx>,
    src: PointerValue<'ctx>,
    out: PointerValue<'ctx>,
    rows: usize,
    cols: usize,
) {
    let f32t = ctx.f32_type();
    let i64t = ctx.i64_type();
    let zero = i64t.const_int(0, false);
    let one = i64t.const_int(1, false);
    let rows_v = i64t.const_int(rows as u64, false);
    let cols_v = i64t.const_int(cols as u64, false);

    // outer loop: r
    let r_loop = ctx.append_basic_block(function, "tr_r");
    let c_loop = ctx.append_basic_block(function, "tr_c");
    let c_exit = ctx.append_basic_block(function, "tr_cexit");
    let r_exit = ctx.append_basic_block(function, "tr_rexit");

    let pre_bb = builder.get_insert_block().unwrap();
    builder.build_unconditional_branch(r_loop).unwrap();
    builder.position_at_end(r_loop);
    let r_phi = builder.build_phi(i64t, "r").unwrap();
    r_phi.add_incoming(&[(&zero, pre_bb)]);
    let r = r_phi.as_basic_value().into_int_value();

    builder.build_unconditional_branch(c_loop).unwrap();
    builder.position_at_end(c_loop);
    let c_phi = builder.build_phi(i64t, "c").unwrap();
    c_phi.add_incoming(&[(&zero, r_loop)]);
    let c = c_phi.as_basic_value().into_int_value();

    // src_idx = r * cols + c
    let rc = builder.build_int_mul(r, cols_v, "rc").unwrap();
    let src_idx = builder.build_int_add(rc, c, "si").unwrap();
    // dst_idx = c * rows + r
    let cr = builder.build_int_mul(c, rows_v, "cr").unwrap();
    let dst_idx = builder.build_int_add(cr, r, "di").unwrap();

    let sp = unsafe { builder.build_gep(f32t, src, &[src_idx], "sp").unwrap() };
    let sv = builder.build_load(f32t, sp, "sv").unwrap();
    let dp = unsafe { builder.build_gep(f32t, out, &[dst_idx], "dp").unwrap() };
    builder.build_store(dp, sv).unwrap();

    let c_next = builder.build_int_add(c, one, "cnext").unwrap();
    let c_cond = builder
        .build_int_compare(IntPredicate::SLT, c_next, cols_v, "cc")
        .unwrap();
    let c_latch = builder.get_insert_block().unwrap();
    c_phi.add_incoming(&[(&c_next, c_latch)]);
    builder
        .build_conditional_branch(c_cond, c_loop, c_exit)
        .unwrap();
    builder.position_at_end(c_exit);

    let r_next = builder.build_int_add(r, one, "rnext").unwrap();
    let r_cond = builder
        .build_int_compare(IntPredicate::SLT, r_next, rows_v, "rc2")
        .unwrap();
    r_phi.add_incoming(&[(&r_next, c_exit)]);
    builder
        .build_conditional_branch(r_cond, r_loop, r_exit)
        .unwrap();
    builder.position_at_end(r_exit);
}

fn emit_loop_memcpy<'ctx>(
    ctx: &'ctx Context,
    _module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    function: inkwell::values::FunctionValue<'ctx>,
    dst: PointerValue<'ctx>,
    src: PointerValue<'ctx>,
    n: usize,
) {
    let f32t = ctx.f32_type();
    let i64t = ctx.i64_type();
    let zero = i64t.const_int(0, false);
    let count = i64t.const_int(n as u64, false);
    let one = i64t.const_int(1, false);

    let pre_bb = builder.get_insert_block().unwrap();
    let loop_bb = ctx.append_basic_block(function, "cpy_loop");
    let exit_bb = ctx.append_basic_block(function, "cpy_exit");
    builder.build_unconditional_branch(loop_bb).unwrap();
    builder.position_at_end(loop_bb);

    let phi = builder.build_phi(i64t, "i").unwrap();
    phi.add_incoming(&[(&zero, pre_bb)]);
    let i = phi.as_basic_value().into_int_value();

    let sp = unsafe { builder.build_gep(f32t, src, &[i], "sp").unwrap() };
    let sv = builder.build_load(f32t, sp, "sv").unwrap();
    let dp = unsafe { builder.build_gep(f32t, dst, &[i], "dp").unwrap() };
    builder.build_store(dp, sv).unwrap();

    let next = builder.build_int_add(i, one, "next").unwrap();
    let cond = builder
        .build_int_compare(IntPredicate::SLT, next, count, "cond")
        .unwrap();
    let cpy_latch = builder.get_insert_block().unwrap();
    phi.add_incoming(&[(&next, cpy_latch)]);
    builder
        .build_conditional_branch(cond, loop_bb, exit_bb)
        .unwrap();
    builder.position_at_end(exit_bb);
}

/// Emit: for i in 0..n { dst[i] = src[i]; }  with volatile stores.
/// Volatile stores cannot be eliminated by any LLVM optimization pass,
/// guaranteeing the output is written to the caller's buffer.
fn emit_volatile_memcpy<'ctx>(
    ctx: &'ctx Context,
    builder: &Builder<'ctx>,
    function: inkwell::values::FunctionValue<'ctx>,
    dst: PointerValue<'ctx>,
    src: PointerValue<'ctx>,
    n: usize,
) {
    let f32t = ctx.f32_type();
    let i64t = ctx.i64_type();
    let zero = i64t.const_int(0, false);
    let count = i64t.const_int(n as u64, false);
    let one = i64t.const_int(1, false);

    let pre_bb = builder.get_insert_block().unwrap();
    let loop_bb = ctx.append_basic_block(function, "vout_loop");
    let exit_bb = ctx.append_basic_block(function, "vout_exit");
    builder.build_unconditional_branch(loop_bb).unwrap();
    builder.position_at_end(loop_bb);

    let phi = builder.build_phi(i64t, "vi").unwrap();
    phi.add_incoming(&[(&zero, pre_bb)]);
    let i = phi.as_basic_value().into_int_value();

    let sp = unsafe { builder.build_gep(f32t, src, &[i], "vsp").unwrap() };
    let sv = builder.build_load(f32t, sp, "vsv").unwrap();
    let dp = unsafe { builder.build_gep(f32t, dst, &[i], "vdp").unwrap() };
    // Volatile store — LLVM cannot eliminate this
    let st = builder.build_store(dp, sv).unwrap();
    st.set_volatile(true).unwrap();

    let next = builder.build_int_add(i, one, "vnxt").unwrap();
    let cond = builder
        .build_int_compare(IntPredicate::SLT, next, count, "vcond")
        .unwrap();
    let latch = builder.get_insert_block().unwrap();
    phi.add_incoming(&[(&next, latch)]);
    builder
        .build_conditional_branch(cond, loop_bb, exit_bb)
        .unwrap();
    builder.position_at_end(exit_bb);
}

// ─── shape inference ─────────────────────────────────────────────────────────

fn infer_matmul_shape(
    graph: &Graph,
    lhs: ValueId,
    rhs: ValueId,
) -> Result<(usize, usize, usize), CodegenError> {
    let ls = infer_shape(graph, lhs);
    let rs = infer_shape(graph, rhs);
    match (ls.as_slice(), rs.as_slice()) {
        ([m, k], [k2, n]) if k == k2 => Ok((*m, *k, *n)),
        _ => Err(cg_err!("Cannot infer matmul shape: {:?} @ {:?}", ls, rs)),
    }
}

fn infer_numel(graph: &Graph, v: ValueId) -> usize {
    infer_shape(graph, v).iter().product::<usize>().max(1)
}

fn infer_2d_shape(graph: &Graph, v: ValueId) -> (usize, usize) {
    let s = infer_shape(graph, v);
    match s.as_slice() {
        [r, c] => (*r, *c),
        _ => (1, s.iter().product()),
    }
}

fn infer_shape(graph: &Graph, v: ValueId) -> Vec<usize> {
    let node = match graph.nodes.get(v.0) {
        Some(n) => n,
        None => return vec![1],
    };
    match &node.op {
        Op::ConstTensor { shape, .. } => shape.clone(),
        Op::Parameter(name) => graph
            .shape_signature
            .parameters
            .get(name)
            .cloned()
            .unwrap_or_else(|| vec![1]),
        Op::Input(name) => graph
            .shape_signature
            .inputs
            .get(name)
            .cloned()
            .unwrap_or_else(|| vec![1]),
        Op::MatMul(l, r) => {
            let ls = infer_shape(graph, *l);
            let rs = infer_shape(graph, *r);
            match (ls.as_slice(), rs.as_slice()) {
                ([m, _], [_, n]) => vec![*m, *n],
                _ => vec![1],
            }
        }
        Op::Transpose(v) => {
            let s = infer_shape(graph, *v);
            if s.len() == 2 { vec![s[1], s[0]] } else { s }
        }
        Op::Relu(v) | Op::Identity(v) | Op::Output(v) => infer_shape(graph, *v),
        Op::Add(l, _) | Op::Sub(l, _) | Op::Mul(l, _) => infer_shape(graph, *l),
        _ => vec![1],
    }
}

#[cfg(test)]
mod tests {
    use super::GEMM_SHIM_C;

    #[test]
    fn shim_bytes_not_empty() {
        assert!(
            GEMM_SHIM_C.len() > 0,
            "GEMM_SHIM_C must be non-empty (include_bytes! embed check)"
        );
    }
}
