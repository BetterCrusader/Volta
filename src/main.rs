use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use volta::ast::{Program, Stmt};
use volta::diagnostics::{best_suggestion, render_diagnostic, render_span_diagnostic};
use volta::executor::Executor;
#[cfg(feature = "cuda")]
use volta::ir::CudaBackend;
use volta::ir::{
    Backend, BackendCapabilities, BackendMaturity, BackendVendor, CpuBackend, DeterminismLevel,
    DeviceClass,
};
use volta::lexer::Lexer;
use volta::parser::Parser;
use volta::semantic::SemanticAnalyzer;

const USAGE: &str = "Usage:
  volta run <file.vt> [--quiet]
  volta check <file.vt> [--quiet]
  volta info <file.vt>
  volta extract <file.gguf|file.safetensors>  (GGUF/SafeTensors → .vt)
  volta export-py <file.vt>
  volta compile <file.vt> [-o <output>]       (requires --features llvm-codegen)
  volta compile-train <file.vt> [-o <output.dll>] [--rust]   (MLP-only today)
  volta doctor [--json] [--strict]
  volta init [project_dir]
  volta version
  volta help";
const CLI_COMMANDS: [&str; 11] = [
    "run",
    "check",
    "info",
    "extract",
    "export-py",
    "compile",
    "compile-train",
    "doctor",
    "init",
    "version",
    "help",
];
const INIT_MODEL_TEMPLATE: &str = "x 1\nprint x\n";
const INIT_CONFIG_TEMPLATE: &str = "[project]\nname = \"volta-project\"\nentry = \"model.vt\"\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandKind {
    Run,
    Check,
    Info,
    Doctor,
    Init,
    Version,
    Help,
    ExportPy,
    Surgeon,
    Compile,
    CompileTrain,
    LegacyBenchInfer,
    LegacyTuneMatmul,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommandSpec {
    kind: CommandKind,
    path: Option<String>,
    output_path: Option<String>,
    doctor_json: bool,
    doctor_strict: bool,
    quiet: bool,
    use_rust: bool,
}

#[derive(Debug, Default)]
struct ProgramStats {
    statements_total: usize,
    var_decl: usize,
    assign: usize,
    model: usize,
    dataset: usize,
    train: usize,
    save: usize,
    load: usize,
    print: usize,
    function: usize,
    return_stmt: usize,
    loop_stmt: usize,
    if_stmt: usize,
}

#[allow(clippy::too_many_lines)]
fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let command = match parse_command(&args) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("{message}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    match command.kind {
        CommandKind::Help => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        CommandKind::Version => {
            println!("volta {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        CommandKind::Doctor => {
            let doctor = collect_doctor_report();
            print_doctor(&doctor, command.doctor_json);
            if command.doctor_strict && !doctor.warnings.is_empty() {
                eprintln!(
                    "doctor --strict failed: {} warning(s) detected",
                    doctor.warnings.len()
                );
                return ExitCode::from(1);
            }
            ExitCode::SUCCESS
        }
        CommandKind::Init => {
            let target_dir = command.path.as_deref().unwrap_or(".");
            match init_project(target_dir) {
                Ok(summary) => {
                    println!("{summary}");
                    ExitCode::SUCCESS
                }
                Err(message) => {
                    eprintln!("{message}");
                    ExitCode::from(1)
                }
            }
        }
        CommandKind::LegacyBenchInfer => {
            println!("Legacy '--bench-infer' mode is deprecated and currently a no-op.");
            ExitCode::SUCCESS
        }
        CommandKind::LegacyTuneMatmul => {
            println!("Legacy '--tune-matmul' mode is deprecated and currently a no-op.");
            ExitCode::SUCCESS
        }
        CommandKind::Surgeon => {
            let Some(path) = command.path.as_deref() else {
                eprintln!("Internal CLI error: missing file path for surgeon");
                return ExitCode::from(2);
            };
            if let Err(e) = volta::utils::surgeon::hunt_and_extract(path) {
                eprintln!("❌ Surgeon error: {}", e);
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }

        CommandKind::Compile => {
            let Some(_path) = command.path.as_deref() else {
                eprintln!("Internal CLI error: missing file path for compile");
                return ExitCode::from(2);
            };
            #[cfg(feature = "llvm-codegen")]
            {
                use volta::executor::Executor;
                let path = _path;
                let source = match read_source(path) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("{e}");
                        return ExitCode::from(2);
                    }
                };
                let mut parser = Parser::new(Lexer::new(&source).tokenize());
                let program = match parser.parse_program() {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("Parse error: {}", e.message);
                        return ExitCode::from(1);
                    }
                };
                // Run the program to train the model (or load weights), then
                // extract the trained graph + weights for codegen.
                let mut executor = Executor::new();
                match executor.execute(&program) {
                    Ok(()) => {}
                    Err(e) => {
                        eprintln!("Runtime error: {}", e.message);
                        return ExitCode::from(1);
                    }
                }
                match executor.compile_first_model_to_object(path, command.output_path.as_deref()) {
                    Ok(exe_path) => {
                        println!("Compiled: {exe_path}");
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("Compile error: {e}");
                        ExitCode::from(1)
                    }
                }
            }
            #[cfg(not(feature = "llvm-codegen"))]
            {
                eprintln!(
                    "'compile' requires the 'llvm-codegen' feature. Rebuild with: cargo build --features llvm-codegen"
                );
                ExitCode::from(1)
            }
        }

        CommandKind::CompileTrain => {
            let Some(path) = command.path.as_deref() else {
                eprintln!("Internal CLI error: missing file path for compile-train");
                return ExitCode::from(2);
            };
            let source = match read_source(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            };
            let mut parser = Parser::new(Lexer::new(&source).tokenize());
            let program = match parser.parse_program() {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Parse error: {}", e.message);
                    return ExitCode::from(1);
                }
            };
            let mut executor = Executor::new();
            // Run the script — this trains the model (or initializes it)
            match executor.execute(&program) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("Runtime error: {}", e.message);
                    return ExitCode::from(1);
                }
            }
            let result = if command.use_rust {
                executor.compile_first_model_to_train_rust_dll(path, command.output_path.as_deref())
            } else {
                executor.compile_first_model_to_train_dll(path, command.output_path.as_deref())
            };
            match result {
                Ok(dll_path) => {
                    println!("Training DLL compiled: {dll_path}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile-train error: {e}");
                    ExitCode::from(1)
                }
            }
        }

        CommandKind::Run | CommandKind::Check | CommandKind::Info | CommandKind::ExportPy => {
            let Some(path) = command.path.as_deref() else {
                eprintln!("Internal CLI error: missing file path for file command");
                return ExitCode::from(2);
            };
            let source = match read_source(path) {
                Ok(source) => source,
                Err(message) => {
                    eprintln!("{message}");
                    return ExitCode::from(2);
                }
            };

            let mut parser = Parser::new(Lexer::new(&source).tokenize());
            let program = match parser.parse_program() {
                Ok(program) => program,
                Err(err) => {
                    eprintln!(
                        "{}",
                        render_diagnostic(
                            "Parse error",
                            &err.message,
                            err.line,
                            err.column,
                            &source,
                            err.hint.as_deref(),
                        )
                    );
                    return ExitCode::from(1);
                }
            };

            let mut analyzer = SemanticAnalyzer::new();
            if let Err(err) = analyzer.analyze(&program) {
                eprintln!(
                    "{}",
                    render_span_diagnostic(
                        "Semantic error",
                        &err.message,
                        err.span,
                        &source,
                        err.hint.as_deref(),
                    )
                );
                return ExitCode::from(1);
            }

            for warning in analyzer.warnings() {
                eprintln!(
                    "{}",
                    render_span_diagnostic(
                        "Warning",
                        &warning.message,
                        warning.span,
                        &source,
                        None,
                    )
                );
            }

            match command.kind {
                CommandKind::Check => {
                    if !command.quiet {
                        println!(
                            "Check passed: syntax+semantic OK (warnings: {})",
                            analyzer.warnings().len()
                        );
                    }
                    ExitCode::SUCCESS
                }
                CommandKind::Info => {
                    print_info(path, &program, analyzer.warnings().len());
                    ExitCode::SUCCESS
                }
                CommandKind::ExportPy => {
                    let python_code =
                        volta::utils::interop::python_exporter::emit_pytorch(&program);
                    println!("{}", python_code);
                    ExitCode::SUCCESS
                }
                CommandKind::Run => {
                    let mut executor = Executor::new();
                    match executor.execute(&program) {
                        Ok(()) => {
                            if !command.quiet {
                                println!("Run completed: {path}");
                            }
                            ExitCode::SUCCESS
                        }
                        Err(err) => {
                            eprintln!(
                                "{}",
                                render_span_diagnostic(
                                    "Runtime error",
                                    &err.message,
                                    err.span,
                                    &source,
                                    err.hint.as_deref(),
                                )
                            );
                            ExitCode::from(1)
                        }
                    }
                }
                _ => ExitCode::from(2),
            }
        }
    }
}

#[allow(clippy::too_many_lines)]
fn parse_command(args: &[String]) -> Result<CommandSpec, String> {
    if args.is_empty() {
        return Ok(CommandSpec {
            kind: CommandKind::Help,
            path: None,
            output_path: None,
            doctor_json: false,
            doctor_strict: false,
            quiet: false,
            use_rust: false,
        });
    }

    let cmd = args[0].to_ascii_lowercase();
    if cmd == "--bench-infer" {
        return Ok(CommandSpec {
            kind: CommandKind::LegacyBenchInfer,
            path: None,
            output_path: None,
            doctor_json: false,
            doctor_strict: false,
            quiet: false,
            use_rust: false,
        });
    }
    if cmd == "--tune-matmul" {
        return Ok(CommandSpec {
            kind: CommandKind::LegacyTuneMatmul,
            path: None,
            output_path: None,
            doctor_json: false,
            doctor_strict: false,
            quiet: false,
            use_rust: false,
        });
    }

    match cmd.as_str() {
        "run" => parse_file_command(CommandKind::Run, args, true),
        "check" => parse_file_command(CommandKind::Check, args, true),
        "info" => parse_file_command(CommandKind::Info, args, false),
        "export-py" => parse_file_command(CommandKind::ExportPy, args, false),
        "compile" => parse_compile_command(args),
        "compile-train" => parse_compile_command_kind(CommandKind::CompileTrain, args),
        "extract" => {
            if args.len() < 2 {
                return Err("'extract' requires a model name or path".to_string());
            }
            Ok(CommandSpec {
                kind: CommandKind::Surgeon,
                path: Some(args[1].clone()),
                output_path: None,
                doctor_json: false,
                doctor_strict: false,
                quiet: false,
                use_rust: false,
            })
        }
        "doctor" => {
            let mut doctor_json = false;
            let mut doctor_strict = false;
            for arg in args.iter().skip(1) {
                match arg.as_str() {
                    "--json" => {
                        if doctor_json {
                            return Err("'doctor --json' was provided more than once".to_string());
                        }
                        doctor_json = true;
                    }
                    "--strict" => {
                        if doctor_strict {
                            return Err("'doctor --strict' was provided more than once".to_string());
                        }
                        doctor_strict = true;
                    }
                    _ => {
                        return Err(
                            "'doctor' accepts only optional '--json' and '--strict'".to_string()
                        );
                    }
                }
            }
            Ok(CommandSpec {
                kind: CommandKind::Doctor,
                path: None,
                output_path: None,
                doctor_json,
                doctor_strict,
                quiet: false,
                use_rust: false,
            })
        }
        "init" => parse_init_command(args),
        "version" | "-v" | "--version" => {
            if args.len() != 1 {
                return Err("'version' does not accept positional arguments".to_string());
            }
            Ok(CommandSpec {
                kind: CommandKind::Version,
                path: None,
                output_path: None,
                doctor_json: false,
                doctor_strict: false,
                quiet: false,
                use_rust: false,
            })
        }
        "help" | "-h" | "--help" => {
            if args.len() != 1 {
                return Err("'help' does not accept positional arguments".to_string());
            }
            Ok(CommandSpec {
                kind: CommandKind::Help,
                path: None,
                output_path: None,
                doctor_json: false,
                doctor_strict: false,
                quiet: false,
                use_rust: false,
            })
        }
        _ if args.len() == 1 && !cmd.starts_with('-') => {
            let token = args[0].as_str();
            let looks_like_path = std::path::Path::new(token)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("vt"))
                || token.contains('/')
                || token.contains('\\')
                || Path::new(token).exists();
            if looks_like_path {
                Ok(CommandSpec {
                    kind: CommandKind::Run,
                    path: Some(args[0].clone()),
                    output_path: None,
                    doctor_json: false,
                    doctor_strict: false,
                    quiet: false,
                    use_rust: false,
                })
            } else {
                Err(unknown_command_message(&args[0]))
            }
        }
        _ => Err(unknown_command_message(&args[0])),
    }
}

fn parse_compile_command(args: &[String]) -> Result<CommandSpec, String> {
    let mut path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut i = 1usize;
    while i < args.len() {
        if args[i] == "-o" {
            let output = args
                .get(i + 1)
                .ok_or_else(|| "'compile -o' requires an output path".to_string())?;
            if output.starts_with('-') {
                return Err("'compile -o' requires an output path".to_string());
            }
            if output_path.is_some() {
                return Err("'compile' accepts '-o' at most once".to_string());
            }
            output_path = Some(output.clone());
            i += 2;
            continue;
        }
        if args[i].starts_with('-') {
            return Err(format!("'compile' does not accept flag '{}'", args[i]));
        }
        if path.is_some() {
            return Err("'compile' expects exactly one .vt file".to_string());
        }
        path = Some(args[i].clone());
        i += 1;
    }
    let Some(path) = path else {
        return Err("'compile' expects a .vt file path".to_string());
    };
    Ok(CommandSpec {
        kind: CommandKind::Compile,
        path: Some(path),
        output_path,
        doctor_json: false,
        doctor_strict: false,
        quiet: false,
        use_rust: false,
    })
}

fn parse_compile_command_kind(kind: CommandKind, args: &[String]) -> Result<CommandSpec, String> {
    let cmd_name = &args[0];
    let mut path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut use_rust = false;
    let mut i = 1usize;
    while i < args.len() {
        if args[i] == "-o" {
            let output = args
                .get(i + 1)
                .ok_or_else(|| format!("'{cmd_name} -o' requires an output path"))?;
            if output.starts_with('-') {
                return Err(format!("'{cmd_name} -o' requires an output path"));
            }
            if output_path.is_some() {
                return Err(format!("'{cmd_name}' accepts '-o' at most once"));
            }
            output_path = Some(output.clone());
            i += 2;
            continue;
        }
        if args[i] == "--rust" {
            if use_rust {
                return Err(format!("'{cmd_name}' accepts '--rust' at most once"));
            }
            use_rust = true;
            i += 1;
            continue;
        }
        if args[i].starts_with('-') {
            return Err(format!("'{cmd_name}' does not accept flag '{}'", args[i]));
        }
        if path.is_some() {
            return Err(format!("'{cmd_name}' expects exactly one .vt file"));
        }
        path = Some(args[i].clone());
        i += 1;
    }
    let Some(path) = path else {
        return Err(format!("'{cmd_name}' expects a .vt file path"));
    };
    Ok(CommandSpec {
        kind,
        path: Some(path),
        output_path,
        doctor_json: false,
        doctor_strict: false,
        quiet: false,
        use_rust,
    })
}

fn parse_init_command(args: &[String]) -> Result<CommandSpec, String> {
    if args.len() > 2 {
        return Err("'init' accepts at most one optional project directory".to_string());
    }
    if args.len() == 2 && args[1].starts_with('-') {
        return Err("'init' does not accept flags".to_string());
    }

    let target_dir = if args.len() == 2 {
        let trimmed = args[1].trim();
        if trimmed.is_empty() {
            return Err("'init' expects a non-empty project directory".to_string());
        }
        trimmed.to_string()
    } else {
        ".".to_string()
    };

    Ok(CommandSpec {
        kind: CommandKind::Init,
        path: Some(target_dir),
        output_path: None,
        doctor_json: false,
        doctor_strict: false,
        quiet: false,
        use_rust: false,
    })
}

fn unknown_command_message(input: &str) -> String {
    const EXPECTED_COMMANDS: &str =
        "run/check/info/extract/export-py/compile/compile-train/doctor/init/version/help";
    if let Some(suggestion) = best_suggestion(input, &CLI_COMMANDS) {
        return format!(
            "Unknown command '{input}'. Did you mean '{suggestion}'? Expected {EXPECTED_COMMANDS}"
        );
    }
    format!("Unknown command '{input}'. Expected {EXPECTED_COMMANDS}")
}

fn parse_file_command(
    kind: CommandKind,
    args: &[String],
    allow_quiet: bool,
) -> Result<CommandSpec, String> {
    let mut quiet = false;
    let mut path: Option<&str> = None;

    for arg in args.iter().skip(1) {
        if arg == "--quiet" {
            if !allow_quiet {
                return Err(format!("Command '{}' does not accept '--quiet'", args[0]));
            }
            if quiet {
                return Err(format!(
                    "Command '{}' received '--quiet' more than once",
                    args[0]
                ));
            }
            quiet = true;
            continue;
        }
        if arg.starts_with('-') {
            if allow_quiet {
                if let Some(suggestion) = best_suggestion(arg, &["--quiet"]) {
                    return Err(format!(
                        "Command '{}' accepts only optional '--quiet' plus one file path (did you mean '{}'?)",
                        args[0], suggestion
                    ));
                }
                return Err(format!(
                    "Command '{}' accepts only optional '--quiet' plus one file path",
                    args[0]
                ));
            }
            return Err(format!("Command '{}' does not accept flags", args[0]));
        }
        if path.is_some() {
            return Err(format!(
                "Command '{}' expects exactly one file path",
                args[0]
            ));
        }
        path = Some(arg.as_str());
    }

    let Some(path) = path else {
        return Err(format!(
            "Command '{}' expects exactly one file path",
            args[0]
        ));
    };

    let path = path.trim();
    if path.is_empty() {
        return Err(format!(
            "Command '{}' expects a non-empty file path",
            args[0]
        ));
    }
    Ok(CommandSpec {
        kind,
        path: Some(path.to_string()),
        output_path: None,
        doctor_json: false,
        doctor_strict: false,
        quiet,
        use_rust: false,
    })
}

fn read_source(path: &str) -> Result<String, String> {
    let p = Path::new(path);
    fs::read_to_string(p).map_err(|err| format!("Failed to read '{}': {}", p.display(), err))
}

fn init_project(target_dir: &str) -> Result<String, String> {
    let dir = Path::new(target_dir);
    if !dir.exists() {
        fs::create_dir_all(dir).map_err(|err| {
            format!(
                "Failed to create project directory '{}': {}",
                dir.display(),
                err
            )
        })?;
    } else if !dir.is_dir() {
        return Err(format!(
            "Init target '{}' exists but is not a directory",
            dir.display()
        ));
    }

    let files = [
        ("model.vt", INIT_MODEL_TEMPLATE),
        ("volta.toml", INIT_CONFIG_TEMPLATE),
    ];

    let mut created = Vec::new();
    let mut skipped = Vec::new();
    for (name, content) in files {
        let path = dir.join(name);
        if path.exists() {
            skipped.push(path.display().to_string());
            continue;
        }
        fs::write(&path, content)
            .map_err(|err| format!("Failed to write '{}': {}", path.display(), err))?;
        created.push(path.display().to_string());
    }

    let mut lines = vec![format!("Initialized Volta project at '{}'", dir.display())];
    if created.is_empty() {
        lines.push("Created: none".to_string());
    } else {
        lines.push(format!("Created: {}", created.join(", ")));
    }
    if !skipped.is_empty() {
        lines.push(format!("Skipped existing: {}", skipped.join(", ")));
    }
    lines.push("Next: volta run model.vt".to_string());
    Ok(lines.join("\n"))
}

#[derive(Debug, Clone)]
struct DoctorReport {
    cpu_threads: usize,
    onnx_import_enabled: bool,
    gpu_env: GpuEnvStatus,
    backends: Vec<BackendDoctorEntry>,
    warnings: Vec<String>,
    healthy: bool,
    mkl_lib_path: Option<String>,
    llvm_info: Option<String>,
    sgd_backend_env: Option<String>,
    llvm_prefix_env: Option<String>,
}

#[derive(Debug, Clone)]
struct BackendDoctorEntry {
    name: String,
    capabilities: BackendCapabilities,
}

fn collect_doctor_report() -> DoctorReport {
    let cpu_threads = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let gpu_env = parse_gpu_env_status();
    let onnx_import_enabled = cfg!(feature = "onnx-import");
    let backends = collect_backend_report();
    let mkl_lib_path = check_mkl_available();
    let llvm_info = check_llvm_available();
    let sgd_backend_env = std::env::var("VOLTA_SGD_BACKEND").ok();
    let llvm_prefix_env = std::env::var("LLVM_SYS_210_PREFIX").ok();
    let mut warnings = Vec::new();
    if let Some(warning) = &gpu_env.parse_warning {
        warnings.push(warning.clone());
    }
    for backend in &backends {
        if backend.capabilities.maturity == BackendMaturity::Experimental {
            warnings.push(format!("backend '{}' is marked experimental", backend.name));
        }
    }
    if mkl_lib_path.is_none() {
        warnings.push(
            "MKL not found — Adam/AdamW --rust codegen will not link; run: conda install -c conda-forge mkl  OR  set MKL_LIB_DIR".to_string(),
        );
    }
    let healthy = warnings.is_empty();
    DoctorReport {
        cpu_threads,
        onnx_import_enabled,
        gpu_env,
        backends,
        warnings,
        healthy,
        mkl_lib_path,
        llvm_info,
        sgd_backend_env,
        llvm_prefix_env,
    }
}

fn print_doctor(report: &DoctorReport, json: bool) {
    if json {
        let raw = report.gpu_env.raw.clone().unwrap_or_default();
        let backends_json = report
            .backends
            .iter()
            .map(|backend| {
                format!(
                    "{{\"name\":\"{}\",\"device_class\":\"{}\",\"vendor\":\"{}\",\"maturity\":\"{}\",\"supports_inference\":{},\"supports_training\":{},\"supports_runtime_execution\":{},\"supports_gradient_updates\":{},\"supports_strict_determinism\":{},\"supports_balanced_determinism\":{},\"supports_fast_determinism\":{},\"default_determinism\":\"{}\"}}",
                    json_escape(&backend.name),
                    device_class_name(backend.capabilities.device_class),
                    backend_vendor_name(backend.capabilities.vendor),
                    backend_maturity_name(backend.capabilities.maturity),
                    backend.capabilities.supports_inference,
                    backend.capabilities.supports_training,
                    backend.capabilities.supports_runtime_execution,
                    backend.capabilities.supports_gradient_updates,
                    backend.capabilities.supports_strict_determinism,
                    backend.capabilities.supports_balanced_determinism,
                    backend.capabilities.supports_fast_determinism,
                    determinism_name(backend.capabilities.default_determinism),
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let mkl_available = report.mkl_lib_path.is_some();
        let mkl_lib_path = report.mkl_lib_path.clone().unwrap_or_default();
        let llvm_available = report.llvm_info.is_some();
        let llvm_info = report.llvm_info.clone().unwrap_or_default();
        println!(
            "{{\"tool\":\"volta-doctor\",\"version\":\"{}\",\"os\":\"{}\",\"arch\":\"{}\",\"cpu_threads\":{},\"gpu_available\":{},\"gpu_env_raw\":\"{}\",\"gpu_env_valid\":{},\"feature_onnx_import\":{},\"mkl_available\":{},\"mkl_lib_path\":\"{}\",\"llvm_available\":{},\"llvm_info\":\"{}\",\"backends\":[{}],\"warning_count\":{},\"warnings\":[{}],\"healthy\":{},\"strict_would_fail\":{}}}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH,
            report.cpu_threads,
            report.gpu_env.available,
            json_escape(&raw),
            report.gpu_env.parse_warning.is_none(),
            report.onnx_import_enabled,
            mkl_available,
            json_escape(&mkl_lib_path),
            llvm_available,
            json_escape(&llvm_info),
            backends_json,
            report.warnings.len(),
            report
                .warnings
                .iter()
                .map(|w| format!("\"{}\"", json_escape(w)))
                .collect::<Vec<_>>()
                .join(","),
            report.healthy,
            !report.healthy
        );
        return;
    }

    // --- Text output: structured sections ---
    println!("--- Volta Doctor ---");
    println!();

    // Environment section
    println!("Environment");
    println!(
        "  os: {} / arch: {} / threads: {}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        report.cpu_threads
    );
    println!();

    // Capability Matrix
    println!("Capability Matrix");
    println!(
        "  {:<8} {:<7} {:<11} {:<10} {:<9} {}",
        "backend", "device", "maturity", "inference", "training", "determinism (default)"
    );
    println!(
        "  {:<8} {:<7} {:<11} {:<10} {:<9} {}",
        "-------", "------", "----------", "---------", "--------", "---------------------"
    );
    for backend in &report.backends {
        let caps = &backend.capabilities;
        let det_parts = {
            let mut parts = Vec::new();
            if caps.supports_strict_determinism {
                parts.push("strict");
            }
            if caps.supports_balanced_determinism {
                parts.push("balanced");
            }
            if caps.supports_fast_determinism {
                parts.push("fast");
            }
            parts.join("/")
        };
        println!(
            "  {:<8} {:<7} {:<11} {:<10} {:<9} {} ({})",
            backend.name,
            device_class_name(caps.device_class),
            backend_maturity_name(caps.maturity),
            yes_no(caps.supports_inference),
            yes_no(caps.supports_training),
            det_parts,
            determinism_name(caps.default_determinism),
        );
    }
    println!();

    // AOT Codegen section
    println!("AOT Codegen (compile-train)");
    println!("  Rust path (--rust):  MLP-only — other architectures rejected at compile time");
    println!("  C path (default):    MLP-only");
    println!();

    // Environment Variables section
    println!("Environment Variables");
    let mkl_lib_dir = std::env::var("MKL_LIB_DIR").unwrap_or_else(|_| "not set".to_string());
    let mklroot = std::env::var("MKLROOT").unwrap_or_else(|_| "not set".to_string());
    let conda_prefix = std::env::var("CONDA_PREFIX").unwrap_or_else(|_| "not set".to_string());
    println!("  MKL_LIB_DIR:         {mkl_lib_dir}");
    println!("  MKLROOT:             {mklroot}");
    println!("  CONDA_PREFIX:        {conda_prefix}");
    let mkl_resolution = match &report.mkl_lib_path {
        Some(path) => format!("OK: {path}"),
        None => "FAIL — Adam/AdamW --rust will not link".to_string(),
    };
    println!("  MKL resolution:      {mkl_resolution}");
    let sgd_backend = report
        .sgd_backend_env
        .as_deref()
        .unwrap_or("not set (default: mkl)");
    println!("  VOLTA_SGD_BACKEND:   {sgd_backend}");
    let llvm_prefix = report.llvm_prefix_env.as_deref().unwrap_or("not set");
    println!("  LLVM_SYS_210_PREFIX: {llvm_prefix}");
    let llvm_clang = match &report.llvm_info {
        Some(desc) => format!("found: {desc}"),
        None => "not found in PATH".to_string(),
    };
    println!("  LLVM (clang):        {llvm_clang}");
    println!();

    // Next Steps section
    println!("Next Steps");
    match &report.mkl_lib_path {
        Some(path) => println!("  [OK] MKL found at {path} — Adam/AdamW --rust codegen available"),
        None => println!(
            "  [WARN] MKL not found — set MKL_LIB_DIR or run: conda install -c conda-forge mkl"
        ),
    }
    match &report.llvm_info {
        Some(_) => println!("  [OK] LLVM found — volta compile available"),
        None => println!(
            "  [WARN] LLVM not found — volta compile requires LLVM 21; set LLVM_SYS_210_PREFIX"
        ),
    }
    // GPU env warnings
    if let Some(warning) = &report.gpu_env.parse_warning {
        println!("  [WARN] {warning}");
    }
    // Backend experimental warnings (excluding MKL warning already handled above)
    for backend in &report.backends {
        if backend.capabilities.maturity == BackendMaturity::Experimental {
            println!(
                "  [WARN] backend '{}' is marked experimental",
                backend.name
            );
        }
    }
    // Only print "all good" line if there's nothing wrong (MKL and LLVM both found, no gpu warnings)
    if report.mkl_lib_path.is_some()
        && report.llvm_info.is_some()
        && report.gpu_env.parse_warning.is_none()
    {
        println!("  [OK] Environment looks good for interpreter and CPU training paths");
    }
    println!();

    println!(
        "healthy: {}  |  warnings: {}",
        if report.healthy { "yes" } else { "no" },
        report.warnings.len()
    );
}

fn collect_backend_report() -> Vec<BackendDoctorEntry> {
    let mut backends = Vec::new();

    let cpu = CpuBackend;
    backends.push(BackendDoctorEntry {
        name: "cpu".to_string(),
        capabilities: cpu.capabilities(),
    });

    #[cfg(feature = "cuda")]
    {
        let cuda = CudaBackend;
        backends.push(BackendDoctorEntry {
            name: "cuda".to_string(),
            capabilities: cuda.capabilities(),
        });
    }

    backends
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn device_class_name(device_class: DeviceClass) -> &'static str {
    match device_class {
        DeviceClass::Cpu => "cpu",
        DeviceClass::Gpu => "gpu",
    }
}

fn backend_vendor_name(vendor: BackendVendor) -> &'static str {
    match vendor {
        BackendVendor::GenericCpu => "generic-cpu",
        BackendVendor::Nvidia => "nvidia",
    }
}

fn backend_maturity_name(maturity: BackendMaturity) -> &'static str {
    match maturity {
        BackendMaturity::Experimental => "experimental",
        BackendMaturity::Validated => "validated",
    }
}

fn determinism_name(level: DeterminismLevel) -> &'static str {
    match level {
        DeterminismLevel::Strict => "strict",
        DeterminismLevel::Balanced => "balanced",
        DeterminismLevel::Fast => "fast",
    }
}

#[derive(Debug, Clone)]
struct GpuEnvStatus {
    available: bool,
    raw: Option<String>,
    parse_warning: Option<String>,
}

/// Inner testable version of MKL detection. Accepts injected env values and lib name.
/// Returns Some(resolved_lib_path) if MKL file exists, None otherwise.
fn check_mkl_from(
    mkl_lib_dir: Option<&str>,
    mklroot: Option<&str>,
    conda_prefix: Option<&str>,
    mkl_lib: &str,
) -> Option<String> {
    let has_mkl = |p: &str| std::path::Path::new(p).join(mkl_lib).exists();

    if let Some(dir) = mkl_lib_dir {
        let dir = dir.replace('\\', "/");
        if has_mkl(&dir) {
            return Some(dir);
        }
    }
    if let Some(root) = mklroot {
        let p = format!("{}/lib", root.replace('\\', "/"));
        if has_mkl(&p) {
            return Some(p);
        }
    }
    if let Some(conda) = conda_prefix {
        let p = format!("{}/Library/lib", conda.replace('\\', "/"));
        if has_mkl(&p) {
            return Some(p);
        }
    }
    None
}

/// Returns Some(resolved_lib_path) if MKL is reachable, None otherwise.
/// Checks file existence (mkl_rt.lib / libmkl_rt.so) — not just env var presence.
fn check_mkl_available() -> Option<String> {
    let mkl_lib = if cfg!(windows) { "mkl_rt.lib" } else { "libmkl_rt.so" };
    check_mkl_from(
        std::env::var("MKL_LIB_DIR").ok().as_deref(),
        std::env::var("MKLROOT").ok().as_deref(),
        std::env::var("CONDA_PREFIX").ok().as_deref(),
        mkl_lib,
    )
}

/// Returns Some(description) if LLVM is reachable, None otherwise.
fn check_llvm_available() -> Option<String> {
    if let Ok(prefix) = std::env::var("LLVM_SYS_210_PREFIX") {
        return Some(format!("via LLVM_SYS_210_PREFIX={prefix}"));
    }
    let result = std::process::Command::new("clang").arg("--version").output();
    if let Ok(out) = result {
        if out.status.success() {
            let ver = String::from_utf8_lossy(&out.stdout);
            let first_line = ver.lines().next().unwrap_or("clang").to_string();
            return Some(format!("clang in PATH: {first_line}"));
        }
    }
    None
}

fn parse_gpu_env_status() -> GpuEnvStatus {
    let raw = std::env::var("VOLTA_GPU_AVAILABLE").ok();
    let Some(value) = raw.clone() else {
        return GpuEnvStatus {
            available: false,
            raw: None,
            parse_warning: None,
        };
    };

    if value == "1" || value.eq_ignore_ascii_case("true") {
        return GpuEnvStatus {
            available: true,
            raw,
            parse_warning: None,
        };
    }

    if value == "0" || value.eq_ignore_ascii_case("false") {
        return GpuEnvStatus {
            available: false,
            raw,
            parse_warning: None,
        };
    }

    GpuEnvStatus {
        available: false,
        raw,
        parse_warning: Some(
            "VOLTA_GPU_AVAILABLE has invalid value; expected one of: 1, true, 0, false".to_string(),
        ),
    }
}

fn json_escape(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn print_info(path: &str, program: &Program, warning_count: usize) {
    let mut stats = ProgramStats::default();
    for stmt in &program.statements {
        collect_stats(stmt, &mut stats);
    }

    println!("File: {path}");
    println!("Top-level statements: {}", program.statements.len());
    println!(
        "Total statements (including nested): {}",
        stats.statements_total
    );
    println!("Warnings: {warning_count}");
    println!("Kinds:");
    println!("  model: {}", stats.model);
    println!("  dataset: {}", stats.dataset);
    println!("  train: {}", stats.train);
    println!("  function: {}", stats.function);
    println!("  if: {}", stats.if_stmt);
    println!("  loop: {}", stats.loop_stmt);
    println!("  var_decl: {}", stats.var_decl);
    println!("  assign: {}", stats.assign);
    println!("  print: {}", stats.print);
    println!("  return: {}", stats.return_stmt);
    println!("  save: {}", stats.save);
    println!("  load: {}", stats.load);
}

fn collect_stats(stmt: &Stmt, stats: &mut ProgramStats) {
    stats.statements_total += 1;
    match stmt {
        Stmt::VarDecl { .. } => stats.var_decl += 1,
        Stmt::Assign { .. } => stats.assign += 1,
        Stmt::Model { .. } => stats.model += 1,
        Stmt::Dataset { .. } => stats.dataset += 1,
        Stmt::Train { .. } => stats.train += 1,
        Stmt::Save { .. } => stats.save += 1,
        Stmt::Load { .. } => stats.load += 1,
        Stmt::Infer { .. } => {}
        Stmt::Print { .. } => stats.print += 1,
        Stmt::Return { .. } => stats.return_stmt += 1,
        Stmt::Function { body, .. } => {
            stats.function += 1;
            for inner in body {
                collect_stats(inner, stats);
            }
        }
        Stmt::Loop { body, .. } => {
            stats.loop_stmt += 1;
            for inner in body {
                collect_stats(inner, stats);
            }
        }
        Stmt::For { body, .. } => {
            stats.loop_stmt += 1;
            for inner in body {
                collect_stats(inner, stats);
            }
        }
        Stmt::If {
            then_branch,
            elif_branches,
            else_branch,
            ..
        } => {
            stats.if_stmt += 1;
            for inner in then_branch {
                collect_stats(inner, stats);
            }
            for (_, branch) in elif_branches {
                for inner in branch {
                    collect_stats(inner, stats);
                }
            }
            if let Some(branch) = else_branch {
                for inner in branch {
                    collect_stats(inner, stats);
                }
            }
        }
        Stmt::Struct { .. } => {
            // Struct stats could be added if needed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CommandKind, parse_command};

    #[test]
    fn parse_command_accepts_run_with_file() {
        let args = vec!["run".to_string(), "example.vt".to_string()];
        let command = parse_command(&args).expect("command should parse");
        assert_eq!(command.kind, CommandKind::Run);
        assert_eq!(command.path.as_deref(), Some("example.vt"));
        assert!(!command.quiet);
        assert!(command.output_path.is_none());
    }

    #[test]
    fn parse_command_rejects_missing_file() {
        let args = vec!["check".to_string()];
        let err = parse_command(&args).expect_err("command should fail");
        assert!(err.contains("expects exactly one file path"));
    }

    #[test]
    fn parse_command_rejects_empty_file_path() {
        let args = vec!["run".to_string(), String::new()];
        let err = parse_command(&args).expect_err("empty path must fail");
        assert!(err.contains("non-empty file path"));
    }

    #[test]
    fn parse_command_defaults_to_help_when_empty() {
        let args = vec![];
        let command = parse_command(&args).expect("command should parse");
        assert_eq!(command.kind, CommandKind::Help);
        assert!(command.path.is_none());
    }

    #[test]
    fn parse_command_accepts_doctor_without_path() {
        let args = vec!["doctor".to_string()];
        let command = parse_command(&args).expect("doctor should parse");
        assert_eq!(command.kind, CommandKind::Doctor);
        assert!(command.path.is_none());
        assert!(!command.doctor_json);
        assert!(!command.doctor_strict);
    }

    #[test]
    fn parse_command_accepts_doctor_json_mode() {
        let args = vec!["doctor".to_string(), "--json".to_string()];
        let command = parse_command(&args).expect("doctor json should parse");
        assert_eq!(command.kind, CommandKind::Doctor);
        assert!(command.path.is_none());
        assert!(command.doctor_json);
        assert!(!command.doctor_strict);
    }

    #[test]
    fn parse_command_accepts_doctor_strict_mode() {
        let args = vec!["doctor".to_string(), "--strict".to_string()];
        let command = parse_command(&args).expect("doctor strict should parse");
        assert_eq!(command.kind, CommandKind::Doctor);
        assert!(command.path.is_none());
        assert!(!command.doctor_json);
        assert!(command.doctor_strict);
    }

    #[test]
    fn parse_command_accepts_doctor_combined_flags() {
        let args = vec![
            "doctor".to_string(),
            "--json".to_string(),
            "--strict".to_string(),
        ];
        let command = parse_command(&args).expect("doctor combined should parse");
        assert_eq!(command.kind, CommandKind::Doctor);
        assert!(command.path.is_none());
        assert!(command.doctor_json);
        assert!(command.doctor_strict);
    }

    #[test]
    fn parse_command_rejects_doctor_unknown_flag() {
        let args = vec!["doctor".to_string(), "--yaml".to_string()];
        let err = parse_command(&args).expect_err("doctor unknown flag must fail");
        assert!(err.contains("accepts only optional '--json' and '--strict'"));
    }

    #[test]
    fn parse_command_rejects_duplicate_doctor_json_flag() {
        let args = vec![
            "doctor".to_string(),
            "--json".to_string(),
            "--json".to_string(),
        ];
        let err = parse_command(&args).expect_err("duplicate --json must fail");
        assert!(err.contains("provided more than once"));
    }

    #[test]
    fn parse_command_rejects_duplicate_doctor_strict_flag() {
        let args = vec![
            "doctor".to_string(),
            "--strict".to_string(),
            "--strict".to_string(),
        ];
        let err = parse_command(&args).expect_err("duplicate --strict must fail");
        assert!(err.contains("provided more than once"));
    }

    #[test]
    fn parse_command_treats_single_path_as_run_for_compat() {
        let args = vec!["examples/mnist.vt".to_string()];
        let command = parse_command(&args).expect("command should parse");
        assert_eq!(command.kind, CommandKind::Run);
        assert_eq!(command.path.as_deref(), Some("examples/mnist.vt"));
        assert!(!command.quiet);
        assert!(command.output_path.is_none());
    }

    #[test]
    fn parse_command_accepts_quiet_before_file_for_run() {
        let args = vec![
            "run".to_string(),
            "--quiet".to_string(),
            "example.vt".to_string(),
        ];
        let command = parse_command(&args).expect("run --quiet should parse");
        assert_eq!(command.kind, CommandKind::Run);
        assert_eq!(command.path.as_deref(), Some("example.vt"));
        assert!(command.quiet);
        assert!(command.output_path.is_none());
    }

    #[test]
    fn parse_command_accepts_quiet_after_file_for_check() {
        let args = vec![
            "check".to_string(),
            "example.vt".to_string(),
            "--quiet".to_string(),
        ];
        let command = parse_command(&args).expect("check --quiet should parse");
        assert_eq!(command.kind, CommandKind::Check);
        assert_eq!(command.path.as_deref(), Some("example.vt"));
        assert!(command.quiet);
        assert!(command.output_path.is_none());
    }

    #[test]
    fn parse_command_rejects_duplicate_quiet_flag() {
        let args = vec![
            "run".to_string(),
            "--quiet".to_string(),
            "--quiet".to_string(),
            "example.vt".to_string(),
        ];
        let err = parse_command(&args).expect_err("duplicate --quiet must fail");
        assert!(err.contains("more than once"));
    }

    #[test]
    fn parse_command_rejects_unknown_flag_for_run() {
        let args = vec![
            "run".to_string(),
            "--verbose".to_string(),
            "example.vt".to_string(),
        ];
        let err = parse_command(&args).expect_err("unknown run flag must fail");
        assert!(err.contains("accepts only optional '--quiet' plus one file path"));
    }

    #[test]
    fn parse_command_suggests_quiet_for_close_run_flag() {
        let args = vec![
            "run".to_string(),
            "--quite".to_string(),
            "example.vt".to_string(),
        ];
        let err = parse_command(&args).expect_err("misspelled quiet must fail");
        assert!(err.contains("did you mean '--quiet'?"));
    }

    #[test]
    fn parse_command_suggests_closest_command_name() {
        let args = vec!["chek".to_string()];
        let err = parse_command(&args).expect_err("unknown command must fail");
        assert!(err.contains("Did you mean 'check'?"));
    }

    #[test]
    fn parse_command_rejects_quiet_for_info() {
        let args = vec![
            "info".to_string(),
            "--quiet".to_string(),
            "example.vt".to_string(),
        ];
        let err = parse_command(&args).expect_err("info must reject --quiet");
        assert!(err.contains("does not accept '--quiet'"));
    }

    #[test]
    fn parse_command_accepts_legacy_bench_flags() {
        let args = vec![
            "--bench-infer".to_string(),
            "--runs".to_string(),
            "1".to_string(),
        ];
        let command = parse_command(&args).expect("command should parse");
        assert_eq!(command.kind, CommandKind::LegacyBenchInfer);

        let args = vec![
            "--tune-matmul".to_string(),
            "--dim".to_string(),
            "64".to_string(),
        ];
        let command = parse_command(&args).expect("command should parse");
        assert_eq!(command.kind, CommandKind::LegacyTuneMatmul);
    }

    #[test]
    fn parse_command_accepts_init_default_dir() {
        let args = vec!["init".to_string()];
        let command = parse_command(&args).expect("init should parse");
        assert_eq!(command.kind, CommandKind::Init);
        assert_eq!(command.path.as_deref(), Some("."));
        assert!(command.output_path.is_none());
    }

    #[test]
    fn parse_command_accepts_init_custom_dir() {
        let args = vec!["init".to_string(), "my-volta-project".to_string()];
        let command = parse_command(&args).expect("init with custom dir should parse");
        assert_eq!(command.kind, CommandKind::Init);
        assert_eq!(command.path.as_deref(), Some("my-volta-project"));
        assert!(command.output_path.is_none());
    }

    #[test]
    fn parse_command_rejects_init_flags() {
        let args = vec!["init".to_string(), "--force".to_string()];
        let err = parse_command(&args).expect_err("init flags should fail");
        assert!(err.contains("'init' does not accept flags"));
    }

    #[test]
    fn parse_compile_accepts_output_path() {
        let args = vec![
            "compile".to_string(),
            "example.vt".to_string(),
            "-o".to_string(),
            "build/out.dll".to_string(),
        ];
        let command = parse_command(&args).expect("compile should parse");
        assert_eq!(command.kind, CommandKind::Compile);
        assert_eq!(command.path.as_deref(), Some("example.vt"));
        assert_eq!(command.output_path.as_deref(), Some("build/out.dll"));
    }

    #[test]
    fn parse_compile_rejects_missing_output_path() {
        let args = vec![
            "compile".to_string(),
            "example.vt".to_string(),
            "-o".to_string(),
        ];
        let err = parse_command(&args).expect_err("missing -o value must fail");
        assert!(err.contains("requires an output path"));
    }

    #[test]
    fn parse_compile_train_accepts_output_path_and_rust() {
        let args = vec![
            "compile-train".to_string(),
            "example.vt".to_string(),
            "-o".to_string(),
            "build/train.dll".to_string(),
            "--rust".to_string(),
        ];
        let command = parse_command(&args).expect("compile-train should parse");
        assert_eq!(command.kind, CommandKind::CompileTrain);
        assert_eq!(command.path.as_deref(), Some("example.vt"));
        assert_eq!(command.output_path.as_deref(), Some("build/train.dll"));
        assert!(command.use_rust);
    }

    #[test]
    fn parse_compile_train_rejects_missing_output_path() {
        let args = vec![
            "compile-train".to_string(),
            "example.vt".to_string(),
            "-o".to_string(),
            "--rust".to_string(),
        ];
        let err = parse_command(&args).expect_err("missing -o value must fail");
        assert!(err.contains("requires an output path"));
    }

    #[test]
    fn json_escape_escapes_quotes_and_backslashes() {
        let escaped = super::json_escape("a\"b\\c");
        assert_eq!(escaped, "a\\\"b\\\\c");
    }

    #[test]
    fn check_mkl_from_returns_none_when_no_vars() {
        let result = super::check_mkl_from(None, None, None, "mkl_rt.lib");
        assert!(result.is_none(), "expected None when no env vars provided");
    }

    #[test]
    fn check_mkl_from_returns_none_when_dir_does_not_contain_lib() {
        // Use a temp dir that exists but has no mkl_rt.lib in it
        let tmp = std::env::temp_dir();
        let tmp_str = tmp.to_string_lossy();
        let result = super::check_mkl_from(
            Some(tmp_str.as_ref()),
            None,
            None,
            "mkl_rt_nonexistent_sentinel_2847.lib",
        );
        assert!(result.is_none(), "expected None when file not present in dir");
    }

    #[test]
    fn check_mkl_from_returns_some_when_mkl_lib_dir_has_file() {
        use std::fs;
        let tmp = std::env::temp_dir().join("volta_mkl_test_dir");
        let _ = fs::create_dir_all(&tmp);
        let lib_file = tmp.join("mkl_rt_test.lib");
        let _ = fs::write(&lib_file, b"stub");
        let tmp_str = tmp.to_string_lossy().replace('\\', "/");
        let result = super::check_mkl_from(Some(&tmp_str), None, None, "mkl_rt_test.lib");
        // Cleanup
        let _ = fs::remove_file(&lib_file);
        let _ = fs::remove_dir(&tmp);
        assert!(result.is_some(), "expected Some path when file exists in MKL_LIB_DIR");
    }

    #[test]
    fn check_mkl_from_returns_some_via_mklroot() {
        use std::fs;
        let tmp = std::env::temp_dir().join("volta_mkl_root_test");
        let lib_dir = tmp.join("lib");
        let _ = fs::create_dir_all(&lib_dir);
        let lib_file = lib_dir.join("mkl_rt_root.lib");
        let _ = fs::write(&lib_file, b"stub");
        let tmp_str = tmp.to_string_lossy().replace('\\', "/");
        let result = super::check_mkl_from(None, Some(&tmp_str), None, "mkl_rt_root.lib");
        // Cleanup
        let _ = fs::remove_file(&lib_file);
        let _ = fs::remove_dir(&lib_dir);
        let _ = fs::remove_dir(&tmp);
        assert!(result.is_some(), "expected Some path when file exists via MKLROOT/lib");
    }

    #[test]
    fn check_mkl_from_mkl_lib_dir_takes_priority_over_mklroot() {
        use std::fs;
        // Set up MKL_LIB_DIR with the file and MKLROOT/lib without it
        let tmp_dir = std::env::temp_dir().join("volta_mkl_prio_dir");
        let _ = fs::create_dir_all(&tmp_dir);
        let lib_file = tmp_dir.join("mkl_rt_prio.lib");
        let _ = fs::write(&lib_file, b"stub");
        let tmp_root = std::env::temp_dir().join("volta_mkl_prio_root");
        let _ = fs::create_dir_all(&tmp_root);
        // MKLROOT/lib doesn't have the file
        let dir_str = tmp_dir.to_string_lossy().replace('\\', "/");
        let root_str = tmp_root.to_string_lossy().replace('\\', "/");
        let result = super::check_mkl_from(
            Some(&dir_str),
            Some(&root_str),
            None,
            "mkl_rt_prio.lib",
        );
        // Cleanup
        let _ = fs::remove_file(&lib_file);
        let _ = fs::remove_dir(&tmp_dir);
        let _ = fs::remove_dir(&tmp_root);
        assert_eq!(
            result.as_deref(),
            Some(dir_str.as_str()),
            "MKL_LIB_DIR should take priority"
        );
    }
}
