use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use volta::ast::{Program, Stmt};
use volta::diagnostics::{best_suggestion, render_diagnostic, render_span_diagnostic};
use volta::executor::Executor;
use volta::lexer::Lexer;
use volta::parser::Parser;
use volta::semantic::SemanticAnalyzer;

const USAGE: &str = "Usage:\n  volta run <file.vt> [--quiet]\n  volta check <file.vt> [--quiet]\n  volta info <file.vt>\n  volta doctor [--json] [--strict]\n  volta init [project_dir]\n  volta version\n  volta help";
const CLI_COMMANDS: [&str; 7] = ["run", "check", "info", "doctor", "init", "version", "help"];
const INIT_MODEL_TEMPLATE: &str = "x 1\nprint x\n";
const INIT_CONFIG_TEMPLATE: &str =
    "[project]\nname = \"volta-project\"\nentry = \"model.vt\"\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandKind {
    Run,
    Check,
    Info,
    Doctor,
    Init,
    Version,
    Help,
    LegacyBenchInfer,
    LegacyTuneMatmul,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommandSpec {
    kind: CommandKind,
    path: Option<String>,
    doctor_json: bool,
    doctor_strict: bool,
    quiet: bool,
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
        CommandKind::Run | CommandKind::Check | CommandKind::Info => {
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

fn parse_command(args: &[String]) -> Result<CommandSpec, String> {
    if args.is_empty() {
        return Ok(CommandSpec {
            kind: CommandKind::Help,
            path: None,
            doctor_json: false,
            doctor_strict: false,
            quiet: false,
        });
    }

    let cmd = args[0].to_ascii_lowercase();
    if cmd == "--bench-infer" {
        return Ok(CommandSpec {
            kind: CommandKind::LegacyBenchInfer,
            path: None,
            doctor_json: false,
            doctor_strict: false,
            quiet: false,
        });
    }
    if cmd == "--tune-matmul" {
        return Ok(CommandSpec {
            kind: CommandKind::LegacyTuneMatmul,
            path: None,
            doctor_json: false,
            doctor_strict: false,
            quiet: false,
        });
    }

    match cmd.as_str() {
        "run" => parse_file_command(CommandKind::Run, args, true),
        "check" => parse_file_command(CommandKind::Check, args, true),
        "info" => parse_file_command(CommandKind::Info, args, false),
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
                doctor_json,
                doctor_strict,
                quiet: false,
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
                doctor_json: false,
                doctor_strict: false,
                quiet: false,
            })
        }
        "help" | "-h" | "--help" => {
            if args.len() != 1 {
                return Err("'help' does not accept positional arguments".to_string());
            }
            Ok(CommandSpec {
                kind: CommandKind::Help,
                path: None,
                doctor_json: false,
                doctor_strict: false,
                quiet: false,
            })
        }
        _ if args.len() == 1 && !cmd.starts_with('-') => {
            let token = args[0].as_str();
            let looks_like_path = token.ends_with(".vt")
                || token.contains('/')
                || token.contains('\\')
                || Path::new(token).exists();
            if looks_like_path {
                Ok(CommandSpec {
                    kind: CommandKind::Run,
                    path: Some(args[0].clone()),
                    doctor_json: false,
                    doctor_strict: false,
                    quiet: false,
                })
            } else {
                Err(unknown_command_message(&args[0]))
            }
        }
        _ => Err(unknown_command_message(&args[0])),
    }
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
        doctor_json: false,
        doctor_strict: false,
        quiet: false,
    })
}

fn unknown_command_message(input: &str) -> String {
    if let Some(suggestion) = best_suggestion(input, &CLI_COMMANDS) {
        return format!(
            "Unknown command '{}'. Did you mean '{}'? Expected run/check/info/doctor/init/version/help",
            input, suggestion
        );
    }
    format!(
        "Unknown command '{}'. Expected run/check/info/doctor/init/version/help",
        input
    )
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
        doctor_json: false,
        doctor_strict: false,
        quiet,
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
    warnings: Vec<String>,
    healthy: bool,
}

fn collect_doctor_report() -> DoctorReport {
    let cpu_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let gpu_env = parse_gpu_env_status();
    let onnx_import_enabled = cfg!(feature = "onnx-import");
    let mut warnings = Vec::new();
    if let Some(warning) = &gpu_env.parse_warning {
        warnings.push(warning.clone());
    }
    let healthy = warnings.is_empty();
    DoctorReport {
        cpu_threads,
        onnx_import_enabled,
        gpu_env,
        warnings,
        healthy,
    }
}

fn print_doctor(report: &DoctorReport, json: bool) {
    if json {
        let raw = report.gpu_env.raw.clone().unwrap_or_default();
        println!(
            "{{\"tool\":\"volta-doctor\",\"version\":\"{}\",\"os\":\"{}\",\"arch\":\"{}\",\"cpu_threads\":{},\"gpu_available\":{},\"gpu_env_raw\":\"{}\",\"gpu_env_valid\":{},\"feature_onnx_import\":{},\"warning_count\":{},\"warnings\":[{}],\"healthy\":{},\"strict_would_fail\":{}}}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH,
            report.cpu_threads,
            report.gpu_env.available,
            json_escape(&raw),
            report.gpu_env.parse_warning.is_none(),
            report.onnx_import_enabled,
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

    println!("Volta doctor");
    println!("  version: {}", env!("CARGO_PKG_VERSION"));
    println!("  os: {}", std::env::consts::OS);
    println!("  arch: {}", std::env::consts::ARCH);
    println!("  cpu_threads: {}", report.cpu_threads);
    println!(
        "  gpu_available: {} (from VOLTA_GPU_AVAILABLE)",
        if report.gpu_env.available {
            "yes"
        } else {
            "no"
        }
    );
    if let Some(raw) = &report.gpu_env.raw {
        println!("  gpu_env_raw: {raw}");
    }
    if report.warnings.is_empty() {
        println!("  warning_count: 0");
    } else {
        println!("  warning_count: {}", report.warnings.len());
        for warning in &report.warnings {
            println!("  warning: {warning}");
        }
    }
    println!("  healthy: {}", if report.healthy { "yes" } else { "no" });
    println!(
        "  strict_would_fail: {}",
        if report.healthy { "no" } else { "yes" }
    );
    println!(
        "  feature_onnx_import: {}",
        if report.onnx_import_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
}

#[derive(Debug, Clone)]
struct GpuEnvStatus {
    available: bool,
    raw: Option<String>,
    parse_warning: Option<String>,
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
    }

    #[test]
    fn parse_command_accepts_init_custom_dir() {
        let args = vec!["init".to_string(), "my-volta-project".to_string()];
        let command = parse_command(&args).expect("init with custom dir should parse");
        assert_eq!(command.kind, CommandKind::Init);
        assert_eq!(command.path.as_deref(), Some("my-volta-project"));
    }

    #[test]
    fn parse_command_rejects_init_flags() {
        let args = vec!["init".to_string(), "--force".to_string()];
        let err = parse_command(&args).expect_err("init flags should fail");
        assert!(err.contains("'init' does not accept flags"));
    }

    #[test]
    fn json_escape_escapes_quotes_and_backslashes() {
        let escaped = super::json_escape("a\"b\\c");
        assert_eq!(escaped, "a\\\"b\\\\c");
    }
}
