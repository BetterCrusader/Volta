use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use volta::ast::{Program, Stmt};
use volta::diagnostics::{render_diagnostic, render_span_diagnostic};
use volta::executor::Executor;
use volta::lexer::Lexer;
use volta::parser::Parser;
use volta::semantic::SemanticAnalyzer;

const USAGE: &str = "Usage:\n  volta run <file.vt>\n  volta check <file.vt>\n  volta info <file.vt>\n  volta version\n  volta help";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandKind {
    Run,
    Check,
    Info,
    Version,
    Help,
    LegacyBenchInfer,
    LegacyTuneMatmul,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommandSpec {
    kind: CommandKind,
    path: Option<String>,
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
                    println!(
                        "Check passed: syntax+semantic OK (warnings: {})",
                        analyzer.warnings().len()
                    );
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
                            println!("Run completed: {path}");
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
        });
    }

    let cmd = args[0].to_ascii_lowercase();
    if cmd == "--bench-infer" {
        return Ok(CommandSpec {
            kind: CommandKind::LegacyBenchInfer,
            path: None,
        });
    }
    if cmd == "--tune-matmul" {
        return Ok(CommandSpec {
            kind: CommandKind::LegacyTuneMatmul,
            path: None,
        });
    }

    match cmd.as_str() {
        "run" => parse_file_command(CommandKind::Run, args),
        "check" => parse_file_command(CommandKind::Check, args),
        "info" => parse_file_command(CommandKind::Info, args),
        "version" | "-v" | "--version" => {
            if args.len() != 1 {
                return Err("'version' does not accept positional arguments".to_string());
            }
            Ok(CommandSpec {
                kind: CommandKind::Version,
                path: None,
            })
        }
        "help" | "-h" | "--help" => {
            if args.len() != 1 {
                return Err("'help' does not accept positional arguments".to_string());
            }
            Ok(CommandSpec {
                kind: CommandKind::Help,
                path: None,
            })
        }
        _ if args.len() == 1 && !cmd.starts_with('-') => Ok(CommandSpec {
            kind: CommandKind::Run,
            path: Some(args[0].clone()),
        }),
        _ => Err(format!(
            "Unknown command '{}'. Expected run/check/info/version/help",
            args[0]
        )),
    }
}

fn parse_file_command(kind: CommandKind, args: &[String]) -> Result<CommandSpec, String> {
    if args.len() != 2 {
        return Err(format!(
            "Command '{}' expects exactly one file path",
            args[0]
        ));
    }
    let path = args[1].trim();
    if path.is_empty() {
        return Err(format!(
            "Command '{}' expects a non-empty file path",
            args[0]
        ));
    }
    Ok(CommandSpec {
        kind,
        path: Some(path.to_string()),
    })
}

fn read_source(path: &str) -> Result<String, String> {
    let p = Path::new(path);
    fs::read_to_string(p).map_err(|err| format!("Failed to read '{}': {}", p.display(), err))
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
    fn parse_command_treats_single_path_as_run_for_compat() {
        let args = vec!["examples/mnist.vt".to_string()];
        let command = parse_command(&args).expect("command should parse");
        assert_eq!(command.kind, CommandKind::Run);
        assert_eq!(command.path.as_deref(), Some("examples/mnist.vt"));
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
}
