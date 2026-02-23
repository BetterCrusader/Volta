//! Benchmarks for the Volta language pipeline.
//!
//! Run with:
//! ```sh
//! cargo bench
//! ```
//!
//! Each benchmark covers one phase of the pipeline so that regressions can be
//! attributed to a specific component (lexer, parser, semantic analysis, or
//! execution).

use volta::executor::Executor;
use volta::lexer::Lexer;
use volta::parser::Parser;
use volta::semantic::SemanticAnalyzer;

// ── Shared source fixtures ────────────────────────────────────────────────────

const SIMPLE_PROGRAM: &str = r#"
x = 42
y = x * 2 + 1
print y
"#;

const MODEL_PROGRAM: &str = r#"
model net
    layers 4 8 4 1
    activation relu
    optimizer adam 0.001

dataset ds
    batch 32
    shuffle true

train net on ds
    epochs 5
    device cpu
"#;

const LOOP_PROGRAM: &str = r#"
counter 0
loop 1000
    counter = counter + 1
print counter
"#;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn compile(src: &str) -> volta::ast::Program {
    let tokens = Lexer::new(src).tokenize();
    Parser::new(tokens).parse_program().expect("parse failed")
}

fn time_bench(label: &str, iters: u32, mut f: impl FnMut()) {
    let start = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    println!("{} ×{}: {:?} avg", label, iters, start.elapsed() / iters);
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    time_bench("lexer (simple)", 10_000, || {
        let _ = Lexer::new(SIMPLE_PROGRAM).tokenize();
    });

    time_bench("parser (model)", 10_000, || {
        let t = Lexer::new(MODEL_PROGRAM).tokenize();
        let _ = Parser::new(t).parse_program();
    });

    let program_semantic = compile(MODEL_PROGRAM);
    time_bench("semantic (model)", 10_000, || {
        let _ = SemanticAnalyzer::new().analyze(&program_semantic);
    });

    let loop_program = compile(LOOP_PROGRAM);
    time_bench("execute (loop 1000)", 10_000, || {
        let mut ex = Executor::new();
        ex.execute(&loop_program).expect("execute failed");
    });

    let model_program = compile(MODEL_PROGRAM);
    time_bench("execute (model+train)", 1_000, || {
        let mut ex = Executor::new();
        ex.execute(&model_program).expect("execute failed");
    });
}
