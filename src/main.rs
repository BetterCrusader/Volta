use volta::diagnostics::{render_diagnostic, render_span_diagnostic};
use volta::executor::Executor;
use volta::lexer::Lexer;
use volta::parser::Parser;
use volta::semantic::SemanticAnalyzer;

fn main() {
    let source = r#"lr 0.001

model brain
    layers 784 256 128 10
    activation relu
    optimizer adam lr
    precision auto

dataset mnist
    batch 32
    shuffle true

train brain on mnist
    epochs 10
    device auto

accuracy 0.92

if accuracy > 0.95
    save brain as "best.vt"
    print "done"
elif accuracy > 0.8
    print "almost"
else
    print "keep training"
"#;

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();

    let mut parser = Parser::new(tokens);
    match parser.parse_program() {
        Ok(program) => {
            let mut analyzer = SemanticAnalyzer::new();
            match analyzer.analyze(&program) {
                Ok(()) => {
                    for warning in analyzer.warnings() {
                        eprintln!(
                            "{}",
                            render_span_diagnostic(
                                "Warning",
                                &warning.message,
                                warning.span,
                                source,
                                None,
                            )
                        );
                    }

                    let mut executor = Executor::new();
                    match executor.execute(&program) {
                        Ok(()) => {
                            println!("{:#?}", program);
                        }
                        Err(err) => {
                            eprintln!(
                                "{}",
                                render_span_diagnostic(
                                    "Runtime error",
                                    &err.message,
                                    err.span,
                                    source,
                                    err.hint.as_deref(),
                                )
                            );
                        }
                    }
                }
                Err(err) => {
                    eprintln!(
                        "{}",
                        render_span_diagnostic(
                            "Semantic error",
                            &err.message,
                            err.span,
                            source,
                            err.hint.as_deref(),
                        )
                    );
                }
            }
        }
        Err(err) => {
            eprintln!(
                "{}",
                render_diagnostic(
                    "Parse error",
                    &err.message,
                    err.line,
                    err.column,
                    source,
                    err.hint.as_deref(),
                )
            );
        }
    }
}
