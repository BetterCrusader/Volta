use volta::executor::Executor;
use volta::lexer::Lexer;
use volta::parser::Parser;
use volta::semantic::SemanticAnalyzer;

fn parse(source: &str) -> volta::ast::Program {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    parser.parse_program().expect("parse failed")
}

fn analyze(program: &volta::ast::Program) {
    let mut analyzer = SemanticAnalyzer::new();
    analyzer.analyze(program).expect("semantic failed");
}

fn execute(program: &volta::ast::Program) {
    let mut executor = Executor::new();
    executor.execute(program).expect("runtime failed");
}

#[test]
fn language_feature_matrix_core_flow_executes() {
    let source = r#"
lr 0.001

model brain
    layers 4 3 2
    activation relu
    optimizer adam
    precision auto

dataset tiny
    batch 2
    shuffle true

fn double(x)
    return x * 2

x double(3)

loop 1
    x = x + 1

train brain on tiny
    epochs 1
    device cpu

if x > 0
    save brain as "brain.vt"
    load brain "brain.vt"
    print "ok"
elif x == 0
    print "zero"
else
    print "negative"
"#;

    let program = parse(source);
    analyze(&program);
    execute(&program);
}

#[test]
fn language_feature_matrix_handles_multiple_expression_forms() {
    let source = r#"
a 10
b 2
c a / b + 3 * 4 - 1

if c >= 0
    print "non-negative"
else
    print "negative"
"#;

    let program = parse(source);
    analyze(&program);
    execute(&program);
}

#[test]
fn language_feature_matrix_returns_structured_errors_for_invalid_scripts() {
    // Missing `on` is a known parser contract violation.
    let bad_parse = "train m d\n    epochs 1\n";
    let mut lexer = Lexer::new(bad_parse);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    let parse_err = parser.parse_program().expect_err("must fail to parse");
    assert!(
        parse_err.message.contains("expected 'on'"),
        "unexpected parse error: {}",
        parse_err.message
    );

    // Undefined model should fail in semantic stage with a non-empty message.
    let semantic_src = "dataset d\ntrain m on d\n";
    let program = parse(semantic_src);
    let mut analyzer = SemanticAnalyzer::new();
    let sem_err = analyzer
        .analyze(&program)
        .expect_err("must fail semantic analysis");
    assert!(
        sem_err.message.contains("Undefined model"),
        "unexpected semantic error: {}",
        sem_err.message
    );
}
