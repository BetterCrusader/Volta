use volta::executor::Executor;
use volta::lexer::Lexer;
use volta::parser::Parser;
use volta::semantic::SemanticAnalyzer;

fn parse(source: &str) -> Result<volta::ast::Program, volta::parser::ParseError> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

#[test]
fn parser_unknown_keyword_reports_suggestion() {
    let err = parse("trian\n").expect_err("must fail parse");
    assert!(err.message.contains("Unknown keyword"), "{}", err.message);
    assert!(
        err.hint
            .as_deref()
            .is_some_and(|hint| hint.contains("Did you mean 'train'")),
        "unexpected hint: {:?}",
        err.hint
    );
}

#[test]
fn semantic_invalid_symbol_reports_fix_hint() {
    let src = r#"
model brain
    layers 4 2
    activation reluu
dataset tiny
train brain on tiny
"#;
    let program = parse(src).expect("parse must pass");
    let mut analyzer = SemanticAnalyzer::new();
    let err = analyzer.analyze(&program).expect_err("must fail semantic");
    assert!(
        err.message.contains("Invalid value 'reluu'"),
        "unexpected message: {}",
        err.message
    );
    assert!(
        err.hint
            .as_deref()
            .is_some_and(|hint| hint.contains("Did you mean 'relu'")),
        "unexpected hint: {:?}",
        err.hint
    );
}

#[test]
fn runtime_invalid_device_reports_fix_hint() {
    let src = r#"
model m
    layers 2 1
dataset d
train m on d
    device gpuu
"#;
    let program = parse(src).expect("parse must pass");
    let mut executor = Executor::new();
    let err = executor.execute(&program).expect_err("must fail runtime");
    assert!(
        err.message.contains("invalid value 'gpuu'"),
        "unexpected message: {}",
        err.message
    );
    assert!(
        err.hint
            .as_deref()
            .is_some_and(|hint| hint.contains("Did you mean 'gpu'")),
        "unexpected hint: {:?}",
        err.hint
    );
}
