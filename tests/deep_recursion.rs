#[test]
fn test_parser_recursion() {
    let mut code = String::from("x ");
    for _ in 0..5000 {
        code.push_str("f(");
    }
    code.push('1');
    for _ in 0..5000 {
        code.push(')');
    }
    let mut lexer = volta::lexer::Lexer::new(&code);
    let tokens = lexer.tokenize();
    let mut parser = volta::parser::Parser::new(tokens);
    let _ = parser.parse_program();
}
