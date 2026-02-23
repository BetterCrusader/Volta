#![no_main]

use libfuzzer_sys::fuzz_target;
use volta::lexer::Lexer;
use volta::parser::Parser;

fuzz_target!(|data: &[u8]| {
    if let Ok(src) = core::str::from_utf8(data) {
        let tokens = Lexer::new(src).tokenize();
        let mut parser = Parser::new(tokens);
        let _ = parser.parse_program();
    }
});
