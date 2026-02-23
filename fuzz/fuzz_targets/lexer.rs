#![no_main]

use libfuzzer_sys::fuzz_target;
use volta::lexer::Lexer;

fuzz_target!(|data: &[u8]| {
    if let Ok(src) = core::str::from_utf8(data) {
        let _ = Lexer::new(src).tokenize();
    }
});
