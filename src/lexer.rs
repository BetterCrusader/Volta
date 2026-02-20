use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Model,
    Dataset,
    Train,
    Save,
    Load,
    Print,
    Fn,
    Return,
    If,
    Elif,
    Else,
    Loop,
    Precision,
    Memory,
    On,
    As,
    True,
    False,

    // Literals
    Int(i64),
    Float(f64),
    Str(String),

    // Identifier
    Ident(String),

    // Operators
    Greater,
    Less,
    GreaterEq,
    LessEq,
    EqualEqual,
    NotEqual,
    Plus,
    Minus,
    Star,
    Slash,
    Assign,
    Arrow,
    LParen,
    RParen,
    Comma,

    // Structure
    Newline,
    Indent,
    Dedent,
    Eof,

    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub column: usize,
}

pub struct Lexer {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
    at_line_start: bool,
    indent_stack: Vec<usize>,
    pending: VecDeque<Token>,
    eof_emitted: bool,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
        Self {
            input: normalized.chars().collect(),
            position: 0,
            line: 1,
            column: 1,
            at_line_start: true,
            indent_stack: vec![0],
            pending: VecDeque::new(),
            eof_emitted: false,
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            let done = matches!(token.kind, TokenKind::Eof);
            tokens.push(token);
            if done {
                break;
            }
        }
        tokens
    }

    pub fn next_token(&mut self) -> Token {
        if let Some(token) = self.pending.pop_front() {
            return token;
        }

        if self.eof_emitted {
            return self.make_token(TokenKind::Eof, self.line, self.column);
        }

        loop {
            if let Some(token) = self.handle_line_start() {
                return token;
            }

            if self.is_eof() {
                if self.indent_stack.len() > 1 {
                    self.indent_stack.pop();
                    return self.make_token(TokenKind::Dedent, self.line, self.column);
                }
                self.eof_emitted = true;
                return self.make_token(TokenKind::Eof, self.line, self.column);
            }

            let line = self.line;
            let column = self.column;
            let ch = match self.current_char() {
                Some(c) => c,
                None => continue,
            };

            match ch {
                ' ' => {
                    self.advance();
                }
                '\t' => {
                    self.advance();
                    return self.make_token(
                        TokenKind::Error("Tabs are not allowed".to_string()),
                        line,
                        column,
                    );
                }
                '#' => {
                    self.skip_comment();
                }
                '\n' => {
                    self.advance();
                    self.at_line_start = true;
                    return self.make_token(TokenKind::Newline, line, column);
                }
                '0'..='9' => {
                    return self.read_number();
                }
                'a'..='z' | 'A'..='Z' | '_' => {
                    return self.read_identifier();
                }
                '"' => {
                    return self.read_string();
                }
                '>' => {
                    self.advance();
                    if self.match_char('=') {
                        return self.make_token(TokenKind::GreaterEq, line, column);
                    }
                    return self.make_token(TokenKind::Greater, line, column);
                }
                '<' => {
                    self.advance();
                    if self.match_char('=') {
                        return self.make_token(TokenKind::LessEq, line, column);
                    }
                    return self.make_token(TokenKind::Less, line, column);
                }
                '=' => {
                    self.advance();
                    if self.match_char('=') {
                        return self.make_token(TokenKind::EqualEqual, line, column);
                    }
                    return self.make_token(TokenKind::Assign, line, column);
                }
                '!' => {
                    self.advance();
                    if self.match_char('=') {
                        return self.make_token(TokenKind::NotEqual, line, column);
                    }
                    return self.make_token(
                        TokenKind::Error("Unexpected character '!'".to_string()),
                        line,
                        column,
                    );
                }
                '+' => {
                    self.advance();
                    return self.make_token(TokenKind::Plus, line, column);
                }
                '-' => {
                    self.advance();
                    if self.match_char('>') {
                        return self.make_token(TokenKind::Arrow, line, column);
                    }
                    return self.make_token(TokenKind::Minus, line, column);
                }
                '*' => {
                    self.advance();
                    return self.make_token(TokenKind::Star, line, column);
                }
                '/' => {
                    self.advance();
                    return self.make_token(TokenKind::Slash, line, column);
                }
                '(' => {
                    self.advance();
                    return self.make_token(TokenKind::LParen, line, column);
                }
                ')' => {
                    self.advance();
                    return self.make_token(TokenKind::RParen, line, column);
                }
                ',' => {
                    self.advance();
                    return self.make_token(TokenKind::Comma, line, column);
                }
                _ => {
                    self.advance();
                    return self.make_token(
                        TokenKind::Error(format!("Unexpected character '{}'", ch)),
                        line,
                        column,
                    );
                }
            }
        }
    }

    fn handle_line_start(&mut self) -> Option<Token> {
        if !self.at_line_start {
            return None;
        }

        loop {
            let line = self.line;
            let mut indent = 0usize;

            while let Some(ch) = self.current_char() {
                if ch == ' ' {
                    indent += 1;
                    self.advance();
                } else if ch == '\t' {
                    self.advance();
                    self.at_line_start = false;
                    return Some(self.make_token(
                        TokenKind::Error("Tabs are not allowed".to_string()),
                        line,
                        1,
                    ));
                } else {
                    break;
                }
            }

            if self.is_eof() {
                return None;
            }

            match self.current_char() {
                Some('\n') => {
                    self.advance();
                    self.at_line_start = true;
                    return Some(self.make_token(TokenKind::Newline, line, 1));
                }
                Some('#') => {
                    self.skip_comment();
                    if self.current_char() == Some('\n') {
                        self.advance();
                        self.at_line_start = true;
                        return Some(self.make_token(TokenKind::Newline, line, 1));
                    }
                    continue;
                }
                _ => {
                    let current = *self.indent_stack.last().unwrap_or(&0);

                    if indent > current {
                        if indent != current + 4 {
                            self.at_line_start = false;
                            return Some(self.make_token(
                                TokenKind::Error("Inconsistent indentation".to_string()),
                                line,
                                1,
                            ));
                        }
                        self.indent_stack.push(indent);
                        self.at_line_start = false;
                        return Some(self.make_token(TokenKind::Indent, line, 1));
                    }

                    if indent < current {
                        while let Some(&top) = self.indent_stack.last() {
                            if indent < top {
                                self.indent_stack.pop();
                                self.pending
                                    .push_back(self.make_token(TokenKind::Dedent, line, 1));
                            } else {
                                break;
                            }
                        }

                        if *self.indent_stack.last().unwrap_or(&0) != indent {
                            self.pending.clear();
                            self.at_line_start = false;
                            return Some(self.make_token(
                                TokenKind::Error("Inconsistent indentation".to_string()),
                                line,
                                1,
                            ));
                        }

                        self.at_line_start = false;
                        return self.pending.pop_front();
                    }

                    self.at_line_start = false;
                    return None;
                }
            }
        }
    }

    fn read_number(&mut self) -> Token {
        let line = self.line;
        let column = self.column;
        let start = self.position;

        while matches!(self.current_char(), Some('0'..='9')) {
            self.advance();
        }

        let mut is_float = false;
        if self.current_char() == Some('.') && matches!(self.peek_char(), Some('0'..='9')) {
            is_float = true;
            self.advance();
            while matches!(self.current_char(), Some('0'..='9')) {
                self.advance();
            }
        }

        let lexeme: String = self.input[start..self.position].iter().collect();
        if is_float {
            match lexeme.parse::<f64>() {
                Ok(value) => self.make_token(TokenKind::Float(value), line, column),
                Err(_) => self.make_token(
                    TokenKind::Error(format!("Invalid float literal {}", lexeme)),
                    line,
                    column,
                ),
            }
        } else {
            match lexeme.parse::<i64>() {
                Ok(value) => self.make_token(TokenKind::Int(value), line, column),
                Err(_) => self.make_token(
                    TokenKind::Error(format!("Invalid integer literal {}", lexeme)),
                    line,
                    column,
                ),
            }
        }
    }

    fn read_identifier(&mut self) -> Token {
        let line = self.line;
        let column = self.column;
        let start = self.position;

        while matches!(
            self.current_char(),
            Some('a'..='z' | 'A'..='Z' | '0'..='9' | '_')
        ) {
            self.advance();
        }

        let ident: String = self.input[start..self.position].iter().collect();
        let kind = match ident.as_str() {
            "model" => TokenKind::Model,
            "dataset" => TokenKind::Dataset,
            "train" => TokenKind::Train,
            "save" => TokenKind::Save,
            "load" => TokenKind::Load,
            "print" => TokenKind::Print,
            "fn" => TokenKind::Fn,
            "return" => TokenKind::Return,
            "if" => TokenKind::If,
            "elif" => TokenKind::Elif,
            "else" => TokenKind::Else,
            "loop" => TokenKind::Loop,
            "precision" => TokenKind::Precision,
            "memory" => TokenKind::Memory,
            "on" => TokenKind::On,
            "as" => TokenKind::As,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            _ => TokenKind::Ident(ident),
        };

        self.make_token(kind, line, column)
    }

    fn read_string(&mut self) -> Token {
        let line = self.line;
        let column = self.column;
        self.advance();

        let mut value = String::new();
        while let Some(ch) = self.current_char() {
            match ch {
                '"' => {
                    self.advance();
                    return self.make_token(TokenKind::Str(value), line, column);
                }
                '\n' => {
                    return self.make_token(
                        TokenKind::Error("Unclosed string literal".to_string()),
                        line,
                        column,
                    );
                }
                '\t' => {
                    value.push('\t');
                    self.advance();
                }
                '\\' => {
                    self.advance();
                    match self.current_char() {
                        Some('"') => {
                            value.push('"');
                            self.advance();
                        }
                        Some('\\') => {
                            value.push('\\');
                            self.advance();
                        }
                        Some('n') => {
                            value.push('\n');
                            self.advance();
                        }
                        Some('t') => {
                            value.push('\t');
                            self.advance();
                        }
                        Some('r') => {
                            value.push('\r');
                            self.advance();
                        }
                        Some(other) => {
                            value.push(other);
                            self.advance();
                        }
                        None => {
                            return self.make_token(
                                TokenKind::Error("Unclosed string literal".to_string()),
                                line,
                                column,
                            );
                        }
                    }
                }
                _ => {
                    value.push(ch);
                    self.advance();
                }
            }
        }

        self.make_token(
            TokenKind::Error("Unclosed string literal".to_string()),
            line,
            column,
        )
    }

    fn skip_comment(&mut self) {
        while let Some(ch) = self.current_char() {
            if ch == '\n' {
                break;
            }
            self.advance();
        }
    }

    fn current_char(&self) -> Option<char> {
        self.input.get(self.position).copied()
    }

    fn peek_char(&self) -> Option<char> {
        self.input.get(self.position + 1).copied()
    }

    fn advance(&mut self) {
        if let Some(ch) = self.current_char() {
            self.position += 1;
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.current_char() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn is_eof(&self) -> bool {
        self.position >= self.input.len()
    }

    fn make_token(&self, kind: TokenKind, line: usize, column: usize) -> Token {
        Token { kind, line, column }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_indent_and_dedent() {
        let src = "model m\n    x 1\n";
        let mut lexer = Lexer::new(src);
        let tokens = lexer.tokenize();
        let kinds: Vec<TokenKind> = tokens.into_iter().map(|t| t.kind).collect();

        assert!(kinds.iter().any(|k| matches!(k, TokenKind::Indent)));
        assert!(kinds.iter().any(|k| matches!(k, TokenKind::Dedent)));
    }

    #[test]
    fn rejects_tabs() {
        let src = "x 1\n\tx 2\n";
        let mut lexer = Lexer::new(src);
        let tokens = lexer.tokenize();
        assert!(
            tokens
                .iter()
                .any(|t| matches!(&t.kind, TokenKind::Error(msg) if msg.contains("Tabs")))
        );
    }

    #[test]
    fn parses_numbers_and_strings() {
        let src = "a 1\nb 2.5\nc \"ok\"\n";
        let mut lexer = Lexer::new(src);
        let tokens = lexer.tokenize();
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Int(1))));
        assert!(
            tokens
                .iter()
                .any(|t| matches!(t.kind, TokenKind::Float(v) if (v - 2.5).abs() < f64::EPSILON))
        );
        assert!(
            tokens
                .iter()
                .any(|t| matches!(&t.kind, TokenKind::Str(v) if v == "ok"))
        );
    }

    #[test]
    fn lexes_function_tokens() {
        let src = "fn add(a, b)\n    return a\n";
        let mut lexer = Lexer::new(src);
        let tokens = lexer.tokenize();
        let kinds: Vec<TokenKind> = tokens.into_iter().map(|t| t.kind).collect();

        assert!(kinds.iter().any(|k| matches!(k, TokenKind::Fn)));
        assert!(kinds.iter().any(|k| matches!(k, TokenKind::Return)));
        assert!(kinds.iter().any(|k| matches!(k, TokenKind::LParen)));
        assert!(kinds.iter().any(|k| matches!(k, TokenKind::RParen)));
        assert!(kinds.iter().any(|k| matches!(k, TokenKind::Comma)));
    }
}
