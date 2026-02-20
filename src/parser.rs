use crate::ast::{BinaryOp, Expr, Program, Property, Span, Stmt};
use crate::lexer::{Token, TokenKind};

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut statements = Vec::new();
        self.consume_newlines()?;

        while !self.check(&TokenKind::Eof) {
            self.fail_on_lexer_error()?;
            statements.push(self.parse_statement()?);
            self.consume_newlines()?;
        }

        self.expect(TokenKind::Eof)?;
        Ok(Program { statements })
    }

    fn parse_statement(&mut self) -> Result<Stmt, ParseError> {
        self.fail_on_lexer_error()?;

        match &self.current().kind {
            TokenKind::Ident(_) => {
                if self.check_next(&TokenKind::Assign) {
                    self.parse_assign()
                } else {
                    self.parse_var_decl()
                }
            }
            TokenKind::Model => self.parse_model(),
            TokenKind::Dataset => self.parse_dataset(),
            TokenKind::Train => self.parse_train(),
            TokenKind::Save => self.parse_save(),
            TokenKind::Load => self.parse_load(),
            TokenKind::Print => self.parse_print(),
            TokenKind::Fn => self.parse_function(),
            TokenKind::Return => self.parse_return(),
            TokenKind::Loop => self.parse_loop(),
            TokenKind::If => self.parse_if(),
            TokenKind::Precision => self.parse_keyword_var_decl("precision"),
            TokenKind::Memory => self.parse_keyword_var_decl("memory"),
            TokenKind::Elif | TokenKind::Else => {
                Err(self.error_here("Unexpected keyword in this position"))
            }
            TokenKind::Error(msg) => Err(self.error_here(&format!("Lexical error: {msg}"))),
            _ => Err(self.error_here("Unexpected token at start of statement")),
        }
    }

    fn parse_var_decl(&mut self) -> Result<Stmt, ParseError> {
        let (name, name_span) = self.expect_ident("Missing identifier for declaration")?;
        let value = self.parse_expression()?;
        self.require_stmt_terminator()?;
        Ok(Stmt::VarDecl {
            name,
            span: Span::merge(name_span, value.span()),
            value,
        })
    }

    fn parse_assign(&mut self) -> Result<Stmt, ParseError> {
        let (name, name_span) = self.expect_ident("Missing identifier for assignment")?;
        self.expect(TokenKind::Assign)?;
        let value = self.parse_expression()?;
        self.require_stmt_terminator()?;
        Ok(Stmt::Assign {
            name,
            span: Span::merge(name_span, value.span()),
            value,
        })
    }

    fn parse_model(&mut self) -> Result<Stmt, ParseError> {
        let model_token = self.expect(TokenKind::Model)?;
        let (name, name_span) = self.expect_ident("Missing model name")?;
        let (props, block_end) = self.parse_property_block()?;
        Ok(Stmt::Model {
            name,
            props,
            span: Span::merge(token_span(&model_token), end_span(name_span, block_end)),
        })
    }

    fn parse_dataset(&mut self) -> Result<Stmt, ParseError> {
        let dataset_token = self.expect(TokenKind::Dataset)?;
        let (name, name_span) = self.expect_ident("Missing dataset name")?;
        let (props, block_end) = self.parse_optional_property_block(name_span)?;
        Ok(Stmt::Dataset {
            name,
            props,
            span: Span::merge(token_span(&dataset_token), end_span(name_span, block_end)),
        })
    }

    fn parse_train(&mut self) -> Result<Stmt, ParseError> {
        let train_token = self.expect(TokenKind::Train)?;
        let (model, model_span) = self.expect_ident("Missing model name in train statement")?;

        if !self.match_kind(&TokenKind::On) {
            return Err(self.error_here("Invalid train syntax: expected 'on'"));
        }

        let (data, data_span) = self.expect_ident("Missing dataset name in train statement")?;
        let (props, block_end) = self.parse_optional_property_block(data_span)?;

        Ok(Stmt::Train {
            model,
            data,
            props,
            span: Span::merge(
                token_span(&train_token),
                end_span(Span::merge(model_span, data_span), block_end),
            ),
        })
    }

    fn parse_save(&mut self) -> Result<Stmt, ParseError> {
        let save_token = self.expect(TokenKind::Save)?;
        let (model, model_span) = self.expect_ident("Missing model name in save statement")?;

        if !self.match_kind(&TokenKind::As) {
            return Err(self.error_here("Invalid save syntax: expected 'as'"));
        }

        let (path, path_span) = self.expect_string("Missing string path in save statement")?;
        self.require_stmt_terminator()?;
        Ok(Stmt::Save {
            model,
            path,
            span: Span::merge(token_span(&save_token), end_span(model_span, path_span)),
        })
    }

    fn parse_load(&mut self) -> Result<Stmt, ParseError> {
        let load_token = self.expect(TokenKind::Load)?;
        let (model, model_span) = self.expect_ident("Missing model name in load statement")?;
        let (path, path_span) = self.expect_string("Missing string path in load statement")?;
        self.require_stmt_terminator()?;
        Ok(Stmt::Load {
            model,
            path,
            span: Span::merge(token_span(&load_token), end_span(model_span, path_span)),
        })
    }

    fn parse_print(&mut self) -> Result<Stmt, ParseError> {
        let print_token = self.expect(TokenKind::Print)?;
        let expr = self.parse_expression()?;
        self.require_stmt_terminator()?;
        Ok(Stmt::Print {
            span: Span::merge(token_span(&print_token), expr.span()),
            expr,
        })
    }

    fn parse_function(&mut self) -> Result<Stmt, ParseError> {
        let fn_token = self.expect(TokenKind::Fn)?;
        let (name, name_span) = self.expect_ident("Missing function name")?;
        self.expect(TokenKind::LParen)?;

        let mut params = Vec::new();
        if !self.check(&TokenKind::RParen) {
            loop {
                let (param, _) = self.expect_ident("Missing function parameter name")?;
                params.push(param);
                if self.match_kind(&TokenKind::Comma) {
                    continue;
                }
                break;
            }
        }

        self.expect(TokenKind::RParen)?;
        let (body, block_end) = self.parse_statement_block()?;

        Ok(Stmt::Function {
            name,
            params,
            body,
            span: Span::merge(token_span(&fn_token), end_span(name_span, block_end)),
        })
    }

    fn parse_return(&mut self) -> Result<Stmt, ParseError> {
        let ret_token = self.expect(TokenKind::Return)?;
        if self.is_stmt_end() {
            self.require_stmt_terminator()?;
            return Ok(Stmt::Return {
                value: None,
                span: token_span(&ret_token),
            });
        }

        let expr = self.parse_expression()?;
        self.require_stmt_terminator()?;
        Ok(Stmt::Return {
            span: Span::merge(token_span(&ret_token), expr.span()),
            value: Some(expr),
        })
    }

    fn parse_loop(&mut self) -> Result<Stmt, ParseError> {
        let loop_token = self.expect(TokenKind::Loop)?;
        let count = self.parse_expression()?;
        let (body, block_end) = self.parse_statement_block()?;
        Ok(Stmt::Loop {
            span: Span::merge(token_span(&loop_token), end_span(count.span(), block_end)),
            count,
            body,
        })
    }

    fn parse_keyword_var_decl(&mut self, keyword_name: &str) -> Result<Stmt, ParseError> {
        let keyword_token = self.advance().clone();
        let (value_name, value_span) =
            self.expect_ident(&format!("Missing identifier after {keyword_name}"))?;
        self.require_stmt_terminator()?;

        let value = Expr::Ident {
            name: value_name,
            span: value_span,
        };

        Ok(Stmt::VarDecl {
            name: keyword_name.to_string(),
            span: Span::merge(token_span(&keyword_token), value.span()),
            value,
        })
    }

    fn parse_if(&mut self) -> Result<Stmt, ParseError> {
        let if_token = self.expect(TokenKind::If)?;
        let condition = self.parse_expression()?;
        let (then_branch, mut end_of_if) = self.parse_statement_block()?;

        let mut elif_branches = Vec::new();
        while self.match_kind(&TokenKind::Elif) {
            let elif_condition = self.parse_expression()?;
            let (elif_body, elif_end) = self.parse_statement_block()?;
            end_of_if = end_span(end_of_if, elif_end);
            elif_branches.push((elif_condition, elif_body));
        }

        let else_branch = if self.match_kind(&TokenKind::Else) {
            let (else_body, else_end) = self.parse_statement_block()?;
            end_of_if = end_span(end_of_if, else_end);
            Some(else_body)
        } else {
            None
        };

        Ok(Stmt::If {
            condition,
            then_branch,
            elif_branches,
            else_branch,
            span: Span::merge(token_span(&if_token), end_of_if),
        })
    }

    fn parse_property_block(&mut self) -> Result<(Vec<Property>, Span), ParseError> {
        self.expect(TokenKind::Newline)?;
        self.expect(TokenKind::Indent)
            .map_err(|_| self.error_here("Missing indent for property block"))?;

        let mut props = Vec::new();
        self.consume_newlines()?;

        while !self.check(&TokenKind::Dedent) {
            if self.check(&TokenKind::Eof) {
                return Err(self.error_here("Missing dedent at end of property block"));
            }

            props.push(self.parse_property()?);
            self.consume_newlines()?;
        }

        let dedent = self.expect(TokenKind::Dedent)?;
        Ok((props, token_span(&dedent)))
    }

    fn parse_optional_property_block(
        &mut self,
        fallback_end: Span,
    ) -> Result<(Vec<Property>, Span), ParseError> {
        if self.check(&TokenKind::Newline) && self.check_next(&TokenKind::Indent) {
            self.parse_property_block()
        } else {
            self.require_stmt_terminator()?;
            Ok((Vec::new(), fallback_end))
        }
    }

    fn parse_property(&mut self) -> Result<Property, ParseError> {
        let (key, key_span) = match self.current().kind.clone() {
            TokenKind::Ident(name) => {
                let token = self.advance().clone();
                (name, token_span(&token))
            }
            TokenKind::Precision => {
                let token = self.advance().clone();
                ("precision".to_string(), token_span(&token))
            }
            TokenKind::Memory => {
                let token = self.advance().clone();
                ("memory".to_string(), token_span(&token))
            }
            _ => return Err(self.error_here("Invalid property structure: missing property key")),
        };

        let mut values = Vec::new();
        while !self.is_stmt_end() {
            values.push(self.parse_property_value()?);
        }

        if values.is_empty() {
            return Err(self.error_here("Invalid property structure: missing property value"));
        }

        let values_end = values.last().map(|v| v.span()).unwrap_or(key_span);
        self.require_stmt_terminator()?;

        Ok(Property {
            key,
            values,
            span: Span::merge(key_span, values_end),
        })
    }

    fn parse_statement_block(&mut self) -> Result<(Vec<Stmt>, Span), ParseError> {
        self.expect(TokenKind::Newline)?;
        self.expect(TokenKind::Indent)
            .map_err(|_| self.error_here("Missing indent for block"))?;

        let mut body = Vec::new();
        self.consume_newlines()?;

        while !self.check(&TokenKind::Dedent) {
            if self.check(&TokenKind::Eof) {
                return Err(self.error_here("Missing dedent at end of block"));
            }

            body.push(self.parse_statement()?);
            self.consume_newlines()?;
        }

        let dedent = self.expect(TokenKind::Dedent)?;
        Ok((body, token_span(&dedent)))
    }

    fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_term()?;

        loop {
            let op = if self.match_kind(&TokenKind::Greater) {
                Some(BinaryOp::Greater)
            } else if self.match_kind(&TokenKind::Less) {
                Some(BinaryOp::Less)
            } else if self.match_kind(&TokenKind::GreaterEq) {
                Some(BinaryOp::GreaterEq)
            } else if self.match_kind(&TokenKind::LessEq) {
                Some(BinaryOp::LessEq)
            } else if self.match_kind(&TokenKind::EqualEqual) {
                Some(BinaryOp::Equal)
            } else if self.match_kind(&TokenKind::NotEqual) {
                Some(BinaryOp::NotEqual)
            } else {
                None
            };

            let Some(op) = op else {
                break;
            };

            let right = self.parse_term()?;
            let span = Span::merge(expr.span(), right.span());
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
                span,
            };
        }

        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_factor()?;

        loop {
            let op = if self.match_kind(&TokenKind::Plus) {
                Some(BinaryOp::Add)
            } else if self.match_kind(&TokenKind::Minus) {
                Some(BinaryOp::Sub)
            } else {
                None
            };

            let Some(op) = op else {
                break;
            };

            let right = self.parse_factor()?;
            let span = Span::merge(expr.span(), right.span());
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
                span,
            };
        }

        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_unary()?;

        loop {
            let op = if self.match_kind(&TokenKind::Star) {
                Some(BinaryOp::Mul)
            } else if self.match_kind(&TokenKind::Slash) {
                Some(BinaryOp::Div)
            } else {
                None
            };

            let Some(op) = op else {
                break;
            };

            let right = self.parse_unary()?;
            let span = Span::merge(expr.span(), right.span());
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(right),
                span,
            };
        }

        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if self.match_kind(&TokenKind::Minus) {
            let minus_span = token_span(&self.tokens[self.position.saturating_sub(1)]);
            let right = self.parse_unary()?;
            return Ok(Expr::Binary {
                left: Box::new(Expr::Int {
                    value: 0,
                    span: minus_span,
                }),
                op: BinaryOp::Sub,
                right: Box::new(right.clone()),
                span: Span::merge(minus_span, right.span()),
            });
        }

        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        self.fail_on_lexer_error()?;

        let token = self.advance().clone();
        let span = token_span(&token);
        match token.kind {
            TokenKind::Int(v) => Ok(Expr::Int { value: v, span }),
            TokenKind::Float(v) => Ok(Expr::Float { value: v, span }),
            TokenKind::Str(v) => Ok(Expr::Str { value: v, span }),
            TokenKind::True => Ok(Expr::Bool { value: true, span }),
            TokenKind::False => Ok(Expr::Bool { value: false, span }),
            TokenKind::Ident(name) => Ok(Expr::Ident { name, span }),
            _ => Err(ParseError {
                message: "Expected expression".to_string(),
                line: token.line,
                column: token.column,
            }),
        }
        .and_then(|expr| self.finish_call(expr))
    }

    fn finish_call(&mut self, expr: Expr) -> Result<Expr, ParseError> {
        if !self.check(&TokenKind::LParen) {
            return Ok(expr);
        }

        let (callee, callee_span) = match expr {
            Expr::Ident { name, span } => (name, span),
            _ => {
                return Err(self.error_here("Only identifier can be used as function callee"));
            }
        };

        self.expect(TokenKind::LParen)?;
        let mut args = Vec::new();
        if !self.check(&TokenKind::RParen) {
            loop {
                args.push(self.parse_expression()?);
                if self.match_kind(&TokenKind::Comma) {
                    continue;
                }
                break;
            }
        }
        let closing = self.expect(TokenKind::RParen)?;

        Ok(Expr::Call {
            callee,
            args,
            span: Span::merge(callee_span, token_span(&closing)),
        })
    }

    fn parse_property_value(&mut self) -> Result<Expr, ParseError> {
        self.fail_on_lexer_error()?;

        if self.match_kind(&TokenKind::Minus) {
            self.fail_on_lexer_error()?;
            let minus = self.tokens[self.position.saturating_sub(1)].clone();
            let next = self.current().clone();
            return match next.kind {
                TokenKind::Int(v) => {
                    self.advance();
                    match v.checked_neg() {
                        Some(n) => Ok(Expr::Int {
                            value: n,
                            span: Span::merge(token_span(&minus), token_span(&next)),
                        }),
                        None => Err(ParseError {
                            message: "Invalid property value; integer underflow".to_string(),
                            line: next.line,
                            column: next.column,
                        }),
                    }
                }
                TokenKind::Float(v) => {
                    self.advance();
                    Ok(Expr::Float {
                        value: -v,
                        span: Span::merge(token_span(&minus), token_span(&next)),
                    })
                }
                _ => Err(ParseError {
                    message: "Invalid property value; '-' must be followed by number".to_string(),
                    line: next.line,
                    column: next.column,
                }),
            };
        }

        let token = self.advance().clone();
        let span = token_span(&token);
        match token.kind {
            TokenKind::Int(v) => Ok(Expr::Int { value: v, span }),
            TokenKind::Float(v) => Ok(Expr::Float { value: v, span }),
            TokenKind::Str(v) => Ok(Expr::Str { value: v, span }),
            TokenKind::True => Ok(Expr::Bool { value: true, span }),
            TokenKind::False => Ok(Expr::Bool { value: false, span }),
            TokenKind::Ident(name) => Ok(Expr::Ident { name, span }),
            _ => Err(ParseError {
                message: "Invalid property value; expected literal or identifier".to_string(),
                line: token.line,
                column: token.column,
            }),
        }
    }

    fn current(&self) -> &Token {
        let idx = if self.position < self.tokens.len() {
            self.position
        } else {
            self.tokens.len().saturating_sub(1)
        };
        &self.tokens[idx]
    }

    fn advance(&mut self) -> &Token {
        let idx = self.position;
        if self.position < self.tokens.len() {
            self.position += 1;
        }
        let safe_idx = if idx < self.tokens.len() {
            idx
        } else {
            self.tokens.len().saturating_sub(1)
        };
        &self.tokens[safe_idx]
    }

    fn expect(&mut self, expected: TokenKind) -> Result<Token, ParseError> {
        self.fail_on_lexer_error()?;

        if self.check(&expected) {
            Ok(self.advance().clone())
        } else {
            Err(ParseError {
                message: format!(
                    "Expected {:?}, found {:?}",
                    expected,
                    self.current().kind.clone()
                ),
                line: self.current().line,
                column: self.current().column,
            })
        }
    }

    fn match_kind(&mut self, expected: &TokenKind) -> bool {
        if self.check(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn consume_newlines(&mut self) -> Result<(), ParseError> {
        while self.check(&TokenKind::Newline) {
            self.advance();
            self.fail_on_lexer_error()?;
        }
        Ok(())
    }

    fn expect_ident(&mut self, msg: &str) -> Result<(String, Span), ParseError> {
        self.fail_on_lexer_error()?;
        let token = self.current().clone();
        match token.kind {
            TokenKind::Ident(ref name) => {
                self.advance();
                Ok((name.clone(), token_span(&token)))
            }
            kind if is_keyword_token(&kind) => Err(ParseError {
                message: "Unexpected keyword where identifier was expected".to_string(),
                line: token.line,
                column: token.column,
            }),
            _ => Err(ParseError {
                message: msg.to_string(),
                line: token.line,
                column: token.column,
            }),
        }
    }

    fn expect_string(&mut self, msg: &str) -> Result<(String, Span), ParseError> {
        self.fail_on_lexer_error()?;
        let token = self.current().clone();
        match token.kind {
            TokenKind::Str(ref value) => {
                self.advance();
                Ok((value.clone(), token_span(&token)))
            }
            _ => Err(ParseError {
                message: msg.to_string(),
                line: token.line,
                column: token.column,
            }),
        }
    }

    fn check(&self, expected: &TokenKind) -> bool {
        same_kind(&self.current().kind, expected)
    }

    fn check_next(&self, expected: &TokenKind) -> bool {
        let next_index = self.position + 1;
        if next_index >= self.tokens.len() {
            return false;
        }
        same_kind(&self.tokens[next_index].kind, expected)
    }

    fn fail_on_lexer_error(&self) -> Result<(), ParseError> {
        if let TokenKind::Error(msg) = &self.current().kind {
            let normalized = if msg.contains("Unexpected character '('")
                || msg.contains("Unexpected character ')'")
            {
                "Parentheses are not supported in Volta expressions".to_string()
            } else {
                format!("Lexical error: {msg}")
            };
            return Err(ParseError {
                message: normalized,
                line: self.current().line,
                column: self.current().column,
            });
        }
        Ok(())
    }

    fn is_stmt_end(&self) -> bool {
        self.check(&TokenKind::Newline)
            || self.check(&TokenKind::Dedent)
            || self.check(&TokenKind::Eof)
    }

    fn require_stmt_terminator(&mut self) -> Result<(), ParseError> {
        if self.match_kind(&TokenKind::Newline) {
            return Ok(());
        }

        if self.check(&TokenKind::Dedent) || self.check(&TokenKind::Eof) {
            Ok(())
        } else {
            Err(self.error_here("Unexpected token after statement"))
        }
    }

    fn error_here(&self, message: &str) -> ParseError {
        ParseError {
            message: message.to_string(),
            line: self.current().line,
            column: self.current().column,
        }
    }
}

fn token_span(token: &Token) -> Span {
    Span::single(token.line, token.column)
}

fn end_span(primary_end: Span, fallback_end: Span) -> Span {
    if fallback_end.line == 0 {
        primary_end
    } else {
        fallback_end
    }
}

fn same_kind(left: &TokenKind, right: &TokenKind) -> bool {
    std::mem::discriminant(left) == std::mem::discriminant(right)
}

fn is_keyword_token(kind: &TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Model
            | TokenKind::Dataset
            | TokenKind::Train
            | TokenKind::Save
            | TokenKind::Load
            | TokenKind::Print
            | TokenKind::If
            | TokenKind::Elif
            | TokenKind::Else
            | TokenKind::Loop
            | TokenKind::Precision
            | TokenKind::Memory
            | TokenKind::Fn
            | TokenKind::Return
            | TokenKind::On
            | TokenKind::As
            | TokenKind::True
            | TokenKind::False
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse(source: &str) -> Result<Program, ParseError> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        parser.parse_program()
    }

    #[test]
    fn parses_var_decl_and_assign() {
        let program = parse("x 1\nx = x + 1\n").expect("parse failed");
        assert_eq!(program.statements.len(), 2);

        match &program.statements[0] {
            Stmt::VarDecl { name, .. } => assert_eq!(name, "x"),
            _ => panic!("expected vardecl"),
        }
        match &program.statements[1] {
            Stmt::Assign { name, .. } => assert_eq!(name, "x"),
            _ => panic!("expected assign"),
        }
    }

    #[test]
    fn preserves_operator_precedence() {
        let program = parse("x 1 + 2 * 3 > 6\n").expect("parse failed");
        let stmt = &program.statements[0];
        let expr = match stmt {
            Stmt::VarDecl { value, .. } => value,
            _ => panic!("expected vardecl"),
        };

        match expr {
            Expr::Binary {
                op: BinaryOp::Greater,
                left,
                right,
                ..
            } => {
                assert!(matches!(**right, Expr::Int { value: 6, .. }));
                match &**left {
                    Expr::Binary {
                        op: BinaryOp::Add,
                        left: add_left,
                        right: add_right,
                        ..
                    } => {
                        assert!(matches!(**add_left, Expr::Int { value: 1, .. }));
                        assert!(matches!(
                            **add_right,
                            Expr::Binary {
                                op: BinaryOp::Mul,
                                ..
                            }
                        ));
                    }
                    _ => panic!("expected add on left side"),
                }
            }
            _ => panic!("expected comparison at top"),
        }
    }

    #[test]
    fn parses_if_elif_else_and_spans() {
        let src =
            "x 1\nif x > 0\n    print \"a\"\nelif x > -1\n    print \"b\"\nelse\n    print \"c\"\n";
        let program = parse(src).expect("parse failed");
        assert_eq!(program.statements.len(), 2);

        match &program.statements[1] {
            Stmt::If {
                elif_branches,
                else_branch,
                span,
                ..
            } => {
                assert_eq!(elif_branches.len(), 1);
                assert!(else_branch.is_some());
                assert!(span.line > 0);
            }
            _ => panic!("expected if stmt"),
        }
    }

    #[test]
    fn rejects_train_without_on() {
        let err = parse("train m d\n    epochs 1\n    device cpu\n").expect_err("must fail");
        assert!(err.message.contains("expected 'on'"));
    }

    #[test]
    fn rejects_save_without_as() {
        let err = parse("save m \"x.vt\"\n").expect_err("must fail");
        assert!(err.message.contains("expected 'as'"));
    }

    #[test]
    fn rejects_missing_indent_after_block_header() {
        let err = parse("model m\nlayers 1 2\n").expect_err("must fail");
        assert!(err.message.contains("Missing indent"));
    }

    #[test]
    fn parses_function_and_call_expression() {
        let src = "fn double(x)\n    return x * 2\ny double(3)\n";
        let program = parse(src).expect("parse failed");
        assert_eq!(program.statements.len(), 2);
        match &program.statements[0] {
            Stmt::Function { name, params, .. } => {
                assert_eq!(name, "double");
                assert_eq!(params, &vec!["x".to_string()]);
            }
            _ => panic!("expected function declaration"),
        }
        match &program.statements[1] {
            Stmt::VarDecl {
                value: Expr::Call { callee, args, .. },
                ..
            } => {
                assert_eq!(callee, "double");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected call expression in var decl"),
        }
    }

    #[test]
    fn ast_snapshot_var_decl_with_span() {
        let program = parse("lr 0.001\n").expect("parse failed");
        let actual = format!("{:#?}", program);
        let expected = r#"Program {
    statements: [
        VarDecl {
            name: "lr",
            value: Float {
                value: 0.001,
                span: Span {
                    line: 1,
                    column: 4,
                    end_line: 1,
                    end_column: 4,
                },
            },
            span: Span {
                line: 1,
                column: 1,
                end_line: 1,
                end_column: 4,
            },
        },
    ],
}"#;

        assert_eq!(actual, expected);
    }
}
