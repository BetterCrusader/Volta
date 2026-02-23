//! # Volta
//!
//! A statically-typed, interpreted language for declaring and training
//! neural network models with automatic hyperparameter selection.
//!
//! ## Pipeline
//!
//! ```text
//! Source text
//!    │
//!    ▼  lexer::Lexer
//! Token stream
//!    │
//!    ▼  parser::Parser
//! ast::Program
//!    │
//!    ▼  semantic::SemanticAnalyzer
//! Validated AST + warnings
//!    │
//!    ▼  executor::Executor
//! Side effects (training, save/load, print)
//! ```
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use volta::executor::Executor;
//! use volta::lexer::Lexer;
//! use volta::parser::Parser;
//! use volta::semantic::SemanticAnalyzer;
//!
//! let src = "x = 1 + 2\nprint x";
//! let tokens = Lexer::new(src).tokenize();
//! let mut parser = Parser::new(tokens);
//! let program = parser.parse_program().unwrap();
//! SemanticAnalyzer::new().analyze(&program).unwrap();
//! Executor::new().execute(&program).unwrap();
//! ```

#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::unused_self,
    clippy::uninlined_format_args,
    clippy::too_many_lines,
    clippy::match_same_arms,
    clippy::manual_let_else,
    clippy::needless_pass_by_value,
    clippy::implicit_hasher,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::unnecessary_wraps,
    clippy::missing_panics_doc,
    clippy::needless_continue,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::struct_excessive_bools,
    clippy::items_after_statements
)]

pub mod ast;
pub mod autopilot;
pub mod diagnostics;
pub mod executor;
pub mod interop;
pub mod ir;
pub mod lexer;
pub mod model;
pub mod parser;
pub mod rules;
pub mod semantic;
