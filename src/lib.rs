#![deny(unsafe_code)]
#![allow(
    clippy::collapsible_if,
    clippy::identity_op,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::manual_range_patterns,
    clippy::manual_rotate,
    clippy::needless_range_loop,
    clippy::redundant_closure,
    clippy::same_item_push,
    clippy::should_implement_trait,
    clippy::single_char_add_str,
    clippy::too_many_arguments,
    clippy::unnecessary_cast,
    clippy::useless_format,
    clippy::useless_vec
)]

pub mod engine;
pub mod frontend;
pub mod utils;

pub use engine::{autopilot, executor, ir, model, rules};
pub use frontend::{ast, lexer, parser, semantic};
pub use utils::{diagnostics, interop};
