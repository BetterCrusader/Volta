use crate::ir::DeterminismLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompilerFlags {
    pub strict: bool,
    pub debug_verify: bool,
    pub unsafe_opt: bool,
    pub determinism: DeterminismLevel,
}

impl CompilerFlags {
    pub fn from_env() -> Self {
        Self {
            strict: read_bool("VOLTA_STRICT", true),
            debug_verify: read_bool("VOLTA_DEBUG_VERIFY", cfg!(debug_assertions)),
            unsafe_opt: read_bool("VOLTA_UNSAFE_OPT", false),
            determinism: read_determinism("VOLTA_DETERMINISM", DeterminismLevel::Balanced),
        }
    }
}

fn read_bool(key: &str, default_value: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(default_value)
}

fn read_determinism(key: &str, default_value: DeterminismLevel) -> DeterminismLevel {
    let Ok(value) = std::env::var(key) else {
        return default_value;
    };

    if value.eq_ignore_ascii_case("strict") {
        DeterminismLevel::Strict
    } else if value.eq_ignore_ascii_case("fast") {
        DeterminismLevel::Fast
    } else if value.eq_ignore_ascii_case("balanced") {
        DeterminismLevel::Balanced
    } else {
        default_value
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{CompilerFlags, DeterminismLevel};

    #[test]
    fn flags_are_readable_from_environment_defaults() {
        let flags = CompilerFlags::from_env();
        if cfg!(debug_assertions) {
            assert!(flags.debug_verify);
        }
        assert_eq!(flags.determinism, DeterminismLevel::Balanced);
    }
}
