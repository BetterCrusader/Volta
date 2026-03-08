#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CpuTargetMode {
    #[default]
    Portable,
    Native,
}

impl CpuTargetMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            Self::Native => "native",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("portable") {
            Some(Self::Portable)
        } else if value.eq_ignore_ascii_case("native") {
            Some(Self::Native)
        } else {
            None
        }
    }

    pub fn rustc_target_cpu_flag(self) -> Option<&'static str> {
        match self {
            Self::Portable => None,
            Self::Native => Some("-C target-cpu=native"),
        }
    }

    pub fn clang_codegen_args(self) -> &'static [&'static str] {
        match self {
            Self::Portable => &["-O3", "-ffast-math", "-funroll-loops"],
            Self::Native => &["-O3", "-march=native", "-ffast-math", "-funroll-loops"],
        }
    }

    pub fn portable_llvm_cpu(self, arch: &str) -> &'static str {
        match self {
            Self::Portable => match arch {
                "x86_64" => "x86-64",
                "aarch64" => "generic",
                _ => "generic",
            },
            Self::Native => "native",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuSupportTier {
    Tier1,
    BestEffort,
}

impl CpuSupportTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tier1 => "tier1",
            Self::BestEffort => "best-effort",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostCpuCapabilities {
    pub arch: String,
    pub support_tier: CpuSupportTier,
    pub isa: Vec<String>,
    pub target_mode: CpuTargetMode,
}

pub fn detect_host_cpu_capabilities(target_mode: CpuTargetMode) -> HostCpuCapabilities {
    let arch = std::env::consts::ARCH.to_string();
    HostCpuCapabilities {
        support_tier: support_tier_for_arch(&arch),
        isa: detect_host_isa(),
        arch,
        target_mode,
    }
}

pub fn support_tier_for_arch(arch: &str) -> CpuSupportTier {
    match arch {
        "x86_64" | "aarch64" => CpuSupportTier::Tier1,
        _ => CpuSupportTier::BestEffort,
    }
}

fn detect_host_isa() -> Vec<String> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut isa = vec!["x86_64".to_string()];
        if std::arch::is_x86_feature_detected!("sse2") {
            isa.push("sse2".to_string());
        }
        if std::arch::is_x86_feature_detected!("fma") {
            isa.push("fma".to_string());
        }
        if std::arch::is_x86_feature_detected!("avx") {
            isa.push("avx".to_string());
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            isa.push("avx2".to_string());
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            isa.push("avx512f".to_string());
        }
        isa
    }

    #[cfg(target_arch = "aarch64")]
    {
        let mut isa = vec!["arm64".to_string()];
        if std::arch::is_aarch64_feature_detected!("neon") {
            isa.push("neon".to_string());
        }
        if std::arch::is_aarch64_feature_detected!("fp16") {
            isa.push("fp16".to_string());
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            isa.push("dotprod".to_string());
        }
        isa
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        vec![std::env::consts::ARCH.to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::{CpuSupportTier, CpuTargetMode, support_tier_for_arch};

    #[test]
    fn parse_cpu_target_mode_accepts_known_values() {
        assert_eq!(
            CpuTargetMode::parse("portable"),
            Some(CpuTargetMode::Portable)
        );
        assert_eq!(CpuTargetMode::parse("native"), Some(CpuTargetMode::Native));
        assert_eq!(CpuTargetMode::parse("NATIVE"), Some(CpuTargetMode::Native));
    }

    #[test]
    fn parse_cpu_target_mode_rejects_unknown_values() {
        assert_eq!(CpuTargetMode::parse("fast"), None);
        assert_eq!(CpuTargetMode::parse(""), None);
    }

    #[test]
    fn support_tier_only_elevates_x86_64_and_aarch64() {
        assert_eq!(support_tier_for_arch("x86_64"), CpuSupportTier::Tier1);
        assert_eq!(support_tier_for_arch("aarch64"), CpuSupportTier::Tier1);
        assert_eq!(support_tier_for_arch("x86"), CpuSupportTier::BestEffort);
    }

    #[test]
    fn portable_mode_has_no_target_cpu_flag() {
        assert_eq!(CpuTargetMode::Portable.rustc_target_cpu_flag(), None);
        assert_eq!(
            CpuTargetMode::Native.rustc_target_cpu_flag(),
            Some("-C target-cpu=native")
        );
    }
}
