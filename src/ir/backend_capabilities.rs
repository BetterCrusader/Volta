#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeterminismLevel {
    Strict,
    Balanced,
    Fast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub backend: BackendKind,
    pub supports_inference: bool,
    pub supports_training: bool,
    pub supports_strict_determinism: bool,
    pub default_determinism: DeterminismLevel,
}
