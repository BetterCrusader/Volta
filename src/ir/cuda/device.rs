#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub name: String,
    pub compute_capability_major: u8,
    pub compute_capability_minor: u8,
}

impl Default for CudaDevice {
    fn default() -> Self {
        Self {
            name: "cuda-scaffold".to_string(),
            compute_capability_major: 0,
            compute_capability_minor: 0,
        }
    }
}
