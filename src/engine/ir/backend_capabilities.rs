#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceClass {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendVendor {
    GenericCpu,
    Nvidia,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendMaturity {
    Experimental,
    Validated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeterminismLevel {
    Strict,
    Balanced,
    Fast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ExecutionPhase {
    #[default]
    Inference,
    Training,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendValidationError {
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub backend: BackendKind,
    pub device_class: DeviceClass,
    pub vendor: BackendVendor,
    pub maturity: BackendMaturity,
    pub supports_inference: bool,
    pub supports_training: bool,
    pub supports_runtime_execution: bool,
    pub supports_gradient_updates: bool,
    pub supports_adam: bool,
    pub supports_strict_determinism: bool,
    pub supports_balanced_determinism: bool,
    pub supports_fast_determinism: bool,
    pub default_determinism: DeterminismLevel,
}

impl BackendCapabilities {
    pub fn supports_phase(self, phase: ExecutionPhase) -> bool {
        match phase {
            ExecutionPhase::Inference => self.supports_inference,
            ExecutionPhase::Training => self.supports_training,
        }
    }

    pub fn supports_determinism(self, determinism: DeterminismLevel) -> bool {
        match determinism {
            DeterminismLevel::Strict => self.supports_strict_determinism,
            DeterminismLevel::Balanced => self.supports_balanced_determinism,
            DeterminismLevel::Fast => self.supports_fast_determinism,
        }
    }

    pub fn validate(
        self,
        phase: ExecutionPhase,
        determinism: DeterminismLevel,
    ) -> Result<(), BackendValidationError> {
        if !self.supports_phase(phase) {
            return Err(BackendValidationError {
                message: format!(
                    "Backend {:?} does not support {:?} execution",
                    self.backend, phase
                ),
            });
        }

        if !self.supports_runtime_execution {
            return Err(BackendValidationError {
                message: format!("Backend {:?} cannot execute plans at runtime", self.backend),
            });
        }

        if phase == ExecutionPhase::Training && !self.supports_gradient_updates {
            return Err(BackendValidationError {
                message: format!(
                    "Backend {:?} cannot apply gradients for training workloads",
                    self.backend
                ),
            });
        }

        if !self.supports_determinism(determinism) {
            return Err(BackendValidationError {
                message: format!(
                    "Backend {:?} does not support {:?} determinism",
                    self.backend, determinism
                ),
            });
        }

        Ok(())
    }

    pub fn validate_optimizer(self, optimizer: &str) -> Result<(), BackendValidationError> {
        let opt = optimizer.to_lowercase();
        if (opt == "adam" || opt == "adamw") && !self.supports_adam {
            return Err(BackendValidationError {
                message: format!(
                    "Backend {:?} does not support Adam optimizer. \
                     Use SGD or a backend/path that explicitly supports Adam.",
                    self.backend
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BackendCapabilities, BackendKind, BackendMaturity, BackendVendor, DeterminismLevel,
        DeviceClass, ExecutionPhase,
    };

    fn cpu_caps() -> BackendCapabilities {
        BackendCapabilities {
            backend: BackendKind::Cpu,
            device_class: DeviceClass::Cpu,
            vendor: BackendVendor::GenericCpu,
            maturity: BackendMaturity::Validated,
            supports_inference: true,
            supports_training: true,
            supports_runtime_execution: true,
            supports_gradient_updates: true,
            supports_strict_determinism: true,
            supports_balanced_determinism: true,
            supports_fast_determinism: true,
            default_determinism: DeterminismLevel::Strict,
            supports_adam: true,
        }
    }

    #[test]
    fn validate_optimizer_sgd_always_ok() {
        let caps = cpu_caps();
        assert!(caps.validate_optimizer("sgd").is_ok());
    }

    #[test]
    fn validate_optimizer_adam_ok_when_supported() {
        let caps = cpu_caps(); // supports_adam: true
        assert!(caps.validate_optimizer("adam").is_ok());
    }

    #[test]
    fn validate_optimizer_adam_err_when_not_supported() {
        let mut caps = cpu_caps();
        caps.supports_adam = false;
        let err = caps
            .validate_optimizer("adam")
            .expect_err("adam must be rejected when supports_adam is false");
        assert!(err.message.contains("Adam"));
    }

    #[test]
    fn validate_optimizer_adamw_err_when_not_supported() {
        let mut caps = cpu_caps();
        caps.supports_adam = false;
        let err = caps
            .validate_optimizer("adamw")
            .expect_err("adamw must be rejected when supports_adam is false");
        assert!(err.message.contains("Adam"));
    }

    #[test]
    fn validate_accepts_supported_phase_and_determinism() {
        let caps = cpu_caps();
        assert!(
            caps.validate(ExecutionPhase::Inference, DeterminismLevel::Strict)
                .is_ok()
        );
        assert!(
            caps.validate(ExecutionPhase::Training, DeterminismLevel::Balanced)
                .is_ok()
        );
    }

    #[test]
    fn validate_rejects_training_without_gradient_updates() {
        let mut caps = cpu_caps();
        caps.supports_gradient_updates = false;
        let err = caps
            .validate(ExecutionPhase::Training, DeterminismLevel::Strict)
            .expect_err("training without gradient support must fail");
        assert!(err.message.contains("apply gradients"));
    }

    #[test]
    fn validate_rejects_unsupported_determinism() {
        let mut caps = cpu_caps();
        caps.supports_strict_determinism = false;
        let err = caps
            .validate(ExecutionPhase::Inference, DeterminismLevel::Strict)
            .expect_err("strict determinism must be rejected");
        assert!(err.message.contains("Strict determinism"));
    }
}
