use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::CudaBlas;
use cudarc::driver::result as driver_result;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream};
use cudarc::nvrtc::CompileOptions;
use cudarc::nvrtc::compile_ptx_with_opts;

#[derive(Debug)]
pub struct CudaDeviceError {
    pub message: String,
}

#[derive(Debug)]
pub struct CudaRuntimeHandles {
    pub context: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub cublas: Mutex<Option<CudaBlas>>,
    pub kernel_functions: Mutex<HashMap<String, CudaFunction>>,
}

#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub device_id: usize,
    pub name: String,
    pub compute_capability_major: u8,
    pub compute_capability_minor: u8,
    pub total_memory_bytes: usize,
    runtime: Option<Arc<CudaRuntimeHandles>>,
}

impl CudaDevice {
    pub fn new(device_id: usize) -> Result<Self, CudaDeviceError> {
        let context = CudaContext::new(device_id).map_err(|err| CudaDeviceError {
            message: format!("CUDA context init failed for device {device_id}: {err}"),
        })?;
        let stream = context.new_stream().map_err(|err| CudaDeviceError {
            message: format!("CUDA stream creation failed for device {device_id}: {err}"),
        })?;

        let name = context.name().map_err(|err| CudaDeviceError {
            message: format!("CUDA device name query failed for device {device_id}: {err}"),
        })?;
        let (major, minor) = context
            .compute_capability()
            .map_err(|err| CudaDeviceError {
                message: format!(
                    "CUDA compute capability query failed for device {device_id}: {err}"
                ),
            })?;
        let (_, total_memory_bytes) =
            driver_result::mem_get_info().map_err(|err| CudaDeviceError {
                message: format!("CUDA memory query failed for device {device_id}: {err}"),
            })?;

        let compute_capability_major = u8::try_from(major).map_err(|_| CudaDeviceError {
            message: format!("CUDA device {device_id} reported invalid major capability: {major}"),
        })?;
        let compute_capability_minor = u8::try_from(minor).map_err(|_| CudaDeviceError {
            message: format!("CUDA device {device_id} reported invalid minor capability: {minor}"),
        })?;

        Ok(Self {
            device_id,
            name,
            compute_capability_major,
            compute_capability_minor,
            total_memory_bytes,
            runtime: Some(Arc::new(CudaRuntimeHandles {
                context,
                stream,
                cublas: Mutex::new(None),
                kernel_functions: Mutex::new(HashMap::new()),
            })),
        })
    }

    pub fn has_runtime_handles(&self) -> bool {
        self.runtime.is_some()
    }

    pub fn context(&self) -> Result<Arc<CudaContext>, CudaDeviceError> {
        let runtime = self.runtime()?;
        Ok(runtime.context.clone())
    }

    pub fn stream(&self) -> Result<Arc<CudaStream>, CudaDeviceError> {
        let runtime = self.runtime()?;
        Ok(runtime.stream.clone())
    }

    pub fn with_cublas<R>(
        &self,
        run: impl FnOnce(&mut CudaBlas) -> Result<R, CudaDeviceError>,
    ) -> Result<R, CudaDeviceError> {
        let runtime = self.runtime()?;
        let mut guard = match runtime.cublas.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if guard.is_none() {
            let stream = runtime.stream.clone();
            let cublas = catch_unwind_silent(|| CudaBlas::new(stream))
                .map_err(|_| CudaDeviceError {
                    message: format!(
                        "cuBLAS init panicked for device {}; verify CUDA libraries are installed",
                        self.device_id
                    ),
                })?
                .map_err(|err| CudaDeviceError {
                    message: format!("cuBLAS init failed for device {}: {err}", self.device_id),
                })?;
            *guard = Some(cublas);
        }

        let Some(cublas) = guard.as_mut() else {
            return Err(CudaDeviceError {
                message: "cuBLAS handle became unavailable".to_string(),
            });
        };
        run(cublas)
    }

    fn runtime(&self) -> Result<&Arc<CudaRuntimeHandles>, CudaDeviceError> {
        self.runtime.as_ref().ok_or_else(|| CudaDeviceError {
            message: "CUDA runtime handles are unavailable on this device instance".to_string(),
        })
    }

    pub fn load_or_get_function(
        &self,
        module_key: &str,
        function_name: &str,
        cuda_source: &str,
        compile_options: CompileOptions,
    ) -> Result<CudaFunction, CudaDeviceError> {
        let runtime = self.runtime()?;
        let compile_key = compile_options_fingerprint(&compile_options);
        let key = format!(
            "{}::{}::sm{}{}::opts{:016x}",
            module_key,
            function_name,
            self.compute_capability_major,
            self.compute_capability_minor,
            compile_key
        );

        {
            let cache = match runtime.kernel_functions.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            if let Some(function) = cache.get(&key).cloned() {
                return Ok(function);
            }
        }

        let ptx =
            compile_ptx_with_opts(cuda_source, compile_options).map_err(|err| CudaDeviceError {
                message: format!("NVRTC compile failed for module '{module_key}': {err}"),
            })?;
        let module = runtime
            .context
            .load_module(ptx)
            .map_err(|err| CudaDeviceError {
                message: format!("CUDA module load failed for '{module_key}': {err}"),
            })?;
        let function = module
            .load_function(function_name)
            .map_err(|err| CudaDeviceError {
                message: format!(
                    "CUDA function load failed for '{}' in module '{}': {}",
                    function_name, module_key, err
                ),
            })?;

        let mut cache = match runtime.kernel_functions.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        cache.insert(key, function.clone());
        Ok(function)
    }

    fn scaffold() -> Self {
        Self {
            device_id: 0,
            name: "cuda-scaffold".to_string(),
            compute_capability_major: 0,
            compute_capability_minor: 0,
            total_memory_bytes: 0,
            runtime: None,
        }
    }

    pub fn default_shared() -> Result<Self, CudaDeviceError> {
        static DEVICE: OnceLock<Result<CudaDevice, String>> = OnceLock::new();
        match DEVICE.get_or_init(|| Self::new(0).map_err(|err| err.message)) {
            Ok(device) => Ok(device.clone()),
            Err(message) => Err(CudaDeviceError {
                message: message.clone(),
            }),
        }
    }
}

fn catch_unwind_silent<T>(
    run: impl FnOnce() -> T + std::panic::UnwindSafe,
) -> Result<T, Box<dyn std::any::Any + Send + 'static>> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(run);
    std::panic::set_hook(previous);
    result
}

fn compile_options_fingerprint(options: &CompileOptions) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    options.hash(&mut hasher);
    hasher.finish()
}

impl Default for CudaDevice {
    fn default() -> Self {
        Self::new(0).unwrap_or_else(|_| Self::scaffold())
    }
}

#[cfg(test)]
mod tests {
    use super::compile_options_fingerprint;
    use cudarc::nvrtc::CompileOptions;

    #[test]
    fn compile_options_fingerprint_changes_with_determinism_flags() {
        let strict = CompileOptions {
            use_fast_math: Some(false),
            options: vec![
                "--gpu-architecture=compute_89".to_string(),
                "--fmad=false".to_string(),
            ],
            ..Default::default()
        };
        let balanced = CompileOptions {
            use_fast_math: Some(false),
            options: vec!["--gpu-architecture=compute_89".to_string()],
            ..Default::default()
        };
        assert_ne!(
            compile_options_fingerprint(&strict),
            compile_options_fingerprint(&balanced)
        );
    }
}
