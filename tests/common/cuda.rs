// Спільні утиліти для CUDA інтеграційних тестів.
// Підключається через: `#[path = "common/cuda.rs"] mod cuda_helpers;`
#![allow(dead_code)]
use std::sync::{Mutex, OnceLock};

/// Перевіряє чи доступний CUDA пристрій на цій машині.
/// При відсутності GPU повертає false замість паніки.
pub fn cuda_runtime_available() -> bool {
    let result = std::panic::catch_unwind(|| volta::ir::cuda::device::CudaDevice::new(0));
    matches!(result, Ok(Ok(_)))
}

/// Намагається ініціалізувати CUDA device id=0. Повертає `Some(device)` якщо GPU доступний, `None` інакше.
pub fn safe_cuda_device() -> Option<volta::ir::cuda::device::CudaDevice> {
    let result = std::panic::catch_unwind(|| volta::ir::cuda::device::CudaDevice::new(0));
    match result {
        Ok(Ok(device)) => Some(device),
        _ => None,
    }
}

/// Перевіряє чи рядок помилки відповідає недоступному CUDA пристрою.
pub fn is_cuda_unavailable(message: &str) -> bool {
    message.contains("CUDA device init failed")
        || message.contains("runtime handles")
        || message.contains("NVRTC")
        || message.contains("cuBLAS init failed")
        || message.contains("CUDA context init failed")
}

/// Встановлює `VOLTA_DETERMINISM` env var, виконує closure, потім відновлює оригінальне значення.
/// Серіалізує доступ через глобальний mutex щоб env var зміни не конфліктували між паралельними тестами.
pub fn with_determinism(level: &str, run: impl FnOnce()) {
    let _guard = match env_lock().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let _restore = EnvVarRestore::set("VOLTA_DETERMINISM", level);
    run();
}

/// Варіант `with_determinism` що повертає значення.
pub fn with_determinism_ret<T>(level: &str, run: impl FnOnce() -> T) -> T {
    let _guard = match env_lock().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let _restore = EnvVarRestore::set("VOLTA_DETERMINISM", level);
    run()
}

/// Повертає глобальний mutex для серіалізації env var операцій у тестах.
pub fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// RAII guard що відновлює env var при drop.
pub struct EnvVarRestore {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarRestore {
    pub fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: тести серіалізовані через env_lock mutex,
        // тому мутація env var є безпечною в рамках одного тесту.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => unsafe { std::env::set_var(self.key, value) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}
