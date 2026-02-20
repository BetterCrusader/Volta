use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::ir::{Tensor, TensorError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulTuningParams {
    pub k_block: usize,
    pub m_block: usize,
}

#[derive(Debug, Clone)]
pub struct MatmulTuningResult {
    pub best_params: MatmulTuningParams,
    pub benchmark_results: Vec<(MatmulTuningParams, f64)>,
    pub matrix_dim: usize,
}

const K_BLOCK_CANDIDATES: [usize; 6] = [64, 96, 128, 160, 192, 256];
const M_BLOCK_CANDIDATES: [usize; 5] = [32, 48, 64, 96, 128];
const WARMUP_RUNS: usize = 2;
const CACHE_ENV_PATH: &str = "VOLTA_MATMUL_CACHE_PATH";
const CACHE_FILE_NAME: &str = ".volta_matmul_tuning_cache";

static TUNING_CACHE: OnceLock<Mutex<HashMap<usize, MatmulTuningParams>>> = OnceLock::new();
static ENV_OVERRIDE: OnceLock<Option<MatmulTuningParams>> = OnceLock::new();

pub fn default_cpu_tuning() -> MatmulTuningParams {
    MatmulTuningParams {
        k_block: 256,
        m_block: 48,
    }
}

pub fn tune_matmul_for_dim(dim: usize, runs: usize) -> Result<MatmulTuningResult, TensorError> {
    if dim == 0 {
        return Err(TensorError {
            message: "tune_matmul_for_dim expects dim > 0".to_string(),
        });
    }
    if runs == 0 {
        return Err(TensorError {
            message: "tune_matmul_for_dim expects runs > 0".to_string(),
        });
    }

    let left = Tensor::rand(vec![dim, dim]);
    let right = Tensor::rand(vec![dim, dim]);

    let mut results = Vec::new();
    for k_block in K_BLOCK_CANDIDATES {
        for m_block in M_BLOCK_CANDIDATES {
            if k_block > dim || m_block > dim {
                continue;
            }

            let tuning = MatmulTuningParams { k_block, m_block };
            for _ in 0..WARMUP_RUNS {
                let _ = left.matmul_with_tuning(&right, &tuning)?;
            }

            let start = Instant::now();
            for _ in 0..runs {
                let _ = left.matmul_with_tuning(&right, &tuning)?;
            }
            let elapsed = start.elapsed().as_secs_f64();
            let ops_per_sec = if elapsed > 0.0 {
                runs as f64 / elapsed
            } else {
                0.0
            };
            results.push((tuning, ops_per_sec));
        }
    }

    if results.is_empty() {
        let fallback = MatmulTuningParams {
            k_block: dim,
            m_block: dim.min(32),
        };
        for _ in 0..WARMUP_RUNS {
            let _ = left.matmul_with_tuning(&right, &fallback)?;
        }
        let start = Instant::now();
        for _ in 0..runs {
            let _ = left.matmul_with_tuning(&right, &fallback)?;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let ops_per_sec = if elapsed > 0.0 {
            runs as f64 / elapsed
        } else {
            0.0
        };
        results.push((fallback, ops_per_sec));
    }

    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    let best = results[0].0;
    store_tuned_params(dim, best)?;

    Ok(MatmulTuningResult {
        best_params: best,
        benchmark_results: results,
        matrix_dim: dim,
    })
}

pub fn resolve_tuning_for_matmul_shape(lhs: &[usize], rhs: &[usize]) -> MatmulTuningParams {
    if let Some(overridden) = env_override_tuning() {
        return overridden;
    }

    if lhs.len() == 2
        && rhs.len() == 2
        && lhs[0] == lhs[1]
        && lhs[1] == rhs[0]
        && rhs[0] == rhs[1]
        && let Some(cached) = load_tuned_params(lhs[0])
    {
        return cached;
    }

    default_cpu_tuning()
}

pub fn store_tuned_params(dim: usize, params: MatmulTuningParams) -> Result<(), TensorError> {
    let cache = tuning_cache();
    let mut guard = cache.lock().map_err(|_| TensorError {
        message: "matmul tuning cache lock poisoned".to_string(),
    })?;
    guard.insert(dim, params);
    persist_cache_to_path(cache_file_path().as_path(), &guard)?;
    Ok(())
}

pub fn load_tuned_params(dim: usize) -> Option<MatmulTuningParams> {
    let cache = tuning_cache();
    let guard = cache.lock().ok()?;
    guard.get(&dim).copied()
}

fn tuning_cache() -> &'static Mutex<HashMap<usize, MatmulTuningParams>> {
    TUNING_CACHE.get_or_init(|| Mutex::new(load_cache_from_path(cache_file_path().as_path())))
}

fn cache_file_path() -> PathBuf {
    match std::env::var(CACHE_ENV_PATH) {
        Ok(path) if !path.trim().is_empty() => PathBuf::from(path),
        _ => PathBuf::from(CACHE_FILE_NAME),
    }
}

fn load_cache_from_path(path: &Path) -> HashMap<usize, MatmulTuningParams> {
    let mut map = HashMap::new();
    let content = match fs::read_to_string(path) {
        Ok(value) => value,
        Err(_) => return map,
    };

    for line in content.lines() {
        let parts = line.split('|').collect::<Vec<_>>();
        if parts.len() != 3 {
            continue;
        }
        let dim = match parts[0].parse::<usize>() {
            Ok(value) if value > 0 => value,
            _ => continue,
        };
        let k_block = match parts[1].parse::<usize>() {
            Ok(value) if value > 0 => value,
            _ => continue,
        };
        let m_block = match parts[2].parse::<usize>() {
            Ok(value) if value > 0 => value,
            _ => continue,
        };
        map.insert(dim, MatmulTuningParams { k_block, m_block });
    }

    map
}

fn persist_cache_to_path(
    path: &Path,
    cache: &HashMap<usize, MatmulTuningParams>,
) -> Result<(), TensorError> {
    let mut entries = cache.iter().collect::<Vec<_>>();
    entries.sort_by_key(|(dim, _)| *dim);

    let mut lines = Vec::with_capacity(entries.len());
    for (dim, params) in entries {
        lines.push(format!("{dim}|{}|{}", params.k_block, params.m_block));
    }

    fs::write(path, lines.join("\n")).map_err(|err| TensorError {
        message: format!(
            "failed to persist matmul tuning cache to '{}': {err}",
            path.display()
        ),
    })
}

fn env_override_tuning() -> Option<MatmulTuningParams> {
    *ENV_OVERRIDE.get_or_init(|| {
        let k = std::env::var("VOLTA_MATMUL_K_BLOCK").ok();
        let m = std::env::var("VOLTA_MATMUL_M_BLOCK").ok();
        match (k, m) {
            (Some(kv), Some(mv)) => {
                let k_block = kv.parse::<usize>().ok()?;
                let m_block = mv.parse::<usize>().ok()?;
                if k_block == 0 || m_block == 0 {
                    return None;
                }
                Some(MatmulTuningParams { k_block, m_block })
            }
            _ => None,
        }
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::{
        MatmulTuningParams, load_cache_from_path, persist_cache_to_path, tune_matmul_for_dim,
    };

    #[test]
    fn tuning_returns_valid_params() {
        let result = tune_matmul_for_dim(64, 1).expect("tune should pass");
        assert!(result.best_params.k_block <= 64);
        assert!(result.best_params.m_block <= 64);
    }

    #[test]
    fn tuning_does_not_panic_small_dim() {
        let result = tune_matmul_for_dim(32, 1).expect("tune should pass");
        assert_eq!(result.matrix_dim, 32);
    }

    #[test]
    fn best_param_in_result_list() {
        let result = tune_matmul_for_dim(64, 1).expect("tune should pass");
        let contains_best = result
            .benchmark_results
            .iter()
            .any(|(params, _)| params == &result.best_params);
        assert!(contains_best);
    }

    #[test]
    fn params_are_comparable() {
        let a = MatmulTuningParams {
            k_block: 64,
            m_block: 32,
        };
        let b = MatmulTuningParams {
            k_block: 64,
            m_block: 32,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn cache_parse_ignores_invalid_lines() {
        let path = "C:\\Users\\User\\Desktop\\Volta\\target\\matmul_cache_parse_test.txt";
        let content = "128|256|48\nbad\n64|0|32\n32|64|16\n";
        fs::write(path, content).expect("write should pass");
        let parsed = load_cache_from_path(Path::new(path));
        assert_eq!(parsed.len(), 2);
        assert_eq!(
            parsed.get(&128),
            Some(&MatmulTuningParams {
                k_block: 256,
                m_block: 48
            })
        );
        assert_eq!(
            parsed.get(&32),
            Some(&MatmulTuningParams {
                k_block: 64,
                m_block: 16
            })
        );
    }

    #[test]
    fn persist_writes_expected_cache_format() {
        let path = "C:\\Users\\User\\Desktop\\Volta\\target\\matmul_cache_persist_test.txt";
        let mut map = std::collections::HashMap::new();
        map.insert(
            96,
            MatmulTuningParams {
                k_block: 128,
                m_block: 48,
            },
        );
        persist_cache_to_path(Path::new(path), &map).expect("persist should pass");
        let saved = fs::read_to_string(path).expect("read should pass");
        assert_eq!(saved, "96|128|48");
    }
}
