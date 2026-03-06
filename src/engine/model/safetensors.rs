/// SafeTensors native loader for Volta.
///
/// SafeTensors format:
///   - 8 bytes: metadata length N (little-endian u64)
///   - N bytes: UTF-8 JSON metadata
///   - tensor data (concatenated, aligned)
///
/// JSON metadata structure:
///   { "__metadata__": {...}, "tensor_name": { "dtype": "F32", "shape": [...], "data_offsets": [start, end] } }
use std::collections::HashMap;
use std::fs;

use crate::ir::Tensor;
use crate::model::TrainApiError;

/// Load tensors from a SafeTensors file.
/// Returns a map from tensor name to Tensor (F32 only; other dtypes are converted if possible).
pub fn load_safetensors(path: &str) -> Result<HashMap<String, Tensor>, TrainApiError> {
    let raw = fs::read(path).map_err(|e| TrainApiError {
        message: format!("Failed to read SafeTensors file '{path}': {e}"),
    })?;

    if raw.len() < 8 {
        return Err(TrainApiError {
            message: format!("SafeTensors file '{path}' is too small"),
        });
    }

    // Read header length
    let header_len = u64::from_le_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]) as usize;

    if 8 + header_len > raw.len() {
        return Err(TrainApiError {
            message: format!("SafeTensors header length {header_len} exceeds file size"),
        });
    }

    let header_bytes = &raw[8..8 + header_len];
    let header_str = std::str::from_utf8(header_bytes).map_err(|_| TrainApiError {
        message: "SafeTensors header is not valid UTF-8".to_string(),
    })?;

    // Parse JSON header with a minimal hand-rolled parser
    let data_start = 8 + header_len;
    let tensor_data = &raw[data_start..];

    parse_safetensors_header(header_str, tensor_data)
}

fn parse_safetensors_header(
    header: &str,
    tensor_data: &[u8],
) -> Result<HashMap<String, Tensor>, TrainApiError> {
    // Use serde_json for header parsing
    let json: serde_json::Value = serde_json::from_str(header).map_err(|e| TrainApiError {
        message: format!("Failed to parse SafeTensors JSON header: {e}"),
    })?;

    let obj = json.as_object().ok_or_else(|| TrainApiError {
        message: "SafeTensors header is not a JSON object".to_string(),
    })?;

    let mut tensors = HashMap::new();

    for (name, info) in obj {
        // Skip metadata key
        if name == "__metadata__" {
            continue;
        }

        let info_obj = info.as_object().ok_or_else(|| TrainApiError {
            message: format!("Tensor entry '{name}' is not a JSON object"),
        })?;

        let dtype = info_obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TrainApiError {
                message: format!("Missing 'dtype' for tensor '{name}'"),
            })?;

        let shape_arr = info_obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TrainApiError {
                message: format!("Missing 'shape' for tensor '{name}'"),
            })?;

        let shape: Vec<usize> = shape_arr
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| TrainApiError {
                        message: format!("Invalid shape element for tensor '{name}'"),
                    })
                    .map(|x| x as usize)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let offsets = info_obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TrainApiError {
                message: format!("Missing 'data_offsets' for tensor '{name}'"),
            })?;

        if offsets.len() != 2 {
            return Err(TrainApiError {
                message: format!("'data_offsets' for tensor '{name}' must have exactly 2 elements"),
            });
        }

        let start = offsets[0].as_u64().ok_or_else(|| TrainApiError {
            message: format!("Invalid data_offsets[0] for tensor '{name}'"),
        })? as usize;
        let end = offsets[1].as_u64().ok_or_else(|| TrainApiError {
            message: format!("Invalid data_offsets[1] for tensor '{name}'"),
        })? as usize;

        if end > tensor_data.len() || start > end {
            return Err(TrainApiError {
                message: format!("data_offsets [{start}, {end}] out of bounds for tensor '{name}'"),
            });
        }

        let bytes = &tensor_data[start..end];
        let data = bytes_to_f32(dtype, bytes, name)?;

        let tensor = Tensor::new(shape, data).map_err(|e| TrainApiError {
            message: format!("Invalid tensor '{name}': {}", e.message),
        })?;

        tensors.insert(name.clone(), tensor);
    }

    Ok(tensors)
}

fn bytes_to_f32(dtype: &str, bytes: &[u8], name: &str) -> Result<Vec<f32>, TrainApiError> {
    match dtype {
        "F32" => {
            if bytes.len() % 4 != 0 {
                return Err(TrainApiError {
                    message: format!(
                        "F32 tensor '{name}' byte length {} not divisible by 4",
                        bytes.len()
                    ),
                });
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        "F16" => {
            if bytes.len() % 2 != 0 {
                return Err(TrainApiError {
                    message: format!(
                        "F16 tensor '{name}' byte length {} not divisible by 2",
                        bytes.len()
                    ),
                });
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect())
        }
        "BF16" => {
            if bytes.len() % 2 != 0 {
                return Err(TrainApiError {
                    message: format!(
                        "BF16 tensor '{name}' byte length {} not divisible by 2",
                        bytes.len()
                    ),
                });
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect())
        }
        "F64" => {
            if bytes.len() % 8 != 0 {
                return Err(TrainApiError {
                    message: format!(
                        "F64 tensor '{name}' byte length {} not divisible by 8",
                        bytes.len()
                    ),
                });
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect())
        }
        "I32" => {
            if bytes.len() % 4 != 0 {
                return Err(TrainApiError {
                    message: format!(
                        "I32 tensor '{name}' byte length {} not divisible by 4",
                        bytes.len()
                    ),
                });
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
                .collect())
        }
        "I64" => {
            if bytes.len() % 8 != 0 {
                return Err(TrainApiError {
                    message: format!(
                        "I64 tensor '{name}' byte length {} not divisible by 8",
                        bytes.len()
                    ),
                });
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|b| {
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect())
        }
        other => Err(TrainApiError {
            message: format!("Unsupported SafeTensors dtype '{other}' for tensor '{name}'"),
        }),
    }
}

/// Convert IEEE 754 half-precision float to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: normalize
        let mut m = mant;
        let mut e = 0u32;
        while m & 0x400 == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = (127 - 15 - e + 1) << 23;
        f32::from_bits(sign | exp32 | ((m & 0x3FF) << 13))
    } else if exp == 0x1F {
        // Inf or NaN
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        let exp32 = (exp + 127 - 15) << 23;
        f32::from_bits(sign | exp32 | (mant << 13))
    }
}

/// Convert bfloat16 to f32 (same exponent width, just extend mantissa with zeros).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Save tensors to SafeTensors format (F32 only).
pub fn save_safetensors(
    path: &str,
    tensors: &HashMap<String, Tensor>,
) -> Result<(), TrainApiError> {
    // Build metadata JSON and pack tensor data
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    let mut tensor_data: Vec<u8> = Vec::new();
    let mut meta_entries: Vec<String> = Vec::new();

    for name in &names {
        let tensor = &tensors[*name];
        let start = tensor_data.len();
        for &v in tensor.data.iter() {
            tensor_data.extend_from_slice(&v.to_le_bytes());
        }
        let end = tensor_data.len();

        let shape_str = tensor
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");

        // Escape name for JSON
        let escaped_name = name.replace('\\', "\\\\").replace('"', "\\\"");
        meta_entries.push(format!(
            r#""{escaped_name}":{{"dtype":"F32","shape":[{shape_str}],"data_offsets":[{start},{end}]}}"#
        ));
    }

    let header_json = format!("{{{}}}", meta_entries.join(","));
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&tensor_data);

    fs::write(path, &out).map_err(|e| TrainApiError {
        message: format!("Failed to write SafeTensors file '{path}': {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Tensor;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp(label: &str) -> String {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("volta-st-{label}-{nonce}.safetensors"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn safetensors_roundtrip_f32() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::new(vec![3], vec![0.1, 0.2, 0.3]).unwrap(),
        );

        let path = tmp("roundtrip");
        save_safetensors(&path, &tensors).expect("save");
        let loaded = load_safetensors(&path).expect("load");

        assert_eq!(loaded.len(), 2);
        let w = &loaded["weight"];
        assert_eq!(w.shape, vec![2, 3]);
        for (a, b) in w.data.iter().zip([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn f16_to_f32_converts_known_values() {
        // 1.0 in F16 = 0x3C00
        let v = super::f16_to_f32(0x3C00);
        assert!((v - 1.0f32).abs() < 1e-4, "f16 1.0 -> f32: got {v}");

        // -2.0 in F16 = 0xC000
        let v2 = super::f16_to_f32(0xC000);
        assert!((v2 - (-2.0f32)).abs() < 1e-4, "f16 -2.0 -> f32: got {v2}");
    }

    #[test]
    fn bf16_to_f32_converts_known_values() {
        // 1.0 in BF16 = 0x3F80
        let v = super::bf16_to_f32(0x3F80);
        assert!((v - 1.0f32).abs() < 1e-6, "bf16 1.0 -> f32: got {v}");
    }
}
