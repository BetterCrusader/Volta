/// GGUF metadata extraction for Volta.
///
/// Reads architecture metadata and tensor descriptors from .gguf files,
/// returning structured information suitable for model loading and analysis.
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use crate::model::TrainApiError;

/// Top-level information extracted from a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufInfo {
    /// GGUF format version (1, 2, or 3).
    pub version: u32,
    /// Architecture name (e.g. "llama", "mistral", "gpt2").
    pub architecture: String,
    /// All key-value metadata from the GGUF header.
    pub metadata: HashMap<String, GgufMetaValue>,
    /// Tensor descriptors (name, shape, dtype).
    pub tensors: Vec<GgufTensorInfo>,
    /// Total number of parameters (element count sum across all tensors).
    pub total_params: u64,
}

impl GgufInfo {
    /// Get a string metadata value.
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufMetaValue::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get an integer metadata value.
    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(GgufMetaValue::U64(v)) => Some(*v),
            Some(GgufMetaValue::U32(v)) => Some(*v as u64),
            Some(GgufMetaValue::I32(v)) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Number of transformer layers / blocks.
    pub fn num_layers(&self) -> Option<u64> {
        self.meta_u64(&format!("{}.block_count", self.architecture))
            .or_else(|| self.metadata.iter()
                .find(|(k, _)| k.contains("block_count"))
                .and_then(|(_, v)| match v {
                    GgufMetaValue::U32(n) => Some(*n as u64),
                    GgufMetaValue::U64(n) => Some(*n),
                    _ => None,
                })
            )
    }

    /// Model hidden / embedding dimension.
    pub fn hidden_size(&self) -> Option<u64> {
        self.meta_u64(&format!("{}.embedding_length", self.architecture))
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> Option<u64> {
        self.meta_u64(&format!("{}.vocab_size", self.architecture))
            .or_else(|| self.metadata.iter()
                .find(|(k, _)| k.contains("vocab_size"))
                .and_then(|(_, v)| match v {
                    GgufMetaValue::U32(n) => Some(*n as u64),
                    GgufMetaValue::U64(n) => Some(*n),
                    _ => None,
                })
            )
    }
}

/// A metadata value from a GGUF header.
#[derive(Debug, Clone)]
pub enum GgufMetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<GgufMetaValue>),
}

/// Tensor descriptor from a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgufDtype,
    pub offset: u64,
}

impl GgufTensorInfo {
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// GGUF tensor data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDtype {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    BF16,
    I8,
    I16,
    I32,
    I64,
    F64,
    Unknown(u32),
}

impl GgufDtype {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            16 => Self::I8,
            17 => Self::I16,
            18 => Self::I32,
            19 => Self::I64,
            20 => Self::F64,
            other => Self::Unknown(other),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32", Self::F16 => "F16", Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0", Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0", Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0", Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K", Self::Q3K => "Q3_K", Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K", Self::Q6K => "Q6_K", Self::Q8K => "Q8_K",
            Self::I8 => "I8", Self::I16 => "I16", Self::I32 => "I32",
            Self::I64 => "I64", Self::F64 => "F64",
            Self::Unknown(_) => "Unknown",
        }
    }
}

/// Read metadata and tensor descriptors from a GGUF file.
pub fn read_gguf_info(path: &str) -> Result<GgufInfo, TrainApiError> {
    let mut file = File::open(path).map_err(|e| TrainApiError {
        message: format!("Failed to open GGUF file '{path}': {e}"),
    })?;

    // GGUF files can be large; we only need the header and tensor table.
    // Read enough to parse all metadata (first 50MB should cover most models).
    let max_read = 50 * 1024 * 1024usize;
    let mut buf = vec![0u8; max_read];
    let n = file.read(&mut buf).map_err(|e| TrainApiError {
        message: format!("Failed to read GGUF file: {e}"),
    })?;
    let buf = &buf[..n];

    parse_gguf(buf, path)
}

fn parse_gguf(buf: &[u8], path: &str) -> Result<GgufInfo, TrainApiError> {
    let mut pos = 0usize;

    // Magic: "GGUF"
    let magic = read_bytes(buf, &mut pos, 4)?;
    if magic != b"GGUF" {
        return Err(TrainApiError {
            message: format!("File '{path}' is not a GGUF file (bad magic)"),
        });
    }

    // Version (u32 LE)
    let version = read_u32(buf, &mut pos)?;
    if version < 1 || version > 3 {
        return Err(TrainApiError {
            message: format!("Unsupported GGUF version {version}"),
        });
    }

    // Tensor count and metadata count (v1: u32, v2+: u64)
    let (tensor_count, metadata_kv_count) = if version == 1 {
        (read_u32(buf, &mut pos)? as u64, read_u32(buf, &mut pos)? as u64)
    } else {
        (read_u64(buf, &mut pos)?, read_u64(buf, &mut pos)?)
    };

    // Read metadata KV pairs
    let mut metadata: HashMap<String, GgufMetaValue> = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(buf, &mut pos, version)?;
        let value_type = read_u32(buf, &mut pos)?;
        let value = read_gguf_value(buf, &mut pos, value_type, version)?;
        metadata.insert(key, value);
    }

    // Get architecture
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetaValue::Str(s)) => s.clone(),
        _ => "unknown".to_string(),
    };

    // Read tensor info
    let mut tensors = Vec::with_capacity(tensor_count.min(10000) as usize);
    let mut total_params = 0u64;

    for _ in 0..tensor_count {
        let name = read_gguf_string(buf, &mut pos, version)?;

        // n_dims (u32)
        let n_dims = read_u32(buf, &mut pos)? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(read_u64(buf, &mut pos)?);
        }

        let dtype_raw = read_u32(buf, &mut pos)?;
        let dtype = GgufDtype::from_u32(dtype_raw);

        let offset = read_u64(buf, &mut pos)?;

        let elements: u64 = shape.iter().product();
        total_params += elements;

        tensors.push(GgufTensorInfo { name, shape, dtype, offset });
    }

    Ok(GgufInfo { version, architecture, metadata, tensors, total_params })
}

// ── Binary reading helpers ────────────────────────────────────────────────────

fn read_bytes<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], TrainApiError> {
    if *pos + n > buf.len() {
        return Err(TrainApiError {
            message: format!("Unexpected end of GGUF data at offset {}", *pos),
        });
    }
    let s = &buf[*pos..*pos + n];
    *pos += n;
    Ok(s)
}

fn read_u8(buf: &[u8], pos: &mut usize) -> Result<u8, TrainApiError> {
    Ok(read_bytes(buf, pos, 1)?[0])
}
fn read_i8(buf: &[u8], pos: &mut usize) -> Result<i8, TrainApiError> {
    Ok(read_u8(buf, pos)? as i8)
}
fn read_u16(buf: &[u8], pos: &mut usize) -> Result<u16, TrainApiError> {
    let b = read_bytes(buf, pos, 2)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}
fn read_i16(buf: &[u8], pos: &mut usize) -> Result<i16, TrainApiError> {
    Ok(read_u16(buf, pos)? as i16)
}
fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, TrainApiError> {
    let b = read_bytes(buf, pos, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
fn read_i32(buf: &[u8], pos: &mut usize) -> Result<i32, TrainApiError> {
    Ok(read_u32(buf, pos)? as i32)
}
fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, TrainApiError> {
    let b = read_bytes(buf, pos, 8)?;
    Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}
fn read_i64(buf: &[u8], pos: &mut usize) -> Result<i64, TrainApiError> {
    Ok(read_u64(buf, pos)? as i64)
}
fn read_f32(buf: &[u8], pos: &mut usize) -> Result<f32, TrainApiError> {
    let b = read_bytes(buf, pos, 4)?;
    Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
fn read_f64(buf: &[u8], pos: &mut usize) -> Result<f64, TrainApiError> {
    let b = read_bytes(buf, pos, 8)?;
    Ok(f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}

fn read_gguf_string(buf: &[u8], pos: &mut usize, version: u32) -> Result<String, TrainApiError> {
    let len = if version == 1 {
        read_u32(buf, pos)? as usize
    } else {
        read_u64(buf, pos)? as usize
    };

    if len > 1_000_000 {
        return Err(TrainApiError {
            message: format!("GGUF string too long: {len} bytes"),
        });
    }

    let bytes = read_bytes(buf, pos, len)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| TrainApiError {
        message: "Invalid UTF-8 in GGUF string".to_string(),
    })
}

fn read_gguf_value(
    buf: &[u8],
    pos: &mut usize,
    value_type: u32,
    version: u32,
) -> Result<GgufMetaValue, TrainApiError> {
    match value_type {
        0  => Ok(GgufMetaValue::U8(read_u8(buf, pos)?)),
        1  => Ok(GgufMetaValue::I8(read_i8(buf, pos)?)),
        2  => Ok(GgufMetaValue::U16(read_u16(buf, pos)?)),
        3  => Ok(GgufMetaValue::I16(read_i16(buf, pos)?)),
        4  => Ok(GgufMetaValue::U32(read_u32(buf, pos)?)),
        5  => Ok(GgufMetaValue::I32(read_i32(buf, pos)?)),
        6  => Ok(GgufMetaValue::F32(read_f32(buf, pos)?)),
        7  => Ok(GgufMetaValue::Bool(read_u8(buf, pos)? != 0)),
        8  => Ok(GgufMetaValue::Str(read_gguf_string(buf, pos, version)?)),
        9  => {
            // Array: element_type(u32) + count(u64) + elements
            let elem_type = read_u32(buf, pos)?;
            let count = read_u64(buf, pos)? as usize;
            let mut arr = Vec::with_capacity(count.min(10000));
            for _ in 0..count.min(10000) {
                arr.push(read_gguf_value(buf, pos, elem_type, version)?);
            }
            Ok(GgufMetaValue::Array(arr))
        }
        10 => Ok(GgufMetaValue::U64(read_u64(buf, pos)?)),
        11 => Ok(GgufMetaValue::I64(read_i64(buf, pos)?)),
        12 => Ok(GgufMetaValue::F64(read_f64(buf, pos)?)),
        other => Err(TrainApiError {
            message: format!("Unknown GGUF metadata value type {other}"),
        }),
    }
}

/// Print a human-readable summary of a GGUF file.
pub fn print_gguf_info(info: &GgufInfo) {
    println!("=== GGUF File Info ===");
    println!("Version:      {}", info.version);
    println!("Architecture: {}", info.architecture);
    println!("Tensors:      {}", info.tensors.len());
    println!("Total params: {:.2}B ({} elements)", info.total_params as f64 / 1e9, info.total_params);

    if let Some(n) = info.num_layers() { println!("Layers:       {n}"); }
    if let Some(h) = info.hidden_size() { println!("Hidden size:  {h}"); }
    if let Some(v) = info.vocab_size() { println!("Vocab size:   {v}"); }

    println!("\nTop 10 tensors:");
    println!("{:<50} {:>12} {:>8}", "Name", "Elements", "Dtype");
    println!("{}", "-".repeat(72));
    for t in info.tensors.iter().take(10) {
        let shape_str = t.shape.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("×");
        println!("{:<50} {:>12} {:>8}  [{}]",
            &t.name[..t.name.len().min(49)],
            t.num_elements(),
            t.dtype.name(),
            shape_str
        );
    }
    if info.tensors.len() > 10 {
        println!("... and {} more tensors", info.tensors.len() - 10);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_gguf_info_rejects_non_gguf_file() {
        let path = std::env::temp_dir()
            .join("volta_gguf_test_not_gguf.bin")
            .to_string_lossy()
            .into_owned();
        std::fs::write(&path, b"NOT A GGUF FILE AT ALL").unwrap();
        let err = read_gguf_info(&path).expect_err("Should fail on non-GGUF file");
        assert!(err.message.contains("GGUF") || err.message.contains("magic"),
            "Error should mention GGUF magic, got: {}", err.message);
    }

    #[test]
    fn gguf_dtype_name_roundtrip() {
        assert_eq!(GgufDtype::from_u32(0).name(), "F32");
        assert_eq!(GgufDtype::from_u32(1).name(), "F16");
        assert_eq!(GgufDtype::from_u32(12).name(), "Q4_K");
        assert_eq!(GgufDtype::from_u32(9999).name(), "Unknown");
    }

    #[test]
    fn gguf_tensor_info_num_elements() {
        let t = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![4096, 4096],
            dtype: GgufDtype::F16,
            offset: 0,
        };
        assert_eq!(t.num_elements(), 4096 * 4096);
    }
}
