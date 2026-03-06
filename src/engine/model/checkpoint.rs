use std::collections::HashMap;
use std::fs;

use crate::ir::Tensor;

use crate::model::TrainApiError;

const CHECKPOINT_HEADER_PREFIX: &str = "#volta-checkpoint:";
const CHECKPOINT_VERSION_V1: &str = "v1";
const CHECKPOINT_HEADER_V1: &str = "#volta-checkpoint:v1";
/// Magic bytes for the compressed binary checkpoint format v2.
const CHECKPOINT_MAGIC: &[u8; 8] = b"VOLTA\x02\x00\x00";

pub fn save_checkpoint(
    path: &str,
    parameters: &HashMap<String, Tensor>,
) -> Result<(), TrainApiError> {
    let mut entries = parameters.iter().collect::<Vec<_>>();
    entries.sort_by(|(a, _), (b, _)| a.cmp(b));

    let mut lines = vec![
        CHECKPOINT_HEADER_V1.to_string(),
        "#writer:train_api".to_string(),
    ];
    for (name, tensor) in entries {
        let shape = tensor
            .shape
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let data = tensor
            .data
            .iter()
            .map(|v| format!("{v:.9}"))
            .collect::<Vec<_>>()
            .join(",");
        lines.push(format!("{name}|{shape}|{data}"));
    }

    fs::write(path, lines.join("\n")).map_err(|err| TrainApiError {
        message: format!("Failed to save checkpoint to '{path}': {err}"),
    })
}

pub fn load_checkpoint(path: &str) -> Result<HashMap<String, Tensor>, TrainApiError> {
    let content = fs::read_to_string(path).map_err(|err| TrainApiError {
        message: format!("Failed to read checkpoint from '{path}': {err}"),
    })?;

    let mut parameters = HashMap::new();
    let mut lines = content.lines().enumerate().peekable();

    while let Some((_, line)) = lines.peek() {
        if line.trim().is_empty() {
            lines.next();
            continue;
        }
        break;
    }

    if let Some((line_no, line)) = lines.peek().copied()
        && parse_header_line(line_no, line)?
    {
        lines.next();
    }

    for (line_no, line) in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if parse_header_line(line_no, line)? {
            continue;
        }

        if trimmed.starts_with('#') {
            continue;
        }

        let (name, tensor) = parse_parameter_line(line_no, trimmed)?;
        parameters.insert(name, tensor);
    }

    Ok(parameters)
}

fn parse_header_line(line_no: usize, line: &str) -> Result<bool, TrainApiError> {
    let trimmed = line.trim();
    let Some(version) = trimmed.strip_prefix(CHECKPOINT_HEADER_PREFIX) else {
        return Ok(false);
    };

    if version != CHECKPOINT_VERSION_V1 {
        return Err(TrainApiError {
            message: format!(
                "Unsupported checkpoint version '{}' at line {}",
                version,
                line_no + 1
            ),
        });
    }

    Ok(true)
}

fn parse_parameter_line(line_no: usize, line: &str) -> Result<(String, Tensor), TrainApiError> {
    let parts = line.split('|').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(TrainApiError {
            message: format!("Invalid checkpoint format at line {}", line_no + 1),
        });
    }

    let name = parts[0].to_string();
    let shape = if parts[1].is_empty() {
        vec![]
    } else {
        let mut dims = Vec::new();
        for dim in parts[1].split(',') {
            let parsed = dim.parse::<usize>().map_err(|err| TrainApiError {
                message: format!(
                    "Invalid shape dimension '{}' at line {}: {err}",
                    dim,
                    line_no + 1
                ),
            })?;
            dims.push(parsed);
        }
        dims
    };

    let data = if parts[2].is_empty() {
        vec![]
    } else {
        let mut out = Vec::new();
        for value in parts[2].split(',') {
            let parsed = value.parse::<f32>().map_err(|err| TrainApiError {
                message: format!(
                    "Invalid data value '{}' at line {}: {err}",
                    value,
                    line_no + 1
                ),
            })?;
            out.push(parsed);
        }
        out
    };

    let tensor = Tensor::new(shape, data).map_err(|err| TrainApiError {
        message: format!(
            "Invalid tensor in checkpoint at line {}: {}",
            line_no + 1,
            err.message
        ),
    })?;

    Ok((name, tensor))
}

/// Save checkpoint in compressed binary format (v2).
///
/// Format: magic(8) + entry_count(u32 LE) + for each entry:
///   name_len(u16 LE) + name bytes + ndim(u32 LE) + dims(u32 LE each) + data(f32 LE each)
/// Then the whole payload is zlib-compressed.
pub fn save_checkpoint_compressed(
    path: &str,
    parameters: &HashMap<String, Tensor>,
) -> Result<(), TrainApiError> {
    let mut payload: Vec<u8> = Vec::new();

    // Write entry count
    let count = parameters.len() as u32;
    payload.extend_from_slice(&count.to_le_bytes());

    // Sort for determinism
    let mut entries: Vec<(&String, &Tensor)> = parameters.iter().collect();
    entries.sort_by_key(|(k, _)| k.as_str());

    for (name, tensor) in entries {
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() as u16;
        payload.extend_from_slice(&name_len.to_le_bytes());
        payload.extend_from_slice(name_bytes);

        let ndim = tensor.shape.len() as u32;
        payload.extend_from_slice(&ndim.to_le_bytes());
        for &dim in &tensor.shape {
            payload.extend_from_slice(&(dim as u32).to_le_bytes());
        }

        // f32 data
        let n_elems = tensor.data.len() as u32;
        payload.extend_from_slice(&n_elems.to_le_bytes());
        for &v in tensor.data.iter() {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }

    // Zlib compress
    let compressed = zlib_compress(&payload)?;

    // Write file: magic + compressed data
    let mut out = Vec::with_capacity(CHECKPOINT_MAGIC.len() + compressed.len());
    out.extend_from_slice(CHECKPOINT_MAGIC);
    out.extend_from_slice(&compressed);

    fs::write(path, &out).map_err(|e| TrainApiError {
        message: format!("Failed to write compressed checkpoint '{path}': {e}"),
    })
}

/// Load checkpoint from compressed binary format (v2) or text format (v1).
pub fn load_checkpoint_compressed(path: &str) -> Result<HashMap<String, Tensor>, TrainApiError> {
    let raw = fs::read(path).map_err(|e| TrainApiError {
        message: format!("Failed to read checkpoint '{path}': {e}"),
    })?;

    // Detect format by magic bytes
    if raw.starts_with(CHECKPOINT_MAGIC) {
        load_v2_binary(&raw[CHECKPOINT_MAGIC.len()..])
    } else {
        // Fall back to text format
        let text = std::str::from_utf8(&raw).map_err(|_| TrainApiError {
            message: format!("Checkpoint '{path}' has unknown format"),
        })?;
        load_checkpoint_from_str(text)
    }
}

fn load_v2_binary(compressed: &[u8]) -> Result<HashMap<String, Tensor>, TrainApiError> {
    let payload = zlib_decompress(compressed)?;
    let mut pos = 0usize;

    let entry_count = read_u32(&payload, &mut pos)? as usize;
    let mut parameters = HashMap::with_capacity(entry_count);

    for _ in 0..entry_count {
        let name_len = read_u16(&payload, &mut pos)? as usize;
        let name_bytes = read_bytes(&payload, &mut pos, name_len)?;
        let name = String::from_utf8(name_bytes.to_vec()).map_err(|_| TrainApiError {
            message: "Invalid UTF-8 in parameter name".to_string(),
        })?;

        let ndim = read_u32(&payload, &mut pos)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u32(&payload, &mut pos)? as usize);
        }

        let n_elems = read_u32(&payload, &mut pos)? as usize;
        let mut data = Vec::with_capacity(n_elems);
        for _ in 0..n_elems {
            let bytes = read_bytes(&payload, &mut pos, 4)?;
            data.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        }

        let tensor = Tensor::new(shape, data).map_err(|e| TrainApiError {
            message: format!("Invalid tensor for '{name}': {}", e.message),
        })?;
        parameters.insert(name, tensor);
    }

    Ok(parameters)
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, TrainApiError> {
    let bytes = read_bytes(buf, pos, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u16(buf: &[u8], pos: &mut usize) -> Result<u16, TrainApiError> {
    let bytes = read_bytes(buf, pos, 2)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_bytes<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], TrainApiError> {
    if *pos + n > buf.len() {
        return Err(TrainApiError {
            message: format!("Unexpected end of checkpoint data at offset {}", *pos),
        });
    }
    let slice = &buf[*pos..*pos + n];
    *pos += n;
    Ok(slice)
}

/// Minimal zlib compress using Deflate (no external dep — uses raw Deflate via std).
/// We use a simple uncompressed deflate blocks approach for correctness without a dep.
/// For real compression we use the "deflate stored blocks" (type=00) which is valid
/// zlib but not compressed. A future version can use flate2 crate for actual compression.
fn zlib_compress(data: &[u8]) -> Result<Vec<u8>, TrainApiError> {
    // zlib header: CMF=0x78 (deflate, window=32768), FLG=0x9C (FCHECK so CMF*256+FLG % 31 == 0)
    let mut out = vec![0x78u8, 0x9C];

    // Emit uncompressed deflate blocks (BTYPE=00), max 65535 bytes each
    let block_size = 65535usize;
    let mut remaining = data;
    while !remaining.is_empty() {
        let chunk_len = remaining.len().min(block_size);
        let chunk = &remaining[..chunk_len];
        remaining = &remaining[chunk_len..];
        let is_last = remaining.is_empty();

        // BFINAL | BTYPE=00
        out.push(if is_last { 0x01 } else { 0x00 });
        let len = chunk_len as u16;
        let nlen = !len;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(chunk);
    }

    // Adler-32 checksum
    let adler = adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());

    Ok(out)
}

fn zlib_decompress(data: &[u8]) -> Result<Vec<u8>, TrainApiError> {
    if data.len() < 6 {
        return Err(TrainApiError {
            message: "Compressed data too short".to_string(),
        });
    }
    // Skip 2-byte zlib header
    let mut pos = 2usize;
    let mut out = Vec::new();

    loop {
        if pos >= data.len() {
            return Err(TrainApiError {
                message: "Unexpected end in deflate stream".to_string(),
            });
        }
        let bfinal_btype = data[pos];
        pos += 1;
        let bfinal = bfinal_btype & 1;
        let btype = (bfinal_btype >> 1) & 3;

        match btype {
            0 => {
                // Uncompressed block
                if pos + 4 > data.len() {
                    return Err(TrainApiError {
                        message: "Truncated deflate block header".to_string(),
                    });
                }
                let len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 4; // skip len + nlen
                if pos + len > data.len() {
                    return Err(TrainApiError {
                        message: "Truncated deflate block data".to_string(),
                    });
                }
                out.extend_from_slice(&data[pos..pos + len]);
                pos += len;
            }
            _ => {
                return Err(TrainApiError {
                    message: format!(
                        "Compressed deflate BTYPE={btype} not supported in this decompressor"
                    ),
                });
            }
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(out)
}

fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

fn load_checkpoint_from_str(content: &str) -> Result<HashMap<String, Tensor>, TrainApiError> {
    let mut parameters = HashMap::new();
    let mut lines = content.lines().enumerate().peekable();

    while let Some((_, line)) = lines.peek() {
        if line.trim().is_empty() {
            lines.next();
            continue;
        }
        break;
    }

    if let Some((line_no, line)) = lines.peek().copied()
        && parse_header_line(line_no, line)?
    {
        lines.next();
    }

    for (line_no, line) in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if parse_header_line(line_no, line)? {
            continue;
        }
        if trimmed.starts_with('#') {
            continue;
        }
        let (name, tensor) = parse_parameter_line(line_no, trimmed)?;
        parameters.insert(name, tensor);
    }

    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::ir::Tensor;
    use crate::model::{load_checkpoint, save_checkpoint};

    fn temp_checkpoint_path(label: &str) -> String {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after unix epoch")
            .as_nanos();
        std::env::temp_dir()
            .join(format!("volta-checkpoint-{label}-{nonce}.txt"))
            .to_string_lossy()
            .into_owned()
    }

    fn sample_parameters() -> HashMap<String, Tensor> {
        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1, 2], vec![1.0, 2.0]).expect("tensor"),
        );
        params.insert(
            "b".to_string(),
            Tensor::new(vec![1, 1], vec![0.25]).expect("tensor"),
        );
        params
    }

    #[test]
    fn checkpoint_v1_roundtrip_preserves_parameters() {
        let path = temp_checkpoint_path("v1-roundtrip");
        let params = sample_parameters();

        save_checkpoint(&path, &params).expect("save should pass");

        let saved = fs::read_to_string(&path).expect("checkpoint file should exist");
        assert!(
            saved.starts_with("#volta-checkpoint:v1\n"),
            "checkpoint must start with v1 header"
        );

        let loaded = load_checkpoint(&path).expect("load should pass");
        assert_eq!(loaded, params, "loaded params must match exactly");
    }

    #[test]
    fn checkpoint_loader_accepts_legacy_format() {
        let path = temp_checkpoint_path("legacy");
        fs::write(&path, "w|1,2|1.000000000,2.000000000\n")
            .expect("legacy checkpoint write should pass");

        let loaded = load_checkpoint(&path).expect("legacy load should pass");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded["w"].shape, vec![1, 2]);
        assert_eq!(*loaded["w"].data, vec![1.0, 2.0]);
    }

    #[test]
    fn checkpoint_loader_ignores_unknown_comment_lines() {
        let path = temp_checkpoint_path("comments");
        let content = [
            "#volta-checkpoint:v1",
            "#this-is-an-unknown-comment",
            "#another-comment=value",
            "w|1,2|1.000000000,2.000000000",
        ]
        .join("\n");
        fs::write(&path, content).expect("commented checkpoint write should pass");

        let loaded = load_checkpoint(&path).expect("loader should ignore comments");
        assert_eq!(loaded["w"].shape, vec![1, 2]);
    }

    #[test]
    fn roundtrip_checkpoint_is_deterministic() {
        let path_a = temp_checkpoint_path("stable-a");
        let path_b = temp_checkpoint_path("stable-b");
        let params = sample_parameters();

        save_checkpoint(&path_a, &params).expect("first save should pass");
        let loaded = load_checkpoint(&path_a).expect("first load should pass");
        save_checkpoint(&path_b, &loaded).expect("second save should pass");

        let first = fs::read_to_string(&path_a).expect("first checkpoint should exist");
        let second = fs::read_to_string(&path_b).expect("second checkpoint should exist");

        assert_eq!(first, second, "checkpoint bytes must be stable");
    }

    #[test]
    fn checkpoint_save_load_save_load_remains_stable() {
        let path_a = temp_checkpoint_path("save-load-save-a");
        let path_b = temp_checkpoint_path("save-load-save-b");
        let params = sample_parameters();

        save_checkpoint(&path_a, &params).expect("initial save should pass");
        let loaded_once = load_checkpoint(&path_a).expect("first load should pass");
        save_checkpoint(&path_b, &loaded_once).expect("second save should pass");
        let loaded_twice = load_checkpoint(&path_b).expect("second load should pass");

        assert_eq!(loaded_once, loaded_twice, "double load must preserve data");

        let file_once = fs::read_to_string(&path_a).expect("first file should exist");
        let file_twice = fs::read_to_string(&path_b).expect("second file should exist");
        assert_eq!(file_once, file_twice, "double save must be byte-identical");
    }
}
