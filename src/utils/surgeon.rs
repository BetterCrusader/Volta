use serde_json::Value;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

/// Шукає модель на комп'ютері за назвою, реверс-інжинірить і створює ідеальний .vt файл.
pub fn hunt_and_extract(model_name: &str) -> Result<(), String> {
    println!("💉 Volta Surgeon: Activated Global Hunt Protocol.");
    println!("🔎 Searching system for: '{}'...", model_name);

    let start_time = Instant::now();
    let found_path = find_model_file(model_name).ok_or_else(|| {
        format!(
            "Model '{}' not found on the system. Check the name.",
            model_name
        )
    })?;

    println!(
        "🎯 Target acquired in {:.2}s: {}",
        start_time.elapsed().as_secs_f64(),
        found_path.display()
    );
    println!("🧬 Commencing architectural extraction...\n");

    let mut file = File::open(&found_path).map_err(|e| format!("Could not open file: {}", e))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|_| "Failed to read magic bytes")?;
    file.seek(std::io::SeekFrom::Start(0)).unwrap();

    if magic == *b"GGUF" {
        extract_gguf(&found_path, model_name)
    } else {
        extract_safetensors(&found_path, model_name)
    }
}

fn extract_gguf(path: &Path, model_name: &str) -> Result<(), String> {
    let mut file = File::open(path).map_err(|e| format!("Could not open file: {}", e))?;

    // Read first 10MB for metadata/tensor info
    let mut buffer = vec![0u8; 10 * 1024 * 1024];
    let n = file
        .read(&mut buffer)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    let buffer = &buffer[..n];

    let gguf_opt =
        gguf::GGUFFile::read(buffer).map_err(|e| format!("Failed to parse GGUF: {}", e))?;
    let gguf = gguf_opt.ok_or("Failed to parse GGUF: returned None")?;

    let mut total_params = 0u64;
    for tensor in &gguf.tensors {
        let mut elements = 1u64;
        for dim in &tensor.dimensions {
            elements *= *dim as u64;
        }
        total_params += elements;
    }

    let mut layer_count = 0;
    let mut hidden_size = 0;
    let mut vocab_size = 0;
    let mut model_type_owned = String::from("transformer");

    for meta in &gguf.header.metadata {
        if meta.key == "general.architecture" {
            if let gguf::GGUFMetadataValue::String(s) = &meta.value {
                model_type_owned = s.clone();
            }
        }
    }

    let model_type = &model_type_owned;

    for meta in &gguf.header.metadata {
        if meta.key == format!("{}.block_count", model_type) {
            match &meta.value {
                gguf::GGUFMetadataValue::Uint32(n) => layer_count = *n as usize,
                gguf::GGUFMetadataValue::Uint64(n) => layer_count = *n as usize,
                _ => {}
            }
        }
        if meta.key == format!("{}.embedding_length", model_type) {
            match &meta.value {
                gguf::GGUFMetadataValue::Uint32(n) => hidden_size = *n as usize,
                gguf::GGUFMetadataValue::Uint64(n) => hidden_size = *n as usize,
                _ => {}
            }
        }
        if meta.key.contains("vocab_size") {
            match &meta.value {
                gguf::GGUFMetadataValue::Uint32(n) => vocab_size = *n as usize,
                gguf::GGUFMetadataValue::Uint64(n) => vocab_size = *n as usize,
                _ => {}
            }
        }
    }

    if layer_count == 0 {
        for meta in &gguf.header.metadata {
            if meta.key.contains("block_count") {
                if let gguf::GGUFMetadataValue::Uint32(n) = &meta.value {
                    layer_count = *n as usize;
                }
            }
        }
    }

    emit_vt_file(
        model_name,
        path,
        model_type,
        total_params,
        layer_count,
        hidden_size,
        vocab_size,
    )
}

fn extract_safetensors(path: &Path, model_name: &str) -> Result<(), String> {
    let mut file = File::open(path).map_err(|e| format!("Could not open file: {}", e))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|_| "Failed to read header length")?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;

    if header_len > 100_000_000 {
        return Err(format!(
            "Header is suspiciously large ({} bytes).",
            header_len
        ));
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|_| "Failed to read JSON header")?;
    let metadata: Value = serde_json::from_str(&String::from_utf8_lossy(&header_bytes))
        .map_err(|_| "Invalid JSON format")?;
    let obj = metadata.as_object().ok_or("Header is not a JSON object")?;

    let mut total_params = 0u64;
    let mut layer_count = 0;
    let mut hidden_size = 0;
    let mut vocab_size = 0;
    let mut actual_model_type = "UnknownModel";

    for (key, val) in obj {
        if key == "__metadata__" {
            if let Some(meta_obj) = val.as_object()
                && let Some(mt) = meta_obj.get("model_type").and_then(|v| v.as_str())
            {
                actual_model_type = mt;
            }
            continue;
        }

        if let Some(info) = val.as_object()
            && let Some(shape) = info.get("shape").and_then(|s| s.as_array())
        {
            let mut elements = 1u64;
            for dim in shape {
                if let Some(d) = dim.as_u64() {
                    elements *= d;
                }
            }
            total_params += elements;

            if key.contains("embed_tokens") && shape.len() == 2 {
                vocab_size = shape[0].as_u64().unwrap_or(0) as usize;
                hidden_size = shape[1].as_u64().unwrap_or(0) as usize;
            }

            if key.contains(".layers.") || key.contains(".blocks.") || key.contains(".h.") {
                let parts: Vec<&str> = key.split('.').collect();
                for part in parts {
                    if let Ok(num) = part.parse::<usize>()
                        && num > layer_count
                    {
                        layer_count = num;
                    }
                }
            }
        }
    }

    if layer_count > 0 {
        layer_count += 1;
    }
    let safe_model_type = if actual_model_type == "UnknownModel" {
        "transformer"
    } else {
        actual_model_type
    };

    emit_vt_file(
        model_name,
        path,
        safe_model_type,
        total_params,
        layer_count,
        hidden_size,
        vocab_size,
    )
}

fn emit_vt_file(
    model_name: &str,
    path: &Path,
    model_type: &str,
    total_params: u64,
    layer_count: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<(), String> {
    let clean_name = model_name
        .replace("-", "_")
        .replace(".safetensors", "")
        .replace(".gguf", "");

    let vt_code = format!(
        "// 🤖 VOLTA AUTO-REVERSE-ENGINEERING PROTOCOL
// Target Location: {}
// Base Architecture: {}
// Total Parameters: ~{:.2}B

model {}_mutant
    source \"{}\"
    architecture {}
    layers {}
    hidden_size {}
    vocab_size {}

    // ⚠️ SURGEON WARNING ⚠️
    // Changing 'vocab_size' or 'hidden_size' has a 99% chance of corrupting the tensor map!
    // If you don't know what you are doing, LEAVE THEM ALONE.

    // ✂️ SURGERY ROOM (Safe structural mutations):
    // drop layers 16 to {}
    // quantize weights to int8

    // 🎛️ RUNTIME HYPERPARAMETERS (Free to change):
    temperature 0.7
    top_p 0.9
    max_tokens 2048

dataset custom_prompts
    batch 1
    stream true

train {}_mutant on custom_prompts
    epochs 1
    device auto

// Run this script to apply mutations:
// volta run mutant.vt
",
        path.display(),
        model_type,
        total_params as f64 / 1_000_000_000.0,
        clean_name,
        path.display(),
        model_type,
        layer_count,
        hidden_size,
        vocab_size,
        layer_count.saturating_sub(1),
        clean_name
    );

    let output_file_name = format!("{}_mutant.vt", clean_name);
    let mut out_file =
        File::create(&output_file_name).map_err(|e| format!("Failed to create file: {}", e))?;
    out_file
        .write_all(vt_code.as_bytes())
        .map_err(|e| format!("Failed to write file: {}", e))?;

    println!(
        "✅ Extraction Complete! Architecture saved to 📄 {}",
        output_file_name
    );
    println!("   Open this file, make your changes, and run it to slice the model.");

    Ok(())
}

/// Шукає файл моделі у стандартних папках завантажень, LM Studio, Ollama та HuggingFace кеші
fn find_model_file(name: &str) -> Option<PathBuf> {
    if Path::new(name).exists() {
        return Some(PathBuf::from(name));
    }

    let search_name =
        if !name.ends_with(".safetensors") && !name.ends_with(".gguf") && !name.ends_with(".bin") {
            format!("{}.safetensors", name)
        } else {
            name.to_string()
        };

    let home_dir = dirs::home_dir()?;
    let win_home = PathBuf::from("/mnt/c/Users/User");

    let mut search_dirs = vec![
        home_dir.join(".cache/huggingface/hub"),
        home_dir.join(".cache/lm-studio/models"),
        home_dir.join(".ollama/models"),
        home_dir.join("Downloads"),
        PathBuf::from("."),
    ];

    if win_home.exists() {
        search_dirs.push(win_home.join(".ollama/models/blobs"));
        search_dirs.push(win_home.join(".cache/lm-studio/models"));
        search_dirs.push(win_home.join("Downloads"));
    }

    for dir in search_dirs {
        if !dir.exists() {
            continue;
        }

        for entry in WalkDir::new(&dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file()
                && let Some(file_name) = path.file_name().and_then(|n| n.to_str())
            {
                if file_name.eq_ignore_ascii_case(&search_name)
                    || (file_name.ends_with(".safetensors")
                        && file_name.to_lowercase().contains(&name.to_lowercase()))
                    || (file_name.ends_with(".gguf")
                        && file_name.to_lowercase().contains(&name.to_lowercase()))
                {
                    return Some(path.to_path_buf());
                }
            }
        }
    }

    None
}
