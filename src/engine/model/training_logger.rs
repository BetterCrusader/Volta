/// Training event logger with TensorBoard and WandB compatibility.
///
/// Supports three output modes:
/// - `JsonLines`: one JSON object per line (compatible with WandB `wandb.log()` replay and custom dashboards)
/// - `TfEvents`: minimal TFEvents v2 binary format for TensorBoard (scalar summaries only)
/// - `Both`: write both simultaneously
///
/// Usage:
/// ```no_run
/// use volta::model::{TrainingLogger, LoggerConfig, LogTarget};
/// let mut logger = TrainingLogger::new(LoggerConfig {
///     dir: "runs/experiment1".to_string(),
///     target: LogTarget::Both,
/// }).unwrap();
/// logger.log_scalar("loss", 0.42, 100);
/// logger.log_scalar("accuracy", 0.95, 100);
/// logger.flush().unwrap();
/// ```
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::model::TrainApiError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogTarget {
    JsonLines,
    TfEvents,
    Both,
}

#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Directory where log files are written.
    pub dir: String,
    /// Which format(s) to write.
    pub target: LogTarget,
}

/// A single logged event.
#[derive(Debug, Clone)]
pub struct LogEvent {
    pub tag: String,
    pub value: f64,
    pub step: u64,
    pub wall_time: f64,
}

/// Training event logger.
pub struct TrainingLogger {
    config: LoggerConfig,
    jsonl_path: Option<PathBuf>,
    tfevents_path: Option<PathBuf>,
    pending: Vec<LogEvent>,
    /// File handle for JSON lines (append mode).
    jsonl_file: Option<File>,
    /// File handle for TFEvents (append mode, binary).
    tfevents_file: Option<File>,
}

impl TrainingLogger {
    /// Create a new logger, creating the output directory if needed.
    pub fn new(config: LoggerConfig) -> Result<Self, TrainApiError> {
        let dir = Path::new(&config.dir);
        fs::create_dir_all(dir).map_err(|e| TrainApiError {
            message: format!("Failed to create log directory '{}': {e}", config.dir),
        })?;

        let wall = wall_time_secs();
        let run_id = (wall * 1000.0) as u64;

        let (jsonl_path, jsonl_file) =
            if matches!(config.target, LogTarget::JsonLines | LogTarget::Both) {
                let p = dir.join("events.jsonl");
                let f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&p)
                    .map_err(|e| TrainApiError {
                        message: format!("Failed to open JSONL log file: {e}"),
                    })?;
                (Some(p), Some(f))
            } else {
                (None, None)
            };

        let (tfevents_path, tfevents_file) =
            if matches!(config.target, LogTarget::TfEvents | LogTarget::Both) {
                let filename = format!("events.out.tfevents.{run_id}.volta");
                let p = dir.join(&filename);
                let mut f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&p)
                    .map_err(|e| TrainApiError {
                        message: format!("Failed to open TFEvents file: {e}"),
                    })?;
                // Write file_version event (empty summary at step 0)
                write_tfevents_file_header(&mut f, wall)?;
                (Some(p), Some(f))
            } else {
                (None, None)
            };

        Ok(Self {
            config,
            jsonl_path,
            tfevents_path,
            pending: Vec::new(),
            jsonl_file,
            tfevents_file,
        })
    }

    /// Log a scalar value at the given step.
    pub fn log_scalar(&mut self, tag: &str, value: f64, step: u64) {
        self.pending.push(LogEvent {
            tag: tag.to_string(),
            value,
            step,
            wall_time: wall_time_secs(),
        });
    }

    /// Log multiple scalars at the same step (e.g., a dict of metrics).
    pub fn log_scalars(&mut self, metrics: &HashMap<&str, f64>, step: u64) {
        let wall = wall_time_secs();
        for (&tag, &value) in metrics {
            self.pending.push(LogEvent {
                tag: tag.to_string(),
                value,
                step,
                wall_time: wall,
            });
        }
    }

    /// Flush all pending events to disk.
    pub fn flush(&mut self) -> Result<(), TrainApiError> {
        let events: Vec<LogEvent> = std::mem::take(&mut self.pending);

        if let Some(ref mut f) = self.jsonl_file {
            for event in &events {
                let line = format!(
                    "{{\"tag\":{:?},\"value\":{},\"step\":{},\"wall_time\":{:.3}}}\n",
                    event.tag, event.value, event.step, event.wall_time
                );
                f.write_all(line.as_bytes()).map_err(|e| TrainApiError {
                    message: format!("Failed to write JSONL event: {e}"),
                })?;
            }
            f.flush().map_err(|e| TrainApiError {
                message: format!("Failed to flush JSONL file: {e}"),
            })?;
        }

        if let Some(ref mut f) = self.tfevents_file {
            for event in &events {
                write_tfevents_scalar(f, event)?;
            }
            f.flush().map_err(|e| TrainApiError {
                message: format!("Failed to flush TFEvents file: {e}"),
            })?;
        }

        Ok(())
    }

    /// Close the logger (flushes pending events).
    pub fn close(&mut self) -> Result<(), TrainApiError> {
        self.flush()
    }

    pub fn log_dir(&self) -> &str {
        &self.config.dir
    }

    pub fn jsonl_path(&self) -> Option<&Path> {
        self.jsonl_path.as_deref()
    }

    pub fn tfevents_path(&self) -> Option<&Path> {
        self.tfevents_path.as_deref()
    }
}

fn wall_time_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ──────────────────────────────────────────────────────────────────────────────
// TFEvents binary format (minimal implementation for scalar summaries)
//
// Each record: length(u64 LE) + masked_crc32(length, u32 LE) + data + masked_crc32(data, u32 LE)
// Data is a serialized tensorflow.Event proto (hand-encoded, no prost dependency here).
//
// We encode just enough proto fields for TensorBoard to read scalar summaries:
//   Event.wall_time (field 1, double)
//   Event.step (field 2, int64)
//   Event.summary (field 5, message):
//     Summary.value (field 1, repeated message):
//       Summary.Value.tag (field 1, string)
//       Summary.Value.simple_value (field 2, float)
// ──────────────────────────────────────────────────────────────────────────────

fn write_tfevents_file_header(f: &mut File, wall: f64) -> Result<(), TrainApiError> {
    // Write a "file_version" event so TensorBoard knows the file is valid
    // Event.file_version = field 9, string "brain.Event:2"
    let mut event = Vec::new();
    encode_proto_double(&mut event, 1, wall);
    encode_proto_varint(&mut event, 2, 0); // step 0
    encode_proto_bytes(&mut event, 9, b"brain.Event:2");
    write_tfevents_record(f, &event)
}

fn write_tfevents_scalar(f: &mut File, event: &LogEvent) -> Result<(), TrainApiError> {
    // Encode Summary.Value
    let mut value_msg = Vec::new();
    encode_proto_bytes(&mut value_msg, 1, event.tag.as_bytes());
    encode_proto_float(&mut value_msg, 2, event.value as f32);

    // Encode Summary
    let mut summary_msg = Vec::new();
    encode_proto_bytes(&mut summary_msg, 1, &value_msg);

    // Encode Event
    let mut ev_msg = Vec::new();
    encode_proto_double(&mut ev_msg, 1, event.wall_time);
    encode_proto_varint(&mut ev_msg, 2, event.step);
    encode_proto_bytes(&mut ev_msg, 5, &summary_msg);

    write_tfevents_record(f, &ev_msg)
}

fn write_tfevents_record(f: &mut File, data: &[u8]) -> Result<(), TrainApiError> {
    let len = data.len() as u64;
    let len_bytes = len.to_le_bytes();
    let crc_len = masked_crc32c(&len_bytes);
    let crc_data = masked_crc32c(data);

    let mut record = Vec::with_capacity(16 + data.len());
    record.extend_from_slice(&len_bytes);
    record.extend_from_slice(&crc_len.to_le_bytes());
    record.extend_from_slice(data);
    record.extend_from_slice(&crc_data.to_le_bytes());

    f.write_all(&record).map_err(|e| TrainApiError {
        message: format!("Failed to write TFEvents record: {e}"),
    })
}

// ── Minimal protobuf encoding helpers ────────────────────────────────────────

fn encode_proto_varint(buf: &mut Vec<u8>, field: u32, value: u64) {
    let tag = (field << 3) | 0; // wire type 0 = varint
    encode_varint(buf, tag as u64);
    encode_varint(buf, value);
}

fn encode_proto_double(buf: &mut Vec<u8>, field: u32, value: f64) {
    let tag = (field << 3) | 1; // wire type 1 = 64-bit
    encode_varint(buf, tag as u64);
    buf.extend_from_slice(&value.to_le_bytes());
}

fn encode_proto_float(buf: &mut Vec<u8>, field: u32, value: f32) {
    let tag = (field << 3) | 5; // wire type 5 = 32-bit
    encode_varint(buf, tag as u64);
    buf.extend_from_slice(&value.to_le_bytes());
}

fn encode_proto_bytes(buf: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    let tag = (field << 3) | 2; // wire type 2 = length-delimited
    encode_varint(buf, tag as u64);
    encode_varint(buf, bytes.len() as u64);
    buf.extend_from_slice(bytes);
}

fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

// ── CRC32C (Castagnoli) with masking as used by TFRecords ─────────────────

fn masked_crc32c(data: &[u8]) -> u32 {
    let crc = crc32c(data);
    // TensorFlow masking: ((crc >> 15) | (crc << 17)).wrapping_add(0xA282EAD8)
    ((crc >> 15) | (crc << 17)).wrapping_add(0xA282_EAD8u32)
}

fn crc32c(data: &[u8]) -> u32 {
    // CRC32C with polynomial 0x82F63B78 (Castagnoli)
    let mut crc: u32 = !0u32;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLE[idx];
    }
    !crc
}

// Precomputed CRC32C lookup table (Castagnoli polynomial 0x82F63B78)
static CRC32C_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let poly: u32 = 0x82F63B78;
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ poly;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_dir(label: &str) -> String {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("volta-log-{label}-{nonce}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn jsonl_logger_creates_file_with_events() {
        let dir = tmp_dir("jsonl");
        let mut logger = TrainingLogger::new(LoggerConfig {
            dir: dir.clone(),
            target: LogTarget::JsonLines,
        })
        .unwrap();

        logger.log_scalar("loss", 0.5, 1);
        logger.log_scalar("acc", 0.8, 1);
        logger.flush().unwrap();

        let path = logger.jsonl_path().unwrap().to_owned();
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"loss\""));
        assert!(lines[1].contains("\"acc\""));
    }

    #[test]
    fn tfevents_logger_creates_non_empty_file() {
        let dir = tmp_dir("tfevents");
        let mut logger = TrainingLogger::new(LoggerConfig {
            dir: dir.clone(),
            target: LogTarget::TfEvents,
        })
        .unwrap();

        logger.log_scalar("loss", 1.23, 10);
        logger.flush().unwrap();

        let path = logger.tfevents_path().unwrap().to_owned();
        let data = std::fs::read(&path).unwrap();
        assert!(data.len() > 16, "TFEvents file should contain binary data");
    }

    #[test]
    fn both_logger_writes_two_files() {
        let dir = tmp_dir("both");
        let mut logger = TrainingLogger::new(LoggerConfig {
            dir: dir.clone(),
            target: LogTarget::Both,
        })
        .unwrap();

        logger.log_scalar("train/loss", 0.1, 5);
        logger.flush().unwrap();

        assert!(logger.jsonl_path().map(|p| p.exists()).unwrap_or(false));
        assert!(logger.tfevents_path().map(|p| p.exists()).unwrap_or(false));
    }

    #[test]
    fn varint_encoding_roundtrip() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 300);
        // 300 = 0b100101100 → [0xAC, 0x02]
        assert_eq!(buf, vec![0xAC, 0x02]);
    }

    #[test]
    fn crc32c_known_value() {
        // CRC32C of empty string is 0x00000000
        assert_eq!(crc32c(b""), 0x00000000);
        // CRC32C of b"123456789" should be 0xE3069283
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }
}
