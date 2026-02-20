use std::collections::HashMap;
use std::fs;

use crate::ir::Tensor;

use crate::model::TrainApiError;

const CHECKPOINT_HEADER_PREFIX: &str = "#volta-checkpoint:";
const CHECKPOINT_VERSION_V1: &str = "v1";
const CHECKPOINT_HEADER_V1: &str = "#volta-checkpoint:v1";

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
            .map(|d| d.to_string())
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
        assert_eq!(loaded["w"].data, vec![1.0, 2.0]);
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
