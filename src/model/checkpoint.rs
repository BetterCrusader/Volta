use std::collections::HashMap;
use std::fs;

use crate::ir::Tensor;

use crate::model::TrainApiError;

pub fn save_checkpoint(
    path: &str,
    parameters: &HashMap<String, Tensor>,
) -> Result<(), TrainApiError> {
    let mut entries = parameters.iter().collect::<Vec<_>>();
    entries.sort_by(|(a, _), (b, _)| a.cmp(b));

    let mut lines = Vec::new();
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
    for (line_no, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
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
        parameters.insert(name, tensor);
    }

    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ir::Tensor;
    use crate::model::{load_checkpoint, save_checkpoint};

    #[test]
    fn roundtrip_checkpoint_is_deterministic() {
        let path = "C:\\Users\\User\\Desktop\\Volta\\target\\checkpoint_test.txt";
        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            Tensor::new(vec![1, 2], vec![1.0, 2.0]).expect("tensor"),
        );

        save_checkpoint(path, &params).expect("save should pass");
        let loaded = load_checkpoint(path).expect("load should pass");
        assert_eq!(loaded.get("w").expect("exists").shape, vec![1, 2]);
    }
}
