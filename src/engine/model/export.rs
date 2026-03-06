use crate::ir::graph_fingerprint;
use crate::model::CompiledModel;

#[derive(Debug, Clone)]
pub struct ModelExportError {
    pub message: String,
}

pub fn export_compiled_model_manifest(model: &CompiledModel) -> Result<String, ModelExportError> {
    let mut param_names = model.parameters.keys().cloned().collect::<Vec<_>>();
    param_names.sort();

    let mut parameter_rows = Vec::new();
    for name in param_names {
        let Some(tensor) = model.parameters.get(&name) else {
            return Err(ModelExportError {
                message: format!("missing parameter tensor for '{name}'"),
            });
        };
        parameter_rows.push(format!(
            "{{\"name\":\"{}\",\"shape\":{:?}}}",
            escape_json(&name),
            tensor.shape
        ));
    }

    let output_shape = format!("{:?}", model.output_shape.0);
    let loss_value = model
        .loss
        .map_or_else(|| "null".to_string(), |value| value.0.to_string());

    Ok(format!(
        concat!(
            "{{",
            "\"graph_fingerprint\":{},",
            "\"node_count\":{},",
            "\"output_value\":{},",
            "\"loss_value\":{},",
            "\"output_shape\":{},",
            "\"parameters\":[{}]",
            "}}"
        ),
        graph_fingerprint(&model.graph),
        model.graph.nodes.len(),
        model.output.0,
        loss_value,
        output_shape,
        parameter_rows.join(",")
    ))
}

fn escape_json(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use crate::model::{build_tiny_transformer_fixture_for_tests, export_compiled_model_manifest};

    #[test]
    fn manifest_contains_expected_keys() {
        let (model, _dataset, _cfg, _infer_input) = build_tiny_transformer_fixture_for_tests();
        let manifest = export_compiled_model_manifest(&model).expect("export should pass");

        assert!(manifest.contains("\"graph_fingerprint\""));
        assert!(manifest.contains("\"parameters\""));
        assert!(manifest.contains("\"output_shape\""));
    }
}
