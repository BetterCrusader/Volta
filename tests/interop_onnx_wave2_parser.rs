#![cfg(feature = "onnx-import")]

use prost::Message;
use tract_onnx::pb;
use volta::interop::import_onnx_bytes;

#[test]
fn parses_reshape_and_fails_with_lowering_todo() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "reshape_wave2".to_string(),
            input: vec![tensor_value_info(
                "x",
                &[1, 4],
                pb::tensor_proto::DataType::Float,
            )],
            output: vec![tensor_value_info(
                "y",
                &[2, 2],
                pb::tensor_proto::DataType::Float,
            )],
            initializer: vec![int64_tensor("shape", &[2], &[2, 2])],
            node: vec![pb::NodeProto {
                op_type: "Reshape".to_string(),
                input: vec!["x".to_string(), "shape".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let err = import_onnx_bytes(&model.encode_to_vec()).expect_err("reshape must fail loudly");
    assert!(err.message.contains("Reshape"));
    assert!(err.message.contains("not lowered"));
}

#[test]
fn parses_concat_and_fails_with_lowering_todo() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "concat_wave2".to_string(),
            input: vec![
                tensor_value_info("x1", &[1, 2], pb::tensor_proto::DataType::Float),
                tensor_value_info("x2", &[1, 2], pb::tensor_proto::DataType::Float),
            ],
            output: vec![tensor_value_info(
                "y",
                &[1, 4],
                pb::tensor_proto::DataType::Float,
            )],
            node: vec![pb::NodeProto {
                op_type: "Concat".to_string(),
                input: vec!["x1".to_string(), "x2".to_string()],
                output: vec!["y".to_string()],
                attribute: vec![pb::AttributeProto {
                    name: "axis".to_string(),
                    i: 1,
                    r#type: pb::attribute_proto::AttributeType::Int as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let err = import_onnx_bytes(&model.encode_to_vec()).expect_err("concat must fail loudly");
    assert!(err.message.contains("Concat"));
    assert!(err.message.contains("not lowered"));
}

#[test]
fn parses_gather_and_fails_with_lowering_todo() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "gather_wave2".to_string(),
            input: vec![tensor_value_info(
                "x",
                &[4],
                pb::tensor_proto::DataType::Float,
            )],
            output: vec![tensor_value_info(
                "y",
                &[2],
                pb::tensor_proto::DataType::Float,
            )],
            initializer: vec![int64_tensor("idx", &[2], &[0, 2])],
            node: vec![pb::NodeProto {
                op_type: "Gather".to_string(),
                input: vec!["x".to_string(), "idx".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let err = import_onnx_bytes(&model.encode_to_vec()).expect_err("gather must fail loudly");
    assert!(err.message.contains("Gather"));
    assert!(err.message.contains("not lowered"));
}

#[test]
fn parses_slice_and_fails_with_lowering_todo() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "slice_wave2".to_string(),
            input: vec![tensor_value_info(
                "x",
                &[1, 8],
                pb::tensor_proto::DataType::Float,
            )],
            output: vec![tensor_value_info(
                "y",
                &[1, 4],
                pb::tensor_proto::DataType::Float,
            )],
            initializer: vec![
                int64_tensor("starts", &[1], &[2]),
                int64_tensor("ends", &[1], &[6]),
                int64_tensor("axes", &[1], &[1]),
            ],
            node: vec![pb::NodeProto {
                op_type: "Slice".to_string(),
                input: vec![
                    "x".to_string(),
                    "starts".to_string(),
                    "ends".to_string(),
                    "axes".to_string(),
                ],
                output: vec!["y".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let err = import_onnx_bytes(&model.encode_to_vec()).expect_err("slice must fail loudly");
    assert!(err.message.contains("Slice"));
    assert!(err.message.contains("not lowered"));
}

fn tensor_value_info(
    name: &str,
    shape: &[i64],
    dtype: pb::tensor_proto::DataType,
) -> pb::ValueInfoProto {
    pb::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(pb::TypeProto {
            value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
                elem_type: dtype as i32,
                shape: Some(pb::TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|dim| pb::tensor_shape_proto::Dimension {
                            value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(*dim)),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn int64_tensor(name: &str, dims: &[i64], values: &[i64]) -> pb::TensorProto {
    pb::TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type: pb::tensor_proto::DataType::Int64 as i32,
        int64_data: values.to_vec(),
        ..Default::default()
    }
}
