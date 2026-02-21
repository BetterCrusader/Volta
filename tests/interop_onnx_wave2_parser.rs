#![cfg(feature = "onnx-import")]

use prost::Message;
use tract_onnx::pb;
use volta::interop::import_onnx_bytes;
use volta::ir::{ExecutionContext, RuntimeValue, execute_value_with_context};

#[test]
fn parses_reshape_and_executes() {
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

    let program = import_onnx_bytes(&model.encode_to_vec()).expect("reshape import should work");
    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor {
            shape: vec![1, 4],
            data: vec![1.0, 2.0, 3.0, 4.0],
        },
    );
    let value = execute_value_with_context(&program.graph, program.output, &context)
        .expect("reshape execute should work");
    assert_eq!(
        value,
        RuntimeValue::Tensor {
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        }
    );
}

#[test]
fn parses_concat_gather_slice_and_executes() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "concat_gather_slice_wave2".to_string(),
            input: vec![
                tensor_value_info("x1", &[1, 2], pb::tensor_proto::DataType::Float),
                tensor_value_info("x2", &[1, 2], pb::tensor_proto::DataType::Float),
            ],
            output: vec![tensor_value_info(
                "y",
                &[1, 1],
                pb::tensor_proto::DataType::Float,
            )],
            initializer: vec![
                int64_tensor("idx", &[1], &[1]),
                int64_tensor("starts", &[1], &[0]),
                int64_tensor("ends", &[1], &[1]),
                int64_tensor("axes", &[1], &[1]),
            ],
            node: vec![
                pb::NodeProto {
                    name: "concat_node".to_string(),
                    op_type: "Concat".to_string(),
                    input: vec!["x1".to_string(), "x2".to_string()],
                    output: vec!["concat_out".to_string()],
                    attribute: vec![pb::AttributeProto {
                        name: "axis".to_string(),
                        i: 1,
                        r#type: pb::attribute_proto::AttributeType::Int as i32,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                pb::NodeProto {
                    name: "gather_node".to_string(),
                    op_type: "Gather".to_string(),
                    input: vec!["concat_out".to_string(), "idx".to_string()],
                    output: vec!["gather_out".to_string()],
                    attribute: vec![pb::AttributeProto {
                        name: "axis".to_string(),
                        i: 1,
                        r#type: pb::attribute_proto::AttributeType::Int as i32,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                pb::NodeProto {
                    name: "slice_node".to_string(),
                    op_type: "Slice".to_string(),
                    input: vec![
                        "gather_out".to_string(),
                        "starts".to_string(),
                        "ends".to_string(),
                        "axes".to_string(),
                    ],
                    output: vec!["y".to_string()],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }),
        ..Default::default()
    };

    let program =
        import_onnx_bytes(&model.encode_to_vec()).expect("concat/gather/slice import should work");
    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x1".to_string(),
        RuntimeValue::Tensor {
            shape: vec![1, 2],
            data: vec![1.0, 2.0],
        },
    );
    context.inputs.insert(
        "x2".to_string(),
        RuntimeValue::Tensor {
            shape: vec![1, 2],
            data: vec![3.0, 4.0],
        },
    );
    let value = execute_value_with_context(&program.graph, program.output, &context)
        .expect("concat/gather/slice execute should work");
    assert_eq!(
        value,
        RuntimeValue::Tensor {
            shape: vec![1, 1],
            data: vec![2.0],
        }
    );
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
