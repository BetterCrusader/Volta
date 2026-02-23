#![cfg(feature = "onnx-import")]

use prost::Message;
use tract_onnx::pb;
use volta::interop::import_onnx_bytes;
use volta::ir::{ExecutionContext, RuntimeValue, Tensor, execute_value_with_context};

#[test]
fn loads_generated_real_onnx_fixture_from_disk_and_executes() {
    let model = build_tiny_gemm_reduce_model();
    let bytes = model.encode_to_vec();

    let fixture_path = std::path::Path::new("target/test-fixtures/tiny-gemm-reduce.onnx");
    std::fs::create_dir_all(fixture_path.parent().expect("fixture dir"))
        .expect("create fixture directory");
    std::fs::write(fixture_path, &bytes).expect("write fixture model bytes");

    let fixture_bytes = std::fs::read(fixture_path).expect("read fixture model bytes");
    let imported = import_onnx_bytes(&fixture_bytes).expect("fixture ONNX import should pass");

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(std::sync::Arc::new(
            Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )),
    );

    let out = execute_value_with_context(&imported.graph, imported.output, &context)
        .expect("fixture model execution should pass");
    let RuntimeValue::Tensor(tensor) = out else {
        panic!("expected tensor output");
    };

    assert_eq!(tensor.shape, vec![2]);
    assert_eq!(tensor.data, vec![4.0, 10.0]);
    assert!(tensor.data.iter().all(|v| v.is_finite()));
}

fn build_tiny_gemm_reduce_model() -> pb::ModelProto {
    let input_x = pb::ValueInfoProto {
        name: "x".to_string(),
        r#type: Some(tensor_type(&[2, 3], pb::tensor_proto::DataType::Float)),
        ..Default::default()
    };
    let output_y = pb::ValueInfoProto {
        name: "y".to_string(),
        r#type: Some(tensor_type(&[2], pb::tensor_proto::DataType::Float)),
        ..Default::default()
    };

    let init_w = pb::TensorProto {
        name: "w".to_string(),
        dims: vec![3, 2],
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ..Default::default()
    };
    let init_b = pb::TensorProto {
        name: "b".to_string(),
        dims: vec![2],
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: vec![0.5, -1.5],
        ..Default::default()
    };
    let init_axes = pb::TensorProto {
        name: "axes".to_string(),
        dims: vec![1],
        data_type: pb::tensor_proto::DataType::Int64 as i32,
        int64_data: vec![1],
        ..Default::default()
    };

    let gemm = pb::NodeProto {
        name: "gemm_node".to_string(),
        op_type: "Gemm".to_string(),
        input: vec!["x".to_string(), "w".to_string(), "b".to_string()],
        output: vec!["gemm_out".to_string()],
        attribute: vec![
            pb::AttributeProto {
                name: "alpha".to_string(),
                f: 1.0,
                r#type: pb::attribute_proto::AttributeType::Float as i32,
                ..Default::default()
            },
            pb::AttributeProto {
                name: "beta".to_string(),
                f: 1.0,
                r#type: pb::attribute_proto::AttributeType::Float as i32,
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    let reduce_mean = pb::NodeProto {
        name: "reduce_mean_node".to_string(),
        op_type: "ReduceMean".to_string(),
        input: vec!["gemm_out".to_string(), "axes".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![pb::AttributeProto {
            name: "keepdims".to_string(),
            i: 0,
            r#type: pb::attribute_proto::AttributeType::Int as i32,
            ..Default::default()
        }],
        ..Default::default()
    };

    pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "tiny_gemm_reduce_fixture".to_string(),
            initializer: vec![init_w, init_b, init_axes],
            input: vec![input_x],
            output: vec![output_y],
            node: vec![gemm, reduce_mean],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn tensor_type(shape: &[i64], elem_type: pb::tensor_proto::DataType) -> pb::TypeProto {
    pb::TypeProto {
        value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
            elem_type: elem_type as i32,
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
    }
}
