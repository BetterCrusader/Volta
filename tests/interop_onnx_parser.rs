#![cfg(feature = "onnx-import")]

use prost::Message;
use tract_onnx::pb;
use volta::interop::import_onnx_bytes;
use volta::ir::{ExecutionContext, RuntimeValue, execute_value_with_context};

#[test]
fn imports_real_onnx_protobuf_linear_graph() {
    let model = build_linear_model_proto();
    let bytes = model.encode_to_vec();
    let imported = import_onnx_bytes(&bytes).expect("real ONNX import should pass");

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor {
            shape: vec![1, 2],
            data: vec![3.0, 1.0],
        },
    );

    let out = execute_value_with_context(&imported.graph, imported.output, &context)
        .expect("imported ONNX graph should execute");
    let RuntimeValue::Tensor { shape, data } = out else {
        panic!("expected tensor output");
    };
    assert_eq!(shape, vec![1, 2]);
    assert_eq!(data, vec![3.25, 0.0]);
}

#[test]
fn rejects_unsupported_onnx_op() {
    let mut model = build_linear_model_proto();
    model
        .graph
        .as_mut()
        .expect("graph")
        .node
        .push(pb::NodeProto {
            input: vec!["x".to_string(), "w".to_string()],
            output: vec!["conv".to_string()],
            op_type: "Conv".to_string(),
            ..Default::default()
        });

    let bytes = model.encode_to_vec();
    let err = import_onnx_bytes(&bytes).expect_err("unsupported op must fail");
    assert!(err.message.contains("unsupported ONNX op 'Conv'"));
}

fn build_linear_model_proto() -> pb::ModelProto {
    let input_x = pb::ValueInfoProto {
        name: "x".to_string(),
        r#type: Some(tensor_type(&[1, 2], pb::tensor_proto::DataType::Float)),
        ..Default::default()
    };
    let output_y = pb::ValueInfoProto {
        name: "y".to_string(),
        r#type: Some(tensor_type(&[1, 2], pb::tensor_proto::DataType::Float)),
        ..Default::default()
    };

    let init_w = pb::TensorProto {
        name: "w".to_string(),
        dims: vec![2, 2],
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: vec![0.5, -1.0, 1.5, 2.0],
        ..Default::default()
    };
    let init_b = pb::TensorProto {
        name: "b".to_string(),
        dims: vec![1, 2],
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: vec![0.25, -0.75],
        ..Default::default()
    };

    let matmul = pb::NodeProto {
        input: vec!["x".to_string(), "w".to_string()],
        output: vec!["mm".to_string()],
        op_type: "MatMul".to_string(),
        ..Default::default()
    };
    let add = pb::NodeProto {
        input: vec!["mm".to_string(), "b".to_string()],
        output: vec!["sum".to_string()],
        op_type: "Add".to_string(),
        ..Default::default()
    };
    let relu = pb::NodeProto {
        input: vec!["sum".to_string()],
        output: vec!["y".to_string()],
        op_type: "Relu".to_string(),
        ..Default::default()
    };

    pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            node: vec![matmul, add, relu],
            name: "linear_proto".to_string(),
            initializer: vec![init_w, init_b],
            input: vec![input_x],
            output: vec![output_y],
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
