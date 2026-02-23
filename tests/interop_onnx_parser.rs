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
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![1, 2], vec![3.0, 1.0]).unwrap(),
        )),
    );

    let out = execute_value_with_context(&imported.graph, imported.output, &context)
        .expect("imported ONNX graph should execute");
    let RuntimeValue::Tensor(tensor) = out else {
        panic!("expected tensor output");
    };
    assert_eq!(tensor.shape, vec![1, 2]);
    assert_eq!(tensor.data, vec![3.25, 0.0]);
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

#[test]
fn imports_gemm_with_transpose_attributes_and_executes() {
    let model = build_gemm_transpose_model_proto();
    let bytes = model.encode_to_vec();
    let imported = import_onnx_bytes(&bytes).expect("Gemm trans import should pass");

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )),
    );

    let out = execute_value_with_context(&imported.graph, imported.output, &context)
        .expect("imported Gemm graph should execute");
    let RuntimeValue::Tensor(tensor) = out else {
        panic!("expected tensor output");
    };
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.data, vec![6.5, 6.5, 8.5, 8.5]);
}

#[test]
fn rejects_static_shape_mismatch_during_onnx_import() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "matmul_shape_mismatch".to_string(),
            input: vec![pb::ValueInfoProto {
                name: "x".to_string(),
                r#type: Some(tensor_type(&[1, 3], pb::tensor_proto::DataType::Float)),
                ..Default::default()
            }],
            output: vec![pb::ValueInfoProto {
                name: "y".to_string(),
                r#type: Some(tensor_type(&[1, 2], pb::tensor_proto::DataType::Float)),
                ..Default::default()
            }],
            initializer: vec![pb::TensorProto {
                name: "w".to_string(),
                dims: vec![2, 2],
                data_type: pb::tensor_proto::DataType::Float as i32,
                float_data: vec![1.0, 0.0, 0.0, 1.0],
                ..Default::default()
            }],
            node: vec![pb::NodeProto {
                op_type: "MatMul".to_string(),
                input: vec!["x".to_string(), "w".to_string()],
                output: vec!["y".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let err =
        import_onnx_bytes(&model.encode_to_vec()).expect_err("shape mismatch must fail import");
    assert!(err.message.contains("import-time shape inference failed"));
    assert!(err.message.contains("Shape mismatch in MatMul"));
}

#[test]
fn imports_gemm_with_transpose_b_and_executes() {
    let model = build_gemm_model_proto(
        "gemm_trans_b",
        &[2, 3],
        &[2, 3],
        &[2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        None,
        1.0,
        1.0,
        0,
        1,
    );
    let imported =
        import_onnx_bytes(&model.encode_to_vec()).expect("gemm transB import should pass");

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )),
    );
    let out = execute_value_with_context(&imported.graph, imported.output, &context)
        .expect("gemm transB graph should execute");
    let RuntimeValue::Tensor(tensor) = out else {
        panic!("expected tensor output");
    };
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.data, vec![4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn imports_gemm_with_transpose_a_and_b_and_executes() {
    let model = build_gemm_model_proto(
        "gemm_trans_ab",
        &[3, 2],
        &[2, 3],
        &[2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0],
        None,
        1.0,
        1.0,
        1,
        1,
    );
    let imported =
        import_onnx_bytes(&model.encode_to_vec()).expect("gemm transA+transB import should pass");

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )),
    );
    let out = execute_value_with_context(&imported.graph, imported.output, &context)
        .expect("gemm transA+transB graph should execute");
    let RuntimeValue::Tensor(tensor) = out else {
        panic!("expected tensor output");
    };
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.data, vec![7.0, 8.0, 10.0, 10.0]);
}

#[test]
fn imports_gemm_with_alpha_beta_scaling_and_bias() {
    let model = build_gemm_model_proto(
        "gemm_alpha_beta",
        &[1, 2],
        &[2, 2],
        &[1, 2],
        vec![2.0, 3.0],
        vec![1.0, 2.0, 3.0, 4.0],
        Some(vec![1.0, 2.0]),
        0.5,
        2.0,
        0,
        0,
    );
    let imported =
        import_onnx_bytes(&model.encode_to_vec()).expect("gemm alpha/beta import should pass");

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![1, 2], vec![2.0, 3.0]).unwrap(),
        )),
    );
    let out = execute_value_with_context(&imported.graph, imported.output, &context)
        .expect("gemm alpha/beta graph should execute");
    let RuntimeValue::Tensor(tensor) = out else {
        panic!("expected tensor output");
    };
    assert_eq!(tensor.shape, vec![1, 2]);
    assert_eq!(tensor.data, vec![7.5, 12.0]);
}

#[test]
fn rejects_gemm_with_invalid_transpose_flag() {
    let model = build_gemm_model_proto(
        "gemm_bad_trans",
        &[1, 2],
        &[2, 2],
        &[1, 2],
        vec![1.0, 2.0],
        vec![1.0, 0.0, 0.0, 1.0],
        None,
        1.0,
        1.0,
        2,
        0,
    );
    let err = import_onnx_bytes(&model.encode_to_vec()).expect_err("invalid transA must fail");
    assert!(err.message.contains("transA must be 0 or 1"));
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

fn build_gemm_transpose_model_proto() -> pb::ModelProto {
    let input_x = pb::ValueInfoProto {
        name: "x".to_string(),
        r#type: Some(tensor_type(&[3, 2], pb::tensor_proto::DataType::Float)),
        ..Default::default()
    };
    let output_y = pb::ValueInfoProto {
        name: "y".to_string(),
        r#type: Some(tensor_type(&[2, 2], pb::tensor_proto::DataType::Float)),
        ..Default::default()
    };

    let init_w = pb::TensorProto {
        name: "w".to_string(),
        dims: vec![3, 2],
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ..Default::default()
    };
    let init_c = pb::TensorProto {
        name: "c".to_string(),
        dims: vec![2],
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: vec![0.5, -1.5],
        ..Default::default()
    };

    let gemm = pb::NodeProto {
        name: "gemm_node".to_string(),
        input: vec!["x".to_string(), "w".to_string(), "c".to_string()],
        output: vec!["y".to_string()],
        op_type: "Gemm".to_string(),
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
            pb::AttributeProto {
                name: "transA".to_string(),
                i: 1,
                r#type: pb::attribute_proto::AttributeType::Int as i32,
                ..Default::default()
            },
            pb::AttributeProto {
                name: "transB".to_string(),
                i: 0,
                r#type: pb::attribute_proto::AttributeType::Int as i32,
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            node: vec![gemm],
            name: "gemm_transpose_proto".to_string(),
            initializer: vec![init_w, init_c],
            input: vec![input_x],
            output: vec![output_y],
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn build_gemm_model_proto(
    name: &str,
    input_shape: &[i64],
    weight_shape: &[i64],
    output_shape: &[i64],
    input_data: Vec<f32>,
    weight_data: Vec<f32>,
    bias_data: Option<Vec<f32>>,
    alpha: f32,
    beta: f32,
    trans_a: i64,
    trans_b: i64,
) -> pb::ModelProto {
    let input_x = pb::ValueInfoProto {
        name: "x".to_string(),
        r#type: Some(tensor_type(input_shape, pb::tensor_proto::DataType::Float)),
        ..Default::default()
    };
    let output_y = pb::ValueInfoProto {
        name: "y".to_string(),
        r#type: Some(tensor_type(output_shape, pb::tensor_proto::DataType::Float)),
        ..Default::default()
    };

    let init_x = pb::TensorProto {
        name: "x_init".to_string(),
        dims: input_shape.to_vec(),
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: input_data,
        ..Default::default()
    };
    let init_w = pb::TensorProto {
        name: "w".to_string(),
        dims: weight_shape.to_vec(),
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: weight_data,
        ..Default::default()
    };

    let mut initializer = vec![init_w];
    let mut gemm_inputs = vec!["x".to_string(), "w".to_string()];
    if let Some(bias) = bias_data {
        let init_b = pb::TensorProto {
            name: "c".to_string(),
            dims: vec![bias.len() as i64],
            data_type: pb::tensor_proto::DataType::Float as i32,
            float_data: bias,
            ..Default::default()
        };
        initializer.push(init_b);
        gemm_inputs.push("c".to_string());
    }

    let gemm = pb::NodeProto {
        name: "gemm_node".to_string(),
        input: gemm_inputs,
        output: vec!["y".to_string()],
        op_type: "Gemm".to_string(),
        attribute: vec![
            pb::AttributeProto {
                name: "alpha".to_string(),
                f: alpha,
                r#type: pb::attribute_proto::AttributeType::Float as i32,
                ..Default::default()
            },
            pb::AttributeProto {
                name: "beta".to_string(),
                f: beta,
                r#type: pb::attribute_proto::AttributeType::Float as i32,
                ..Default::default()
            },
            pb::AttributeProto {
                name: "transA".to_string(),
                i: trans_a,
                r#type: pb::attribute_proto::AttributeType::Int as i32,
                ..Default::default()
            },
            pb::AttributeProto {
                name: "transB".to_string(),
                i: trans_b,
                r#type: pb::attribute_proto::AttributeType::Int as i32,
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    // Keep input as runtime-fed tensor, not initializer; `init_x` is only there to
    // prevent accidental dead-code optimizations in some importers.
    let _ = init_x;

    pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            node: vec![gemm],
            name: name.to_string(),
            initializer,
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
