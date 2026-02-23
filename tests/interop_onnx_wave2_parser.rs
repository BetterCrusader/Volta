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
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        )),
    );
    let value = execute_value_with_context(&program.graph, program.output, &context)
        .expect("reshape execute should work");
    assert_eq!(
        value,
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap()
        ))
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
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap(),
        )),
    );
    context.inputs.insert(
        "x2".to_string(),
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![1, 2], vec![3.0, 4.0]).unwrap(),
        )),
    );
    let value = execute_value_with_context(&program.graph, program.output, &context)
        .expect("concat/gather/slice execute should work");
    assert_eq!(
        value,
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![1, 1], vec![2.0]).unwrap()
        ))
    );
}

#[test]
fn parses_reducemean_with_axes_input_and_executes() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "reduce_mean_wave2".to_string(),
            input: vec![tensor_value_info(
                "x",
                &[2, 2],
                pb::tensor_proto::DataType::Float,
            )],
            output: vec![tensor_value_info(
                "y",
                &[2],
                pb::tensor_proto::DataType::Float,
            )],
            initializer: vec![int64_tensor("axes", &[1], &[1])],
            node: vec![pb::NodeProto {
                name: "reduce_mean_node".to_string(),
                op_type: "ReduceMean".to_string(),
                input: vec!["x".to_string(), "axes".to_string()],
                output: vec!["y".to_string()],
                attribute: vec![pb::AttributeProto {
                    name: "keepdims".to_string(),
                    i: 0,
                    r#type: pb::attribute_proto::AttributeType::Int as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let program =
        import_onnx_bytes(&model.encode_to_vec()).expect("reduce mean import should work");
    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![2, 2], vec![1.0, 3.0, 2.0, 6.0]).unwrap(),
        )),
    );
    let value = execute_value_with_context(&program.graph, program.output, &context)
        .expect("reduce mean execute should work");
    assert_eq!(
        value,
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![2], vec![2.0, 4.0]).unwrap()
        ))
    );
}

#[test]
fn parses_reducemax_keepdims_one_for_wave2_static_import() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "reduce_max_keepdims_reject".to_string(),
            input: vec![tensor_value_info(
                "x",
                &[2, 2],
                pb::tensor_proto::DataType::Float,
            )],
            output: vec![tensor_value_info(
                "y",
                &[2],
                pb::tensor_proto::DataType::Float,
            )],
            node: vec![pb::NodeProto {
                name: "reduce_max_node".to_string(),
                op_type: "ReduceMax".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                attribute: vec![
                    pb::AttributeProto {
                        name: "axis".to_string(),
                        i: 1,
                        r#type: pb::attribute_proto::AttributeType::Int as i32,
                        ..Default::default()
                    },
                    pb::AttributeProto {
                        name: "keepdims".to_string(),
                        i: 1,
                        r#type: pb::attribute_proto::AttributeType::Int as i32,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let program = import_onnx_bytes(&model.encode_to_vec()).expect("keepdims=1 should import");
    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![2, 2], vec![1.0, 4.0, 3.0, 2.0]).unwrap(),
        )),
    );
    let value = execute_value_with_context(&program.graph, program.output, &context)
        .expect("reduce max keepdims execute should work");
    assert_eq!(
        value,
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![2, 1], vec![4.0, 3.0]).unwrap()
        ))
    );
}

#[test]
fn rejects_reducemean_invalid_keepdims_value() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "reduce_mean_bad_keepdims".to_string(),
            input: vec![tensor_value_info(
                "x",
                &[2, 2],
                pb::tensor_proto::DataType::Float,
            )],
            output: vec![tensor_value_info(
                "y",
                &[2],
                pb::tensor_proto::DataType::Float,
            )],
            node: vec![pb::NodeProto {
                name: "reduce_mean_node".to_string(),
                op_type: "ReduceMean".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                attribute: vec![pb::AttributeProto {
                    name: "keepdims".to_string(),
                    i: 2,
                    r#type: pb::attribute_proto::AttributeType::Int as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let err = import_onnx_bytes(&model.encode_to_vec()).expect_err("invalid keepdims must fail");
    assert!(err.message.contains("keepdims must be 0 or 1"));
}

#[test]
fn parses_reducemean_with_negative_axis_attribute() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "reduce_mean_negative_axis".to_string(),
            input: vec![tensor_value_info(
                "x",
                &[2, 2],
                pb::tensor_proto::DataType::Float,
            )],
            output: vec![tensor_value_info(
                "y",
                &[2],
                pb::tensor_proto::DataType::Float,
            )],
            node: vec![pb::NodeProto {
                name: "reduce_mean_node".to_string(),
                op_type: "ReduceMean".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                attribute: vec![
                    pb::AttributeProto {
                        name: "axis".to_string(),
                        i: -1,
                        r#type: pb::attribute_proto::AttributeType::Int as i32,
                        ..Default::default()
                    },
                    pb::AttributeProto {
                        name: "keepdims".to_string(),
                        i: 0,
                        r#type: pb::attribute_proto::AttributeType::Int as i32,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let program =
        import_onnx_bytes(&model.encode_to_vec()).expect("negative axis import should work");
    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![2, 2], vec![2.0, 4.0, 6.0, 8.0]).unwrap(),
        )),
    );
    let value = execute_value_with_context(&program.graph, program.output, &context)
        .expect("reduce mean execute should work");
    assert_eq!(
        value,
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![2], vec![3.0, 7.0]).unwrap()
        ))
    );
}

#[test]
fn rejects_negative_axis_out_of_bounds() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "reduce_negative_axis_oob".to_string(),
            input: vec![tensor_value_info(
                "x",
                &[2, 2],
                pb::tensor_proto::DataType::Float,
            )],
            output: vec![tensor_value_info(
                "y",
                &[2],
                pb::tensor_proto::DataType::Float,
            )],
            node: vec![pb::NodeProto {
                name: "reduce_sum_node".to_string(),
                op_type: "ReduceSum".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                attribute: vec![
                    pb::AttributeProto {
                        name: "axis".to_string(),
                        i: -3,
                        r#type: pb::attribute_proto::AttributeType::Int as i32,
                        ..Default::default()
                    },
                    pb::AttributeProto {
                        name: "keepdims".to_string(),
                        i: 0,
                        r#type: pb::attribute_proto::AttributeType::Int as i32,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let err = import_onnx_bytes(&model.encode_to_vec())
        .expect_err("out-of-bounds negative axis must fail");
    assert!(err.message.contains("axis -3 is out of bounds for rank 2"));
}

#[test]
fn parses_gelu_with_tanh_approximation_and_executes() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "gelu_tanh_wave2".to_string(),
            input: vec![tensor_value_info(
                "x",
                &[1, 3],
                pb::tensor_proto::DataType::Float,
            )],
            output: vec![tensor_value_info(
                "y",
                &[1, 3],
                pb::tensor_proto::DataType::Float,
            )],
            node: vec![pb::NodeProto {
                name: "gelu_node".to_string(),
                op_type: "Gelu".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                attribute: vec![pb::AttributeProto {
                    name: "approximate".to_string(),
                    s: b"tanh".to_vec(),
                    r#type: pb::attribute_proto::AttributeType::String as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let program = import_onnx_bytes(&model.encode_to_vec()).expect("gelu import should work");
    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![1, 3], vec![-1.0, 0.0, 1.0]).unwrap(),
        )),
    );
    let value = execute_value_with_context(&program.graph, program.output, &context)
        .expect("gelu execute should work");
    let RuntimeValue::Tensor(out) = value else {
        panic!("expected tensor output");
    };
    let expected = volta::ir::Tensor::new(vec![1, 3], vec![-1.0, 0.0, 1.0])
        .unwrap()
        .gelu()
        .unwrap();
    assert_eq!(out.shape, expected.shape);
    for (got, exp) in out.data.iter().zip(expected.data.iter()) {
        assert!((got - exp).abs() < 1e-6);
    }
}

#[test]
fn parses_gelu_with_none_approximation_as_exact_and_executes() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "gelu_none_exact".to_string(),
            input: vec![tensor_value_info(
                "x",
                &[1, 2],
                pb::tensor_proto::DataType::Float,
            )],
            output: vec![tensor_value_info(
                "y",
                &[1, 2],
                pb::tensor_proto::DataType::Float,
            )],
            node: vec![pb::NodeProto {
                name: "gelu_node".to_string(),
                op_type: "Gelu".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                attribute: vec![pb::AttributeProto {
                    name: "approximate".to_string(),
                    s: b"none".to_vec(),
                    r#type: pb::attribute_proto::AttributeType::String as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let program = import_onnx_bytes(&model.encode_to_vec()).expect("gelu none should import");
    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor(std::sync::Arc::new(
            volta::ir::Tensor::new(vec![1, 2], vec![-1.0, 1.0]).unwrap(),
        )),
    );
    let value = execute_value_with_context(&program.graph, program.output, &context)
        .expect("gelu none execute should work");
    let RuntimeValue::Tensor(out) = value else {
        panic!("expected tensor output");
    };
    let expected = volta::ir::Tensor::new(vec![1, 2], vec![-1.0, 1.0])
        .unwrap()
        .gelu_exact()
        .unwrap();
    assert_eq!(out.shape, expected.shape);
    for (got, exp) in out.data.iter().zip(expected.data.iter()) {
        assert!((got - exp).abs() < 1e-5);
    }
}

#[test]
fn rejects_gelu_with_unknown_approximation_mode() {
    let model = pb::ModelProto {
        ir_version: 8,
        graph: Some(pb::GraphProto {
            name: "gelu_unknown_reject".to_string(),
            input: vec![tensor_value_info(
                "x",
                &[1, 2],
                pb::tensor_proto::DataType::Float,
            )],
            output: vec![tensor_value_info(
                "y",
                &[1, 2],
                pb::tensor_proto::DataType::Float,
            )],
            node: vec![pb::NodeProto {
                name: "gelu_node".to_string(),
                op_type: "Gelu".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                attribute: vec![pb::AttributeProto {
                    name: "approximate".to_string(),
                    s: b"weird".to_vec(),
                    r#type: pb::attribute_proto::AttributeType::String as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let err = import_onnx_bytes(&model.encode_to_vec()).expect_err("unknown gelu mode must fail");
    assert!(err.message.contains("supported: tanh|none"));
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
