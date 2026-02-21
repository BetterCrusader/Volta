#![cfg(feature = "onnx-import")]

use volta::interop::{OnnxGraphStub, OnnxNodeStub, OnnxOpStub, import_onnx_stub_graph};
use volta::ir::{ExecutionContext, RuntimeValue, execute_value_with_context};

#[test]
fn imports_two_layer_mlp_graph() {
    let source = OnnxGraphStub {
        name: "mlp_softmax".to_string(),
        nodes: vec![
            OnnxNodeStub {
                id: "x".to_string(),
                op: OnnxOpStub::Input {
                    name: "x".to_string(),
                    shape: vec![1, 2],
                },
            },
            OnnxNodeStub {
                id: "w1".to_string(),
                op: OnnxOpStub::ConstTensor {
                    shape: vec![2, 3],
                    data: vec![1.0, -1.0, 0.5, 0.5, 2.0, -0.5],
                },
            },
            OnnxNodeStub {
                id: "h1".to_string(),
                op: OnnxOpStub::MatMul {
                    lhs: "x".to_string(),
                    rhs: "w1".to_string(),
                },
            },
            OnnxNodeStub {
                id: "a1".to_string(),
                op: OnnxOpStub::Relu {
                    input: "h1".to_string(),
                },
            },
            OnnxNodeStub {
                id: "w2".to_string(),
                op: OnnxOpStub::ConstTensor {
                    shape: vec![3, 2],
                    data: vec![1.0, 0.0, 0.0, 1.0, 0.5, -0.5],
                },
            },
            OnnxNodeStub {
                id: "h2".to_string(),
                op: OnnxOpStub::MatMul {
                    lhs: "a1".to_string(),
                    rhs: "w2".to_string(),
                },
            },
            OnnxNodeStub {
                id: "out".to_string(),
                op: OnnxOpStub::Output {
                    value: "h2".to_string(),
                },
            },
        ],
    };

    let imported = import_onnx_stub_graph(&source).expect("import should pass");
    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor {
            shape: vec![1, 2],
            data: vec![2.0, 1.0],
        },
    );

    let output = execute_value_with_context(&imported.graph, imported.output, &context)
        .expect("execution should pass");
    let RuntimeValue::Tensor { shape, data } = output else {
        panic!("expected tensor output");
    };

    assert_eq!(shape, vec![1, 2]);
    assert_eq!(data.len(), 2);
    assert!(data.iter().all(|value| value.is_finite()));
    assert!(data[0] > data[1], "first channel should dominate second");
}
