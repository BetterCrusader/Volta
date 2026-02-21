#![cfg(feature = "onnx-import")]

use volta::interop::{OnnxGraphStub, OnnxNodeStub, OnnxOpStub, import_onnx_stub_graph};
use volta::ir::{ExecutionContext, RuntimeValue, Tensor, execute_value_with_context};

#[test]
fn imports_linear_relu_graph_and_executes() {
    let source = OnnxGraphStub {
        name: "linear_relu".to_string(),
        nodes: vec![
            OnnxNodeStub {
                id: "x".to_string(),
                op: OnnxOpStub::Input {
                    name: "x".to_string(),
                    shape: vec![1, 2],
                },
            },
            OnnxNodeStub {
                id: "w".to_string(),
                op: OnnxOpStub::ConstTensor {
                    shape: vec![2, 2],
                    data: vec![0.5, -1.0, 1.5, 2.0],
                },
            },
            OnnxNodeStub {
                id: "mm".to_string(),
                op: OnnxOpStub::MatMul {
                    lhs: "x".to_string(),
                    rhs: "w".to_string(),
                },
            },
            OnnxNodeStub {
                id: "b".to_string(),
                op: OnnxOpStub::ConstTensor {
                    shape: vec![1, 2],
                    data: vec![0.25, -0.75],
                },
            },
            OnnxNodeStub {
                id: "sum".to_string(),
                op: OnnxOpStub::Add {
                    lhs: "mm".to_string(),
                    rhs: "b".to_string(),
                },
            },
            OnnxNodeStub {
                id: "relu".to_string(),
                op: OnnxOpStub::Relu {
                    input: "sum".to_string(),
                },
            },
            OnnxNodeStub {
                id: "out".to_string(),
                op: OnnxOpStub::Output {
                    value: "relu".to_string(),
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
            data: vec![3.0, 1.0],
        },
    );

    let output = execute_value_with_context(&imported.graph, imported.output, &context)
        .expect("execution should pass");
    let RuntimeValue::Tensor { shape, data } = output else {
        panic!("expected tensor output");
    };

    assert_eq!(shape, vec![1, 2]);
    assert_eq!(
        Tensor::new(shape, data).expect("valid tensor").data,
        vec![3.25, 0.0]
    );
}
