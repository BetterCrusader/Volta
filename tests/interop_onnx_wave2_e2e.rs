#![cfg(feature = "onnx-import")]

use volta::interop::{OnnxGraphStub, OnnxNodeStub, OnnxOpStub, import_onnx_stub_graph};
use volta::ir::{ExecutionContext, RuntimeValue, execute_value_with_context};

#[test]
fn wave2_stub_graph_executes_for_reshape_concat_gather_slice() {
    let graph = OnnxGraphStub {
        name: "wave2_e2e".to_string(),
        nodes: vec![
            OnnxNodeStub {
                id: "x1".to_string(),
                op: OnnxOpStub::Input {
                    name: "x1".to_string(),
                    shape: vec![1, 2],
                },
            },
            OnnxNodeStub {
                id: "x2".to_string(),
                op: OnnxOpStub::Input {
                    name: "x2".to_string(),
                    shape: vec![1, 2],
                },
            },
            OnnxNodeStub {
                id: "cat".to_string(),
                op: OnnxOpStub::Concat {
                    inputs: vec!["x1".to_string(), "x2".to_string()],
                    axis: 1,
                },
            },
            OnnxNodeStub {
                id: "reshape".to_string(),
                op: OnnxOpStub::Reshape {
                    input: "cat".to_string(),
                    shape: vec![2, 2],
                },
            },
            OnnxNodeStub {
                id: "gather".to_string(),
                op: OnnxOpStub::Gather {
                    input: "reshape".to_string(),
                    indices: vec![1],
                    axis: 0,
                },
            },
            OnnxNodeStub {
                id: "slice".to_string(),
                op: OnnxOpStub::Slice {
                    input: "gather".to_string(),
                    starts: vec![0],
                    ends: vec![1],
                    axes: vec![1],
                },
            },
            OnnxNodeStub {
                id: "out".to_string(),
                op: OnnxOpStub::Output {
                    value: "slice".to_string(),
                },
            },
        ],
    };

    let program = import_onnx_stub_graph(&graph).expect("stub import should succeed");
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
        .expect("wave2 runtime execution should succeed");
    assert_eq!(
        value,
        RuntimeValue::Tensor {
            shape: vec![1, 1],
            data: vec![3.0],
        }
    );
}
