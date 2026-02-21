#![cfg(feature = "onnx-import")]

use volta::interop::{OnnxGraphStub, OnnxNodeStub, OnnxOpStub, import_onnx_stub_graph};

#[test]
fn wave2_stub_graphs_fail_with_explicit_runtime_todo_messages() {
    let scenarios = vec![
        (
            "reshape",
            OnnxOpStub::Reshape {
                input: "x".to_string(),
                shape: vec![2, 2],
            },
            "Reshape",
        ),
        (
            "concat",
            OnnxOpStub::Concat {
                inputs: vec!["x".to_string(), "x2".to_string()],
                axis: 1,
            },
            "Concat",
        ),
        (
            "gather",
            OnnxOpStub::Gather {
                input: "x".to_string(),
                indices: vec![0, 2],
                axis: 0,
            },
            "Gather",
        ),
        (
            "slice",
            OnnxOpStub::Slice {
                input: "x".to_string(),
                starts: vec![1],
                ends: vec![3],
                axes: vec![1],
            },
            "Slice",
        ),
    ];

    for (name, op, expected) in scenarios {
        let mut nodes = vec![
            OnnxNodeStub {
                id: "x".to_string(),
                op: OnnxOpStub::Input {
                    name: "x".to_string(),
                    shape: vec![1, 4],
                },
            },
            OnnxNodeStub {
                id: "x2".to_string(),
                op: OnnxOpStub::Input {
                    name: "x2".to_string(),
                    shape: vec![1, 4],
                },
            },
            OnnxNodeStub {
                id: "wave2".to_string(),
                op,
            },
            OnnxNodeStub {
                id: "out".to_string(),
                op: OnnxOpStub::Output {
                    value: "wave2".to_string(),
                },
            },
        ];

        if name == "reshape" || name == "gather" || name == "slice" {
            nodes.remove(1);
        }

        let graph = OnnxGraphStub {
            name: format!("wave2_{name}"),
            nodes,
        };

        let err = import_onnx_stub_graph(&graph).expect_err("Wave2 op must fail loudly for now");
        assert!(
            err.message.contains(expected),
            "unexpected message: {}",
            err.message
        );
        assert!(
            err.message.contains("not lowered"),
            "unexpected message: {}",
            err.message
        );
    }
}
