#![cfg(feature = "onnx-import")]

use volta::interop::{OnnxGraphStub, OnnxNodeStub, OnnxOpStub, import_onnx_stub_graph};
use volta::ir::{ExecutionContext, Graph, Op, RuntimeValue, execute_value_with_context};

#[test]
fn imported_stub_graph_matches_native_ir_execution() {
    let source = OnnxGraphStub {
        name: "parity_linear".to_string(),
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
    let (native_graph, native_output) = build_native_graph();

    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor {
            shape: vec![1, 2],
            data: vec![3.0, 1.0],
        },
    );

    let imported_out = execute_value_with_context(&imported.graph, imported.output, &context)
        .expect("imported graph should execute");
    let native_out = execute_value_with_context(&native_graph, native_output, &context)
        .expect("native graph should execute");

    assert_eq!(imported_out, native_out);
}

fn build_native_graph() -> (Graph, volta::ir::ValueId) {
    let mut graph = Graph::new();
    let block = graph.create_block();
    let (_, x) = graph
        .add_op(block, Op::Input("x".to_string()))
        .expect("add x");
    let (_, w) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![2, 2],
                data: vec![0.5, -1.0, 1.5, 2.0],
            },
        )
        .expect("add w");
    let (_, mm) = graph.add_op(block, Op::MatMul(x, w)).expect("add mm");
    let (_, b) = graph
        .add_op(
            block,
            Op::ConstTensor {
                shape: vec![1, 2],
                data: vec![0.25, -0.75],
            },
        )
        .expect("add b");
    let (_, sum) = graph.add_op(block, Op::Add(mm, b)).expect("add sum");
    let (_, relu) = graph.add_op(block, Op::Relu(sum)).expect("add relu");
    graph.add_op(block, Op::Output(relu)).expect("add output");
    graph.bind_input_shape("x", vec![1, 2]);
    (graph, relu)
}
