#![cfg(feature = "onnx-import")]

use volta::interop::{
    IrContractVersion, IrDataType, IrGraphContract, IrNodeContract, IrOpContract, IrTensorSpec,
};
use volta::ir::{ExecutionContext, RuntimeValue, execute_value_with_context};

#[test]
fn wave2_contract_reshape_compiles_and_executes() {
    let contract = IrGraphContract {
        version: IrContractVersion::V1,
        name: "reshape_contract".to_string(),
        inputs: vec![IrTensorSpec {
            name: "x".to_string(),
            shape: vec![1, 4],
            dtype: IrDataType::F32,
        }],
        parameters: vec![],
        nodes: vec![
            IrNodeContract {
                id: "x".to_string(),
                op: IrOpContract::Input {
                    name: "x".to_string(),
                },
            },
            IrNodeContract {
                id: "reshape".to_string(),
                op: IrOpContract::Reshape {
                    input: "x".to_string(),
                    shape: vec![2, 2],
                },
            },
            IrNodeContract {
                id: "out".to_string(),
                op: IrOpContract::Output {
                    value: "reshape".to_string(),
                },
            },
        ],
    };

    let program = contract.compile().expect("reshape lowering should compile");
    let mut context = ExecutionContext::default();
    context.inputs.insert(
        "x".to_string(),
        RuntimeValue::Tensor {
            shape: vec![1, 4],
            data: vec![1.0, 2.0, 3.0, 4.0],
        },
    );
    let value = execute_value_with_context(&program.graph, program.output, &context)
        .expect("reshape should execute");
    assert_eq!(
        value,
        RuntimeValue::Tensor {
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        }
    );
}

#[test]
fn wave2_contract_concat_requires_multiple_inputs() {
    let contract = IrGraphContract {
        version: IrContractVersion::V1,
        name: "concat_validation".to_string(),
        inputs: vec![IrTensorSpec {
            name: "x".to_string(),
            shape: vec![1, 2],
            dtype: IrDataType::F32,
        }],
        parameters: vec![],
        nodes: vec![
            IrNodeContract {
                id: "x".to_string(),
                op: IrOpContract::Input {
                    name: "x".to_string(),
                },
            },
            IrNodeContract {
                id: "concat".to_string(),
                op: IrOpContract::Concat {
                    inputs: vec!["x".to_string()],
                    axis: 0,
                },
            },
            IrNodeContract {
                id: "out".to_string(),
                op: IrOpContract::Output {
                    value: "concat".to_string(),
                },
            },
        ],
    };

    let err = contract
        .compile()
        .expect_err("concat with one input must fail");
    assert!(err.message.contains("concat requires at least 2 inputs"));
}

#[test]
fn wave2_contract_slice_requires_matching_ranges() {
    let contract = IrGraphContract {
        version: IrContractVersion::V1,
        name: "slice_validation".to_string(),
        inputs: vec![IrTensorSpec {
            name: "x".to_string(),
            shape: vec![1, 8],
            dtype: IrDataType::F32,
        }],
        parameters: vec![],
        nodes: vec![
            IrNodeContract {
                id: "x".to_string(),
                op: IrOpContract::Input {
                    name: "x".to_string(),
                },
            },
            IrNodeContract {
                id: "slice".to_string(),
                op: IrOpContract::Slice {
                    input: "x".to_string(),
                    starts: vec![0, 1],
                    ends: vec![4],
                    axes: vec![1, 0],
                },
            },
            IrNodeContract {
                id: "out".to_string(),
                op: IrOpContract::Output {
                    value: "slice".to_string(),
                },
            },
        ],
    };

    let err = contract
        .compile()
        .expect_err("slice with mismatched range vectors must fail");
    assert!(err.message.contains("starts/ends/axes lengths must match"));
}
