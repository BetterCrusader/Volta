#![cfg(feature = "onnx-import")]

use volta::interop::{
    IrContractVersion, IrDataType, IrGraphContract, IrNodeContract, IrOpContract, IrTensorSpec,
};

#[test]
fn wave2_contract_reshape_is_parsed_but_not_lowered_yet() {
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

    let err = contract
        .compile()
        .expect_err("reshape lowering is not implemented");
    assert!(err.message.contains("Reshape"));
    assert!(err.message.contains("not lowered"));
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
