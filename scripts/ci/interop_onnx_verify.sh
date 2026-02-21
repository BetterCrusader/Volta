#!/usr/bin/env bash
set -euo pipefail

echo "[interop-onnx-verify] contract + importer unit tests (feature=onnx-import)"
cargo test --features onnx-import interop::contract::tests::contract_compile_rejects_unknown_input_id

echo "[interop-onnx-verify] acceptance tests"
cargo test --features onnx-import --test interop_onnx_linear
cargo test --features onnx-import --test interop_onnx_mlp
cargo test --features onnx-import --test interop_onnx_parser
cargo test --features onnx-import --test interop_roundtrip_parity
