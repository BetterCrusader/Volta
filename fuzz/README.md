# Volta Fuzz Targets

## Targets

- `lexer` - tokenization path for arbitrary UTF-8 input.
- `parser` - lexer + parser pipeline for arbitrary UTF-8 input.
- `onnx_bytes` - ONNX protobuf importer over arbitrary bytes.

## Usage

Install cargo-fuzz once:

```bash
cargo install cargo-fuzz
```

Run from repository root (nightly required):

```bash
cargo +nightly fuzz run lexer
cargo +nightly fuzz run parser
cargo +nightly fuzz run onnx_bytes
```

## Windows (MSVC) notes

On Windows, libFuzzer ASAN runtime DLLs come from Visual Studio. Export the VS tool
directory in `PATH` before running targets:

```bash
export PATH="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64:$PATH"
```

5-minute gate (100s per target, 300s total):

```bash
cargo +nightly fuzz run lexer -- -max_total_time=100
cargo +nightly fuzz run parser -- -max_total_time=100
cargo +nightly fuzz run onnx_bytes -- -max_total_time=100
```

The fuzz crate enables Volta's `onnx-import` feature so ONNX importer code is included in coverage.
