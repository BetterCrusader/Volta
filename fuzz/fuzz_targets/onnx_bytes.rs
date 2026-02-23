#![no_main]

use libfuzzer_sys::fuzz_target;
use volta::interop::import_onnx_bytes;

fuzz_target!(|data: &[u8]| {
    let _ = import_onnx_bytes(data);
});
