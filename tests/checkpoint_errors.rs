/// Integration test: bad checkpoint error handling
///
/// Verifies that the Volta CLI returns a non-zero exit code and a meaningful
/// error message for three corrupted/wrong checkpoint cases:
///  1. Wrong magic bytes (not a VOLT file)
///  2. Wrong format version (supported: 1)
///  3. File does not exist
use std::io::Write;
use std::process::Command;

fn volta_bin() -> String {
    std::env::current_dir()
        .unwrap()
        .join("target")
        .join("debug")
        .join("volta.exe")
        .to_string_lossy()
        .to_string()
}

fn run_vt_script(script_path: &str) -> (i32, String) {
    let out = Command::new(volta_bin())
        .arg(script_path)
        .output()
        .expect("Failed to run volta binary");
    let code = out.status.code().unwrap_or(-1);
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (code, combined)
}

fn write_vt_load_script(tmp_dir: &std::path::Path, ckpt_path: &str) -> String {
    let script = format!(
        "model m\n    layers 4 8 3\n    activation \"softmax\"\nload m as \"{}\"\n",
        ckpt_path.replace('\\', "/")
    );
    let script_path = tmp_dir.join("bad_load.vt");
    std::fs::write(&script_path, script).unwrap();
    script_path.to_string_lossy().to_string()
}

#[test]
fn test_wrong_magic_rejected() {
    let tmp = tempdir();
    let ckpt = tmp.path().join("bad_magic.vt");
    // Write file that starts with "BADT" instead of "VOLT"
    let mut f = std::fs::File::create(&ckpt).unwrap();
    f.write_all(b"BADT\x01\x00\x00\x00").unwrap();
    f.write_all(&[0u8; 32]).unwrap(); // padding

    let script = write_vt_load_script(tmp.path(), &ckpt.to_string_lossy());
    let (code, output) = run_vt_script(&script);

    assert_ne!(code, 0, "Expected non-zero exit for wrong magic");
    assert!(
        output.contains("not a valid Volta checkpoint") || output.contains("bad magic"),
        "Expected bad magic error, got:\n{output}"
    );
}

#[test]
fn test_wrong_version_rejected() {
    let tmp = tempdir();
    let ckpt = tmp.path().join("bad_version.vt");
    let mut f = std::fs::File::create(&ckpt).unwrap();
    // Correct magic, but version = 99
    f.write_all(b"VOLT").unwrap();
    f.write_all(&99u32.to_le_bytes()).unwrap();
    f.write_all(&[0u8; 32]).unwrap();

    let script = write_vt_load_script(tmp.path(), &ckpt.to_string_lossy());
    let (code, output) = run_vt_script(&script);

    assert_ne!(code, 0, "Expected non-zero exit for wrong version");
    assert!(
        output.contains("not supported") || output.contains("version"),
        "Expected version error, got:\n{output}"
    );
}

#[test]
fn test_missing_checkpoint_rejected() {
    let tmp = tempdir();
    let ckpt = tmp.path().join("nonexistent.vt");
    // Don't create the file

    let script = write_vt_load_script(tmp.path(), &ckpt.to_string_lossy());
    let (code, output) = run_vt_script(&script);

    assert_ne!(code, 0, "Expected non-zero exit for missing checkpoint");
    assert!(
        output.contains("Failed to read") || output.contains("No such file") || output.contains("cannot find"),
        "Expected file-not-found error, got:\n{output}"
    );
}

/// Minimal tempdir helper (avoids pulling in `tempfile` crate)
struct TempDir(std::path::PathBuf);
impl TempDir {
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tempdir() -> TempDir {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
    let path = std::env::temp_dir().join(format!("volta_test_{ts}"));
    std::fs::create_dir_all(&path).unwrap();
    TempDir(path)
}
