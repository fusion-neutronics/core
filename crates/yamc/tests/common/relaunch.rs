//! Run one test in a child process with an environment variable set.
//!
//! Some library paths are steered by a process-global environment variable
//! (`YAMC_GPU_LAUNCH_CHUNK`, for one). Setting it from a test would race every
//! other test thread in the binary, and `std::env::set_var` is `unsafe` for
//! exactly that reason. Instead the test re-runs itself as a child of the test
//! binary with the variable set on the child's environment only.

use std::process::Command;

/// Returns `true` when this process already has `key=value`, meaning the caller
/// is the child and should run its body. Otherwise re-runs `test_name` from
/// this test binary with `key=value` set, asserts the child passed exactly one
/// test, and returns `false` so the caller returns without running the body.
///
/// `test_name` must be the test function's name: a wrong name would run zero
/// tests and pass, which the one-test check turns into a failure.
pub fn in_child_with_env(test_name: &str, key: &str, value: &str) -> bool {
    if std::env::var(key).as_deref() == Ok(value) {
        return true;
    }
    let output = Command::new(std::env::current_exe().expect("path of the test binary"))
        .args([test_name, "--exact", "--include-ignored", "--nocapture"])
        .env(key, value)
        .output()
        .expect("re-run the test binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    eprint!("{stdout}");
    assert!(
        output.status.success(),
        "{test_name} failed in the child with {key}={value}: {}",
        output.status
    );
    assert!(
        stdout.contains("test result: ok. 1 passed"),
        "the child with {key}={value} did not run {test_name}"
    );
    false
}
