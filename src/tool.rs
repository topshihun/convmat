//! Helpers for invoking the MLIR command-line tools.

use crate::error::{Error, Result};

/// Run an external tool, piping `input` to stdin and returning stdout.
pub(crate) fn run_tool(tool: &str, args: &[&str], input: &str) -> Result<String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut child = Command::new(tool)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Error::Backend(format!("failed to spawn `{tool}`: {e}")))?;

    child
        .stdin
        .take()
        .ok_or_else(|| Error::Backend(format!("failed to open stdin for `{tool}`")))?
        .write_all(input.as_bytes())
        .map_err(|e| Error::Backend(format!("failed to write to `{tool}`: {e}")))?;

    let output = child
        .wait_with_output()
        .map_err(|e| Error::Backend(format!("failed to wait for `{tool}`: {e}")))?;

    if !output.status.success() {
        return Err(Error::Backend(format!(
            "`{tool}` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}
