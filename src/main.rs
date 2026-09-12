use std::env;

use anyhow::{bail, Result};

use convmat::backend::BackendKind;
use convmat::frontend::SourceFile;
use convmat::pipeline;

const USAGE: &str = "usage: convmat [--backend <c|llvm|gpu>] <input.m>";

fn main() -> Result<()> {
    let mut args = env::args().skip(1);

    let mut backend = BackendKind::C;
    let mut input: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                let Some(value) = args.next() else {
                    bail!("{USAGE}");
                };
                backend = match value.as_str() {
                    "c" => BackendKind::C,
                    "llvm" => BackendKind::Llvm,
                    "gpu" => BackendKind::Gpu,
                    other => bail!("unknown backend `{other}` (expected c, llvm, or gpu)"),
                };
            }
            _ if arg.starts_with('-') && arg != "-" => bail!("{USAGE}"),
            _ => {
                if input.is_some() {
                    bail!("{USAGE}");
                }
                input = Some(arg);
            }
        }
    }

    let Some(path) = input else {
        bail!("{USAGE}");
    };

    let source = SourceFile::read(&path)?;
    let c = pipeline::compile(&source, backend)?;
    print!("{c}");

    Ok(())
}
