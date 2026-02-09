//! bytecode2mlir - Convert Tile IR bytecode to MLIR text format

use clap::Parser;
use std::fs;
use std::path::PathBuf;

use bytecode2mlir::cuda_tile_ir::printer::module_to_mlir_text;
use bytecode2mlir::decode::{decode_module, DecodeOptions};

#[derive(Parser)]
#[command(name = "bytecode2mlir")]
#[command(about = "Convert Tile IR bytecode to MLIR text format")]
struct Args {
    /// Input TileIR bytecode file
    input: PathBuf,

    /// Output MLIR file (default: <input_stem>.mlir in the same directory)
    output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let output = args.output.unwrap_or_else(|| {
        let mut p = args.input.clone();
        p.set_extension("mlir");
        p
    });

    // Check if it's a valid bytecode file
    let data = fs::read(&args.input)?;
    if !bytecode2mlir::bytecode::is_tilir_bytecode(&data) {
        eprintln!(
            "Error: '{}' is not a valid TileIR bytecode file",
            args.input.display()
        );
        std::process::exit(1);
    }

    // Parse bytecode to IR
    let opts = DecodeOptions {
        lazy_functions: false,
        ..Default::default()
    };

    let ir_module = decode_module(&data, &opts)?;

    // Convert to MLIR text
    let mlir_text = module_to_mlir_text(&ir_module);

    // Write to output file
    fs::write(&output, mlir_text)?;
    eprintln!("Successfully wrote MLIR to: {}", output.display());

    Ok(())
}
