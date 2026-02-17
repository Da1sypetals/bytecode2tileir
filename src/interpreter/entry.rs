use std::path::Path;

use ndrange::ndrange;

use crate::decode::{DecodeOptions, decode_module};
use crate::interpreter::data_structures::{
    interpreter::{ExecutionContext, Interpreter},
    value::Value,
};

impl Interpreter {
    /// Load a module from a bytecode file and initialize the interpreter.
    ///
    /// # Arguments
    /// * `path` - Path to the bytecode file
    /// * `grid_size` - Grid dimensions (x, y, z) for tile block execution
    ///
    /// # Panics
    /// Panics if the file cannot be read or parsed.
    pub fn from_module<P: AsRef<Path>>(path: P) -> Self {
        let data = std::fs::read(path).expect("Failed to read bytecode file");
        let opts = DecodeOptions {
            lazy_functions: false,
            attach_debug: false,
            keep_const_refs: true,
        };
        let module = decode_module(&data, &opts).expect("Failed to decode bytecode module");
        Interpreter::new(module.arena, module.consts)
    }

    pub fn execute(&mut self, args: Vec<Value>, grid_size: [usize; 3]) {
        // Assume the first function is the entry point
        let func_id = crate::cuda_tile_ir::ids::FunctionId(0);
        let func = self.arena.function_(func_id);
        let region = self.arena.region_(func.body);

        for [ix, iy, iz] in ndrange(&grid_size) {
            let mut ctx = ExecutionContext::new(
                self.arena(),
                [ix, iy, iz],
                grid_size.map(|x| x as u32),
                &self.globals,
                &self.consts,
            );

            // Iterate through each block in the region
            for &block_id in &region.blocks {
                let block = self.arena.block_(block_id);

                // Iterate through each op in the block
                for &op_id in &block.ops {
                    ctx.execute_op(op_id);
                }
            }
        }
    }
}
