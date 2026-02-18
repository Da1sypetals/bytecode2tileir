use std::path::Path;

use ndrange::ndrange;

use crate::cuda_tile_ir::ids::OpId;
use crate::cuda_tile_ir::Opcode;
use crate::decode::{DecodeOptions, decode_module};
use crate::interpreter::data_structures::{
    interpreter::{ExecutionContext, Interpreter},
    value::Value,
};

/// Control flow exception result for break/continue/yield/return operations
pub(crate) enum ControlFlow {
    /// Continue to next iteration with loop-carried values
    Continue(Vec<Value>),
    /// Break from loop with final values
    Break(Vec<Value>),
    /// Yield values from if/scan/reduce
    Yield(Vec<Value>),
    /// Return from function
    /// Kernel does not return anything
    Return,
}

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

        // Get entry block's args (function parameters)
        let entry_block_id = region.blocks[0];
        let entry_block = self.arena.block_(entry_block_id);
        let entry_block_args = &entry_block.args;

        for [ix, iy, iz] in ndrange(&grid_size) {
            let mut ctx = ExecutionContext::new(
                self.arena(),
                [ix, iy, iz],
                grid_size.map(|x| x as u32),
                &self.globals,
                &self.consts,
                entry_block_args,
                &args,
            );

            // Use unified region execution
            let _ = ctx.execute_region(func.body.0, &args);
        }
    }
}

impl ExecutionContext<'_> {
    /// Execute a region with given block arguments.
    /// Returns the control flow result if a terminator operation was encountered.
    ///
    /// # Parameters
    /// * `region_id` - ID of the region to execute
    /// * `block_args` - Values to bind to the block's arguments
    ///
    /// # Behavior
    /// - For control flow regions (if/for/loop): Single block, executes until terminator
    /// - For top-level regions: May have multiple blocks, executes all sequentially
    pub(crate) fn execute_region(
        &mut self,
        region_id: u32,
        block_args: &[Value],
    ) -> Option<ControlFlow> {
        let region = self
            .arena
            .region_(crate::cuda_tile_ir::ids::RegionId(region_id));

        // Execute each block in the region
        for &block_id in &region.blocks {
            let block = self.arena.block_(block_id);

            // Bind block arguments to the provided values
            for (&arg_id, value) in block.args.iter().zip(block_args.iter()) {
                self.set_value(arg_id, value.clone());
            }

            // Execute operations in the block
            for &op_id in &block.ops {
                let result = self.execute_op_with_cf(op_id);
                if let Some(cf_result) = result {
                    return Some(cf_result);
                }
            }
        }

        None
    }

    /// Execute an operation and return a control flow result if one occurs.
    pub(crate) fn execute_op_with_cf(&mut self, op_id: OpId) -> Option<ControlFlow> {
        let op = self.arena.op_(op_id);

        match op.opcode {
            Opcode::Break => {
                let operands: Vec<Value> = op
                    .operands
                    .iter()
                    .map(|&id| self.get_value(id).clone())
                    .collect();
                Some(ControlFlow::Break(operands))
            }
            Opcode::Continue => {
                let operands: Vec<Value> = op
                    .operands
                    .iter()
                    .map(|&id| self.get_value(id).clone())
                    .collect();
                Some(ControlFlow::Continue(operands))
            }
            Opcode::Yield => {
                let operands: Vec<Value> = op
                    .operands
                    .iter()
                    .map(|&id| self.get_value(id).clone())
                    .collect();
                Some(ControlFlow::Yield(operands))
            }
            Opcode::Return => Some(ControlFlow::Return),
            _ => {
                // Delegate to the main execute_op
                self.execute_op(op_id);
                None
            }
        }
    }
}
