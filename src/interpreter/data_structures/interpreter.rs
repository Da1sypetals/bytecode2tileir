use crate::interpreter::data_structures::value::Value;
use std::collections::HashMap;

pub struct InterpreterState {
    /// Value map: IR SSA Id -> Runtime Value
    values: HashMap<u64, Value>,
    /// Number of tile blocks (from launch config)
    num_tile_blocks: (i32, i32, i32),
    /// Global variables
    globals: HashMap<String, Value>,
}
