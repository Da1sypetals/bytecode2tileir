use itertools::Itertools;

use crate::interpreter::data_structures::value::Value;
use std::collections::HashMap;

pub struct Interpreter {
    /// Value map: IR SSA Id -> Runtime Value
    values: HashMap<u64, Value>,
    /// Number of tile blocks (from launch config)
    num_tile_blocks: (u32, u32, u32),
    /// Global variables
    globals: HashMap<String, Value>,
}

impl Interpreter {
    pub fn grid(&self) -> Vec<(u32, u32, u32)> {
        return (0..self.num_tile_blocks.2)
            .cartesian_product(0..self.num_tile_blocks.1)
            .cartesian_product(0..self.num_tile_blocks.0)
            .map(|((z, y), x)| (z, y, x))
            .collect();
    }
}
