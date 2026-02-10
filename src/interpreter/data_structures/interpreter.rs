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
    /// Create a new interpreter with the specified grid dimensions
    pub fn new(num_tile_blocks: (u32, u32, u32)) -> Self {
        Interpreter {
            values: HashMap::new(),
            num_tile_blocks,
            globals: HashMap::new(),
        }
    }

    /// Set a value in the register file (write-once per SSA semantics)
    pub fn set_value(&mut self, id: u64, value: Value) {
        if self.values.contains_key(&id) {
            panic!("Attempting to overwrite SSA value {}", id);
        }
        self.values.insert(id, value);
    }

    /// Get a value from the register file
    pub fn get_value(&self, id: u64) -> Option<&Value> {
        self.values.get(&id)
    }

    pub fn grid(&self) -> Vec<(u32, u32, u32)> {
        return (0..self.num_tile_blocks.2)
            .cartesian_product(0..self.num_tile_blocks.1)
            .cartesian_product(0..self.num_tile_blocks.0)
            .map(|((z, y), x)| (z, y, x))
            .collect();
    }
}
