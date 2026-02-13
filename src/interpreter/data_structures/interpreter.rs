use std::collections::HashMap;

use crate::cuda_tile_ir::arena::IrArena;
use crate::cuda_tile_ir::ids::ValueId;
use crate::interpreter::data_structures::value::Value;

/// Execution context for a single tile block.
pub struct ExecutionContext<'a> {
    /// Immutable reference to the IR arena containing all IR structures.
    pub arena: &'a IrArena,

    /// SSA value storage: ValueId -> Runtime Value.
    values: HashMap<u32, Value>,

    /// Current tile block coordinates (x, y, z).
    pub tile_block_id: (u32, u32, u32),

    /// Total grid dimensions (x, y, z).
    pub grid_size: (u32, u32, u32),

    /// Global memory buffers, keyed by global variable name.
    pub globals: &'a HashMap<String, Value>,
}

unsafe impl Sync for ExecutionContext<'_> {}
unsafe impl Send for ExecutionContext<'_> {}

impl<'a> ExecutionContext<'a> {
    pub fn new(
        arena: &'a IrArena,
        tile_block_id: (u32, u32, u32),
        grid_size: (u32, u32, u32),
        globals: &'a HashMap<String, Value>,
    ) -> Self {
        ExecutionContext {
            arena,
            values: HashMap::new(),
            tile_block_id,
            grid_size,
            globals,
        }
    }

    /// Set a value in the value map (write-once per SSA semantics).
    ///
    /// Panics if attempting to overwrite an existing SSA value, as this violates
    /// SSA semantics.
    pub fn set_value(&mut self, id: ValueId, value: Value) {
        if self.values.contains_key(&id.0) {
            panic!("Attempting to overwrite SSA value {}", id.0);
        }
        self.values.insert(id.0, value);
    }

    /// Get a value, panicking if not found.
    pub fn get_value(&self, id: ValueId) -> &Value {
        self.values
            .get(&id.0)
            .unwrap_or_else(|| panic!("SSA value {} not found", id.0))
    }

    /// Clear all values, resetting the context for reuse.
    ///
    /// Used when the same context needs to be reused for multiple executions.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Update the tile block ID for a new execution.
    pub fn set_tile_block_id(&mut self, id: (u32, u32, u32)) {
        self.tile_block_id = id;
    }
}

/// Top-level interpreter that manages kernel execution across all tile blocks.
pub struct Interpreter {
    /// The IR arena containing all parsed IR structures.
    arena: IrArena,

    /// Grid size
    grid_size: (u32, u32, u32),

    /// Global memory buffers, keyed by global variable name.
    globals: HashMap<String, Value>,
}

impl Interpreter {
    /// Create a new interpreter with the given IR arena.
    pub fn new(arena: IrArena, grid_size: (u32, u32, u32)) -> Self {
        Interpreter {
            arena,
            grid_size,
            globals: HashMap::new(),
        }
    }

    /// Set a global variable's value.
    pub fn set_global(&mut self, name: impl Into<String>, value: Value) {
        self.globals.insert(name.into(), value);
    }

    /// Get a global variable's value.
    pub fn get_global(&self, name: &str) -> &Value {
        self.globals
            .get(name)
            .unwrap_or_else(|| panic!("Global variable {} not found", name))
    }

    /// Create an execution context for a specific tile block.
    pub fn create_context(&self, tile_block_id: (u32, u32, u32)) -> ExecutionContext<'_> {
        ExecutionContext::new(&self.arena, tile_block_id, self.grid_size, &self.globals)
    }
}
