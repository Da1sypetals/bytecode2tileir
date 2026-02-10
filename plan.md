# Implementation Plan: Interpreter Load/Store Operations with Comprehensive Testing

## Context

The bytecode2tileir project has an interpreter module with data structures (Tile, TensorView, PartitionView) already defined, but the actual execution logic for load/store operations is not yet implemented. The user requires:

1. **Full implementation** of load_view_tko and store_view_tko operations
2. **Rigorous testing** including very large scale tests
3. **Complete API coverage** - all types (Bool, I8, I16, I32, I64, F16, F32, F64, Ptr) must be tested
4. **Nightly f16 usage** - use native f16 type, not the `half` crate (already enabled via `#![feature(f16)]`)

The load/store operations must implement the Tile IR semantics as specified in the documentation, including:
- TensorView: structured pointer with shape/strides for accessing memory
- PartitionView: tiled subdivision of TensorView with masking support
- Memory ordering semantics (weak, relaxed, acquire, release)
- Memory scope (tl_blk, device, sys)

## Critical Files

### Implementation Files
- `src/interpreter/data_structures/tensor_view.rs` - Add methods to TensorView and PartitionView
- `src/interpreter/data_structures/tile.rs` - Existing Tile implementation (read-only reference)
- `src/interpreter/data_structures/elem_type.rs` - Existing type system (read-only reference)
- `src/interpreter/data_structures/interpreter.rs` - Add execution engine
- `src/interpreter/data_structures/value.rs` - Existing Value enum (read-only reference)

### Test Files (to be created)
- `src/interpreter/tests/test_load_store.rs` - Comprehensive load/store tests
- `src/interpreter/tests/test_tensor_view.rs` - TensorView API tests
- `src/interpreter/tests/test_partition_view.rs` - PartitionView API tests
- `src/interpreter/tests/mod.rs` - Test module declaration

## Implementation Steps

### Phase 1: Implement TensorView API Methods

Add public methods to `TensorView` in `tensor_view.rs`:

```rust
impl TensorView {
    pub fn new(base_ptr: *mut u8, elem_type: ElemType, shape: Vec<i64>, strides: Vec<i64>) -> Self
    pub fn base_ptr(&self) -> *mut u8
    pub fn elem_type(&self) -> ElemType
    pub fn shape(&self) -> &[i64]
    pub fn strides(&self) -> &[i64]
    pub fn rank(&self) -> usize
}
```

**Note**: TensorView does NOT directly support load/store of scalars. Loading from a TensorView is done via load_view_tko which always loads the entire view as a Tile (even if the view is rank-0, resulting in a scalar tile).

### Phase 2: Implement PartitionView API Methods

Add public methods to `PartitionView` in `tensor_view.rs`:

```rust
impl PartitionView {
    pub fn new(
        tensor_view: TensorView,
        tile_shape: Vec<i32>,
        dim_map: Vec<i32>,
        masked: bool,
        padding_value: Option<Scalar>,
    ) -> Self

    pub fn tensor_view(&self) -> &TensorView
    pub fn tile_shape(&self) -> &[i32]
    pub fn dim_map(&self) -> &[i32]
    pub fn is_masked(&self) -> bool
    pub fn padding_value(&self) -> Option<Scalar>

    // Calculate index space shape
    pub fn index_space_shape(&self) -> Vec<i64>

    // Load tile at grid indices
    pub fn load_tile(&self, grid_indices: &[i64]) -> Tile

    // Store tile at grid indices
    pub fn store_tile(&self, grid_indices: &[i64], tile: &Tile)
}
```

**Load algorithm**:

According to the Tile IR specification (types.md line 94-96), for a partition view with:
- Tensor view: shape `[S_0, ..., S_n]`, strides `[st_0, ..., st_n]`
- Tile size: `[T_0, ..., T_n]`
- Grid index: `[I_0, ..., I_n]`

The location of element `[i_0, ..., i_n]` within the loaded tile is:
```
byte_offset = sum(m=0 to n) { (I_m * T_m + i_m) * st_m } * elem_size_bytes
address = base_ptr + byte_offset
```

Where:
- `I_m` is the grid position (input to load_tile)
- `T_m` is the tile dimension size
- `i_m` is the position within the tile (iterating 0 to T_m-1)
- `st_m` is the stride in elements (from tensor view)
- `elem_size_bytes` is the byte size of the element type

Steps:
1. Create output tile with shape = tile_shape
2. For each position `[i_0, ..., i_n]` in the tile:
   - Compute element position in view: `view_pos[m] = I_m * T_m + i_m`
   - Check if masked and view_pos is out of bounds (view_pos[m] >= S_m)
   - If out of bounds and masked: use padding_value
   - Otherwise: compute byte_offset = sum(view_pos[m] * st_m) * elem_size, load from base_ptr + byte_offset
3. Return constructed tile

**Store algorithm**:
1. Verify tile shape matches tile_shape
2. For each position `[i_0, ..., i_n]` in the tile:
   - Compute element position in view: `view_pos[m] = I_m * T_m + i_m`
   - Check if masked and view_pos is out of bounds
   - If out of bounds and masked: skip this element
   - Otherwise: compute byte_offset, store to base_ptr + byte_offset

### Phase 3: Test TensorView API

Create `src/interpreter/tests/test_tensor_view.rs`:

**Test categories**:
1. Constructor and getters for all element types (Bool, I8, I16, I32, I64, F16, F32, F64, Ptr)
2. Multi-dimensional tensors with arbitrary strides (1D, 2D, 3D, 4D)
3. Various stride patterns:
   - Contiguous (strides=[N*M*..., M, 1])
   - Transposed (strides=[1, M, M*N, ...])
   - Arbitrary strides (prime numbers, mixed orderings)
4. Edge cases: scalar views (rank-0), single element
5. Verify memory layout correctness by checking actual memory addresses

**Scale**: Test with dimensions up to 1024x1024 for 2D, verify memory layout correctness.

### Phase 4: Test PartitionView API

Create `src/interpreter/tests/test_partition_view.rs`:

**Test categories**:
1. Constructor and getters for all element types
2. Index space calculation:
   - Even division (shape=1024, tile=32 → 32 tiles)
   - Uneven division (shape=1000, tile=32 → 32 tiles, last partial)
3. Load/store tiles at various grid positions
4. **Dimension mapping (dim_map) - comprehensive permutation testing**:
   - dim_map is a permutation of (0..n-1) mapping tile dimensions to view dimensions
   - For 2D: test ALL permutations: [0,1], [1,0]
   - For 3D: test ALL 6 permutations: [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]
   - For 4D: test representative permutations (all 24 would be excessive, test ~8 diverse ones)
   - For each permutation, verify that loads/stores access correct memory locations
   - Test with different tensor shapes and tile sizes for each permutation
5. Masked vs unmasked partitions:
   - Unmasked with partial tiles (undefined behavior, document)
   - Masked with partial tiles (out-of-bounds handled)
6. All element types with f16 explicitly tested same as other types

**Scale**: Large partition views with 10000x10000 tensors divided into 128x128 tiles.

### Phase 5: Comprehensive Load/Store Operation Tests

Create `src/interpreter/tests/test_load_store.rs`:

**Test categories**:
1. **Basic functionality** - load then store, verify round-trip for all types
2. **Type coverage** - dedicated test for each element type:
   - Bool (true/false patterns)
   - I8, I16, I32, I64 (negative/positive/zero/max/min values)
   - F16, F32, F64 (special values: 0.0, -0.0, inf, -inf, NaN, subnormals)
   - Ptr (null and various addresses)
3. **Memory patterns**:
   - Sequential access
   - Strided access
   - Random access
   - Diagonal patterns
4. **Tile sizes**: Powers of 2 from 1x1 to 1024x1024
5. **Memory ordering semantics** (weak, relaxed, acquire, release):
   - In serial interpreter, these are mostly no-ops but should be accepted
6. **Memory scope** (tl_blk, device, sys):
   - In serial interpreter, these are mostly no-ops but should be accepted
7. **Token handling**:
   - Load/store with no token (fresh token)
   - Load/store with input token (token threading)
8. **Large scale tests**:
   - 10000x10000 f32 matrix, load/store entire grid
   - 100000 element 1D vectors for all types
   - 3D/4D tensors with complex strides
   - Memory stress test: allocate 100MB+, perform 10000+ load/store ops

Tests may run for extended periods - this is expected as long as there are no infinite loops.

### Phase 6: Interpreter Integration (Minimal)

The Interpreter represents the register file which maps SSA value IDs to immutable values (tiles are immutable per semantics.md line 102). The register file is write-once per SSA semantics.

Extend `src/interpreter/data_structures/interpreter.rs`:

```rust
impl Interpreter {
    pub fn new(num_tile_blocks: (u32, u32, u32)) -> Self

    // Write-once register assignment (SSA semantics)
    pub fn set_value(&mut self, id: u64, value: Value)

    // Read from register file
    pub fn get_value(&self, id: u64) -> Option<&Value>

    // Grid-related queries (already implemented)
    pub fn grid(&self) -> Vec<(u32, u32, u32)>
}
```

**NO execute_* methods** - operations are performed by calling methods directly on Value/Tile/TensorView/PartitionView types, and the interpreter just stores the immutable results in its register file.

**NO get_value_mut** - values are immutable once created and stored in the register file.

## Implementation Requirements

1. **Use nightly f16 directly** - Code for F16 should look identical to F32/F64, using native operators
2. **No simplified examples** - All tests must use real, complete scenarios
3. **Exhaustive type coverage** - Every test that applies to multiple types MUST test all types
4. **Memory safety** - Use unsafe blocks for raw pointer access, document safety invariants
5. **Error handling** - Panic on violations (bounds, type mismatches) as per CLAUDE.md (no try-except)
6. **Documentation** - Add doc comments explaining semantics from Tile IR spec

## Testing Strategy

### Test Organization
```
src/interpreter/tests/
├── mod.rs (declare test modules)
├── test_tensor_view.rs (1000+ lines)
├── test_partition_view.rs (1000+ lines)
└── test_load_store.rs (2000+ lines)
```

### Test Execution
- Run with: `cargo test --lib interpreter::tests`
- Large scale tests should have descriptive output showing progress
- All tests must pass without simplification or omission

## Verification Plan

After implementation:
1. Run `cargo test` - all tests pass
2. Run `cargo test -- --nocapture` - verify large scale test output
3. Manually verify f16 code matches f32/f64 patterns
4. Check that no test is simplified or type coverage missing
5. Review unsafe code for correctness and documentation

## Notes

- The bytecode parsing already exists (ops_mem.rs), we only implement execution
- Tests allocate real memory using `Vec<u8>` as backing storage
- F16 should be dealt with in the same way as f32 or f64. It is NOT FROM THE `half` CRATE. I have ALREADY used nightly Rust.
- Memory ordering/scope will be properly implemented when adding concurrency support later
- You are NOT allowed to make any simplification, placeholders, half-baked results, etc. to fool me.
- You are not required to conform to Rust's safety considerations. Program like C.

## Behavior notes

- Stop to ask for my comments on whether we should continue after you complete EVERY STAGE.