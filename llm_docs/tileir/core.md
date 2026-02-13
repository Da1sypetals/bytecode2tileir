
## 8.3. Core

### 8.3.1. cuda_tile.broadcast

*Broadcast tile to new shape*

```
cuda_tile.broadcast %source
```

#### Parameters

- **source** (tile) - The tile to broadcast.

#### Results

- **result** (tile) - The broadcasted tile.

#### Description

The `broadcast` operation expands each unary (`1`) dimension in the input tile by duplicating the data along that dimension.

Expansion happens only for dimensions of size one that are stretched or "copied" to match the size of the dimension implied by the result type of the operation. The operation does not change the rank of the source tile. Any change to the rank of the source tile must be made using reshape-like operations before broadcasting.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same element type (tile).
- `source` and `result` must have the same rank.

### 8.3.2. cuda_tile.cat

*Concatenate tiles along specified dimension*

```
cuda_tile.cat %lhs %rhs %dim
```

#### Parameters

- **lhs** (tile) - The left hand side operand.
- **rhs** (tile) - The right hand side operand.
- **dim** (i64) - The dimension along which to concatenate.

#### Results

- **result** (tile) - The concatenated result tile.

#### Description

The `cat` operation concatenates the two input tiles. The input tiles must have the same shape in all but the concatenating dimension. Concatenation happens along the dimension specified by the the attribute `dim` the resulting dimension is the sum of the the two input tiles concatenating dimension.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same rank.
- `lhs`, `rhs` and `result` must have the same element type (tile).

#### Examples

```mlir
// A valid invocation of cat.
%0 = cat %arg0, %arg1 dim = 1
  : tile<2x4xf32>, tile<2x4xf32> -> tile<2x8xf32>

// >>> %arg0 = tile([[ A, B, C ],
//                   [ D, E, F ]])
// >>> %arg1 = tile([[ 1, 2, 3 ],
//                   [ 4, 5, 6 ]])
// >>> %0 = tile([[ A, B, C, 1, 2, 3 ],
//                [ D, E, F, 4, 5, 6 ]])

// A valid invocation of cat.
%1 = cat %arg0, %arg1 dim = 0
  : tile<2x4xf32>, tile<2x4xf32> -> tile<4x4xf32>

// >>> %arg0 = tile([[ A, B, C ],
//                   [ D, E, F ]])
//
// >>> %arg1 = tile([[ 1, 2, 3 ],
//                   [ 4, 5, 6 ]])
//
// >>> %1 = tile([[ A, B, C ],
//                [ D, E, F ],
//                [ 1, 2, 3 ],
//                [ 4, 5, 6 ]])
```

See cuda_tile.cat_0 for the full example listing.

### 8.3.3. cuda_tile.constant

*Create constant tile*

```
cuda_tile.constant %value
```

#### Parameters

- **value** (DenseConstant) - The constant value to create.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The constant tile.

#### Description

The `constant` operation creates a tile initialized by `$value`.

There are two main forms of using the operation:

- One where the value is a single constant specified by `dense<c>` and the tile is filled with identical values for all elements.
- One where the value is a list of constants specified by `dense<[c0, c1, c2, ...]>` and the constant value's shape must match the tile's shape.

The annotated type of the tile constrains its rank, shape, and element type.

#### Constraints

- The operation has no operands and may be constant folded.
- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `value` and `result` must have the same shape and element type (DenseConstant).
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
%c0 = constant dense<0> : tile<i32>
%c1 = constant dense<1> : tile<i64>
%c2 = constant dense<[0, 1, 2, 3]> : tile<4xi32>
%c3 = constant dense<0.0> : tile<2x4xf32>
%c4 = constant dense<[0.0, 1.0, 2.0, 3.0]> : tile<4xf64>
```

See cuda_tile.constant_0 for the full example listing.

### 8.3.4. cuda_tile.entry

*Define a tile kernel*

```
cuda_tile.entry %sym_name %function_type %arg_attrs %res_attrs %optimization_hints
```

#### Parameters

- **sym_name** (Symbol) - The name of the function.
- **function_type** (Type) - The type of the function.
- **arg_attrs** (Attributes) - The argument attributes of the function: none of these are supported by CUDA Tile IR at the moment.
- **res_attrs** (Attributes) - The result attributes of the function: none of these are supported by CUDA Tile IR at the moment.
- **optimization_hints** (OptimizationHints) - Compiler architecture-specific optimization hints

#### Results

No results.

#### Description

The `entry` operation defines a tile kernel; a kernel is a function that can serve as the program entry point. It has a unique name per-module. A kernel cannot return any value. It must be launched from the host side using `cuLaunchKernel` or similar CUDA runtime API functions.

Tile kernels require that the user specifies the 3-d grid dimensions at launch which defines the number of tile blocks (or kernel instances) that will execute the kernel in parallel.

For detailed semantics of tile kernels see Tile Kernel.

The `optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries.

The hints are specified for each architecture (e.g., `sm_100`, `sm_120`) and for each architecture the user can specify specific hints for each operation.

- `num_cta_in_cga` - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for cuda_tile.entry.
- `allow_tma` - suggest whether to use TMA for cuda_tile.load_view_tko and cuda_tile.store_view_tko.
- `latency` - latency hint for cuda_tile.load_view_tko and cuda_tile.store_view_tko.

For example they can be annotated as:

```mlir
optimization_hints=<{
  sm_100 = {num_cta_in_cga = 8},
  sm_120 = {num_cta_in_cga = 16}
}>
```

#### Constraints

- The operation must be a symbol in the global symbol table.
- The operation must implement callable target interface.
- The operation must implement function-like behavior interface.
- The region must not capture SSA values defined above the operation.
- The operation must provide custom parsing and printing methods.
- Each provided region must contain exactly one block.

### 8.3.5. cuda_tile.extract

*Extract a subtile from a tile*

```
cuda_tile.extract %source %indices
```

#### Parameters

- **source** (tile) - The source tile to extract from.
- **indices** (Variadic<tile<i32>>) - The indices of the slice to extract.

#### Results

- **result** (tile) - The extracted subtile.

#### Description

The `extract` operation extracts a subtile from the given source tile.

The shape of the result tile must divide the shape of the source tile evenly e.g., `tile<4xf32>` is a valid extraction from `tile<8xf32>`, but `tile<3xf32>` is not.

The `$indices` indicate the number of the slice to extract, but *importantly* not the offsets used to construct the subtile for extraction. The semantics of extract means that only full size slices can be extracted.

Slices of a source tile with the same shape are non-overlapping by definition for unique indices.

> **Warning**
> If the `indices` specify a non-existent (i.e., out-of-bounds) slice, the behavior of the operation is undefined.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same rank.

#### Examples

```mlir
// Extract a subtile from %t at dim_0 = [4;8) and dim_1 = [4;6).
%c1 = constant dense<1> : tile<i32>
%c2 = constant dense<2> : tile<i32>
%t = constant dense<0.0> : tile<32x8xf32>
// Valid indices are: [ {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3} ]
%0 = extract %t[%c1, %c2]
    : tile<32x8xf32> -> tile<4x2xf32>
```

See cuda_tile.extract_0 for the full example listing.

### 8.3.6. cuda_tile.get_global

*Get a pointer to a global variable*

```
cuda_tile.get_global %name
```

#### Parameters

- **name** (Symbol) - The name of the global variable.

#### Results

- **result** (tile<T*>) - The pointer to the global variable.

#### Description

The `get_global` operation returns a pointer to the specified `global` variable. A global variable is a form of static global memory allocation that can be declared using the cuda_tile.global operation.

The element type of the returned pointer will be of the same type as the element type of the declared global variable.

For detailed semantics of global variables see Global Variable.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.

#### Examples

```mlir
global @val dense<[0.1, 0.2, 0.3, 0.4]> : tile<4xf32>

entry @example() {
  %ptr = get_global @val : tile<ptr<f32>>
  return
}
```

See cuda_tile.get_global_0 for the full example listing.

### 8.3.7. cuda_tile.get_num_tile_blocks

*Get total number of tile blocks*

```
cuda_tile.get_num_tile_blocks
```

#### Parameters

No parameters.

#### Results

- **gridSize_x** (tile<i32>) - The number of tile blocks in dimension `x`.
- **gridSize_y** (tile<i32>) - The number of tile blocks in dimension `y`.
- **gridSize_z** (tile<i32>) - The number of tile blocks in dimension `z`.

#### Description

The `get_num_tile_blocks` operation queries the total number of tile blocks in the form of a 3-tuple specifying the extent of each grid dimension.

A tile `id` is a coordinate in 3-space and therefore the must also be a 3-tuple containing the extent of each dimension: `x`, `y` and `z`.

When launching 1- or 2-dimensional grids, the unspecified dimensions will have a cardinality of 1.

For example if the grid used to launch the kernel is `(1024, 1024)` then the result of this operation will be `(1024, 1024, 1)`.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
entry @example() {
  %x, %y, %z = get_num_tile_blocks : tile<3xi32>
  print "x: %, y: %, z: %\n", %x, %y, %z : tile<i32>, tile<i32>, tile<i32>
}
```

See cuda_tile.get_num_tile_blocks_0 for the full example listing.

### 8.3.8. cuda_tile.get_tile_block_id

*Get the currently executing tile block coordinates*

```
cuda_tile.get_tile_block_id
```

#### Parameters

No parameters.

#### Results

- **blockId_x** (tile<i32>) - The tile block ID for dimension `x`.
- **blockId_y** (tile<i32>) - The tile block ID for dimension `y`.
- **blockId_z** (tile<i32>) - The tile block ID for dimension `z`.

#### Description

`get_tile_block_id` returns a 3-d tile block coordinates (or ID) of the currently executing tile block.

A tile ID has three dimensions: `x`, `y`, and `z`. This operation returns all three of them simultaneously. The value of each dimension returned by this operation is between `0` (including) and the value returned by `get_num_tile_blocks` for the respective axis (excluding), represented by the inclusive interval `[0, get_num_tile_blocks(dim) - 1]`. Grid dimensions unspecified at kernel launch (i.e., a 1-d or 2-d grid) will always be `0` for all tile blocks.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- The operation's result type may be inferred from its operands and attributes。

### 8.3.9. cuda_tile.global

*Allocate static global memory*

```
cuda_tile.global %sym_name %value %alignment
```

#### Parameters

- **sym_name** (Symbol) - The name of the global variable.
- **value** (DenseConstant) - The value to initialize the allocation with.
- **alignment** (i64) - The alignment of the buffer.

#### Results

No results.

#### Description

The `global` operation statically allocates a mutable 1-dimensional location in global memory and initializes it using `value`. The initialization of the allocation is performed at CUDA module load time. The lifetime of the allocation is the same as the lifetime of the module.

The allocation may be read or written to by first using cuda_tile.get_global to obtain a pointer to the the memory and then read using cuda_tile.load_ptr_tko or written to using cuda_tile.store_ptr_tko.

The initial values are stored in memory in linear order, so the pointer returned by cuda_tile.get_global points to the first element, and offsetting the pointer by x would allow to load element at position x.

`global` operations must be directly nested within the **Tile IR** module. They cannot be defined inside functions. As globals are defined at the module scope their names are globally unique symbols and must not collide with any other symbol in the module.

For more detailed semantics of global variables see Global Variable.

#### Constraints

- The operation must be a symbol in the global symbol table.

#### Examples

```mlir
global @val alignment = 128 dense<[0.1, 0.2, 0.3, 0.4]> : tile<4xf32>
entry @example() {}
```

### 8.3.10. cuda_tile.iota

*Generate a 1-d tile range from 0 to n-1*

```
cuda_tile.iota
```

#### Parameters

No parameters.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The result of the iota operation.

#### Description

The `iota` operation generates a 1-d tile with a sequence of integer values. The starting value is `0` and the stride is `1`. If the shape of the result tile is `(n)`, then the generated values are `[0, n - 1]`.

> **Note**
> The number of elements in the result tile must not exceed the maximum value that the element type can express.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects。

### 8.3.11. cuda_tile.module

*Top-level module containing a series of defined items.*

```
cuda_tile.module %sym_name
```

#### Parameters

- **sym_name** (Symbol) - The name of the module.

#### Results

No results.

#### Description

A `module` operation represents a single compilation unit and contains zero or more items (global variables, functions, or kernels).

For detailed description of the semantics of modules, and the full definition of each item type see Modules.

The `module` operation is the top-level operation in a **Tile IR** module and must contain only **Tile IR** operations and no other dialects.

#### Constraints

- The region must not capture SSA values defined above the operation.
- The operation must provide custom parsing and printing methods.
- All regions must have zero arguments.
- Each provided region must contain exactly one block.
- The operation must define a symbol scope.
- The region must not require explicit terminator operations.
- The operation must specify whether regions are SSACFG or Graph kind.
- The operation must contain only dataflow graph regions.

### 8.3.12. cuda_tile.offset

*Offsets a tile of pointers*

```
cuda_tile.offset %ptr %offset
```

#### Parameters

- **ptr** (ptr) - The base pointer tile to advance.
- **offset** (tile<i1 | i8 | i16 | i32 | i64>) - The offset tile to add to the pointer.

#### Results

- **result** (ptr) - The resulting pointer tile after advancement.

#### Description

`offset` advances a tile of pointers. It takes `ptr` as base and `offset` as increment, and performs element-wise addition of `ptr` by `offset`:

```mlir
result[i,j] = ptr[i,j] + offset[i,j] * bitwidth
```

`ptr` is interpreted as an unsigned integer. `offset` is interpreted as a signed integer. `bitwidth` is the storage bitwidth of the pointee type. The multiplication must not overflow (wrap-around) in a signed sense. The addition must not overflow (wrap-around) in an unsigned sense. In case of an overflow, the result is undefined.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- The operation must apply element-wise to its operands.
- `ptr`, `offset` and `result` must have the same shape.
- `result` and `ptr` must have the same shape and element type (ptr).
- The operation's result type may be inferred from its operands and attributes.

### 8.3.13. cuda_tile.permute

*Permute tile dimensions*

```
cuda_tile.permute %source %permutation
```

#### Parameters

- **source** (tile) - The input tile.
- **permutation** (Array<i32>) - The permutation of the dimensions.

#### Results

- **result** (tile) - The permuted tile.

#### Description

Permute the dimensions of the input tile `source` according to the `permutation` array. The `permutation` array is a list of integers that specify the new order of the dimensions.

For example, if the input tile has shape `[2, 4, 8]`, and the permutation is `[2, 0, 1]`, the output tile will have shape `[8, 2, 4]`.

This operation logically is a change in the indexing of the tile.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same element type (tile).
- `source` and `result` must have the same rank.

#### Examples

```mlir
%arg0 = constant dense<0.0> : tile<2x4x8xf16>
%0 = permute %arg0 [2, 0, 1] : tile<2x4x8xf16> -> tile<8x2x4xf16>
```

See cuda_tile.permute_0 for the full example listing.

### 8.3.14. cuda_tile.reduce

*Variadic tile reduction across dimensions*

```
cuda_tile.reduce %operands %dim %identities
```

#### Parameters

- **operands** (Variadic<tile>) - The tiles to reduce.
- **dim** (i32) - The index dimension that needs to be reduced.
- **identities** (Array) - The reduction identities for each operand.

#### Results

- **results** (Variadic<tile>) - The reduced tiles.

#### Description

Applies a reduction function `body` to `operands` and `identities` along dimensions `dimensions` and produces new `results` tile values. The order of reduction is implementation-defined but the result is deterministic.

Argument explained:

- `operands` are the tiles to reduce.
- `identities` are the reduction identities for each operand. Identity at position i binds with the operand at the same position. Identities are properties of the reduction function in the `body`. For example, the identity of a min reduction is +inf, while the identity of a sum is 0.
- `dim` is the index of the dimension to be reduced.
- `body` is a region carrying the reduction(s) semantics. Each operation within the region must be a cuda_tile operation with 0-rank cuda_tile tile types. Region arguments are bound to operands in the following way: [operand_0_current_iter, operand_0_prev_iter, operand_1_current_iter, operand_1_prev_iter…]. operand_i_current_iter is the current element to reduce from operand at index i. operand_i_prev_iter is the accumulator that might be an element of the same operand at index i, the result of the previous reduction step or the identity value associated with `operand_i_current_iter`.

#### Constraints

- The operation must provide custom parsing and printing methods.
- The operation only has an effect if and only if it the region's operation have an effect.
- All operands must have identical shapes.
- Each provided region must contain exactly one block.

### 8.3.15. cuda_tile.reshape

*Reshape tile dimensions*

```
cuda_tile.reshape %source
```

#### Parameters

- **source** (tile) - The source tile to reshape.

#### Results

- **result** (tile) - The reshaped tile.

#### Description

The `reshape` operation changes the shape of the `source` operand. `reshape` is only a change in the indexing of the tile. The number of elements and element type must remain unchanged.

0-d tiles (i.e., scalars) contain precisely one element and thus are the one exception where a 0-d tile can be reshaped to shape where the `size(shape) == 1`.

Conceptually reshaping a tile is equivalent to first creating a 1-d tile from the data of the source assuming a row-major layout and then converting the 1-d tile into the new shape in a row-major layout.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same element type (tile).

#### Examples

```mlir
%cst = constant dense<0> : tile<i8>
%0 = reshape %cst
    : tile<i8> -> tile<1x1x1xi8>

%t = constant dense<0.0> : tile<8x2xf32>
%1 = reshape %t
    : tile<8x2xf32> -> tile<2x2x4x1xf32>
```

See cuda_tile.reshape_0 for the full example listing.

```mlir
  %cst = constant dense<[[0, 1, 2, 3], [4, 5, 6, 7]]>
      : tile<2x4xi32>
  %r0 = reshape %cst
: tile<2x4xi32> -> tile<2x2x2xi32>

// Step 1: Turn source into 1D tile. Use row-major by convention.
// %tmp: [0, 1, 2, 3, 4, 5, 6, 7]
%tmp = reshape %cst
    : tile<2x4xi32> -> tile<8xi32>

// Step 2: Turn 1D tile into result tile. Use row-major by convention.
// %r: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
%r1 =  reshape %tmp
        : tile<8xi32> -> tile<2x2x2xi32>
```

See cuda_tile.reshape_1 for the full example listing.

### 8.3.16. cuda_tile.scan

*A parallel prefix sum operation*

```
cuda_tile.scan %operands %dim %reverse %identities
```

#### Parameters

- **operands** (Variadic<tile>) - The a set of tiles to scan.
- **dim** (i32) - The index of the dimension along which to scan.
- **reverse** (bool) - Whether to scan in reverse order.
- **identities** (Array) - The identities of the scan operation.

#### Results

- **results** (Variadic<tile>) - The resulting tiles from the scan operation.

#### Description

Applies a scan function `body` to `operands` and `identities` along dimension `dim` and produces new `results` tile values. The scan operation maintains a carry value that is updated as it processes elements along the specified dimension. For each element, the scan function combines the current element with the carry value to produce both a result and an updated carry. The order of scan is implementation-defined but the result is deterministic.

`identities` are the scan identities for each operand. Identity at position i binds with the operand at the same position. Identities are properties of the scan function in the `body`. For example, the identity of a min scan is +inf, while the identity of a sum is 0.

`body` is a region carrying the scan semantics. Each operation within the region must be a cuda_tile operation with 0-rank cuda_tile tile types. Region arguments are bound to operands in the following way: `[operand_0_current_iter, operand_0_prev_iter, operand_1_current_iter, operand_1_prev_iter...]`. `operand_i_current_iter` is the current element to scan from operand at index `i`. `operand_i_prev_iter` is the accumulator that might be an element of the same operand at index `i`, the result of the previous scan step or the identity value associated with `operand_i_current_iter`.

> **Warning**
> The current implementation only supports single tile input.

#### Constraints

- The operation must provide custom parsing and printing methods.

#### Examples

```mlir
%input = constant dense<0.0> : tile<8x16xf32>
%result = scan %input dim=1 reverse=false identities=[1.0 : f32] : tile<8x16xf32> -> tile<8x16xf32>
(%acc: tile<f32>, %elem: tile<f32>) {
  %prod = mulf %acc, %elem rounding<nearest_even>: tile<f32>
  yield %prod : tile<f32>
}
```

See cuda_tile.scan_0 for the full example listing.

### 8.3.17. cuda_tile.select

*Select values based on condition*

```
cuda_tile.select %cond %val_if_true %val_if_false
```

#### Parameters

- **cond** (tile<i1>) - The condition tile.
- **val_if_true** (tile) - The value if true tile.
- **val_if_false** (tile) - The value if false tile.

#### Results

- **result** (tile) - The tile of selected values.

#### Description

The `select` op chooses values based on the binary conditions supplied as the `cond` operand. The `val_if_true` operand contains the value(s) to use if the condition is 1. The `val_if_false` operand contains the value(s) to use if the condition is 0. The choice is made element-wise according to the values in the condition tile.

All tiles must have the same shape. The tiles `val_if_true`, `val_if_false`, and the result must have the same element type. The `cond` tile must be a tile of `i1` values.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `val_if_true`, `val_if_false` and `result` must have the same shape and element type (tile).
- The operation's result type may be inferred from its operands and attributes.
