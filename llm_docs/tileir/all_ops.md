# 8. Operations

This section describes a complete and categorized list of all **Tile IR** instructions names, signatures, and semantics.

## 8.1. Meta Types

Operations have arguments which are **Tile IR** values with **Tile IR** types but many operations have immediate or static arguments which correspond to attributes in the MLIR dialect. These **meta types** are not representable in the **Tile IR** type system but are used to construct **Tile IR** programs and only present at compile time. Operations in the specification are described abstractly in both the **Tile IR** IR and bytecode independent of the MLIR or bytecode encoding. For each of these types we provide a definition of them below and link to them from each operation.

> **Note**
> The convention is that the meta types are capitalized and **Tile IR** types are snake cased.

The convention is that the meta types are capitalized and the native **Tile IR** types are camel cased are snake cased.

### 8.1.1. Symbol

`Symbol` a symbol in the program, begins with `@` and uniquely identifies a symbol in the program.

### 8.1.2. Flag

`Flag` a boolean value that can be used to control the behavior of an operation.

### 8.1.3. Token

Token represents a memory ordering token that can be used to control the ordering of memory operations.

### 8.1.4. Variadic

`Variadic` represents an argument which can accept a statically sized, but variable, number of arguments.

### 8.1.5. Any

`Any` represents a value of any valid **Tile IR** type.

### 8.1.6. Name

`Name` represents a name in the program, begins with `#` and uniquely identifies a name in the program.

### 8.1.7. Type

`Type` represents a **Tile IR** type and are attached as attributes to operations which define IR items.

### 8.1.8. Array

`Array` represents a statically sized array of values that can be passed to attributes.

### 8.1.9. String

`String` represents a string value that can be passed to attributes.

### 8.1.10. bool

`bool` represents a boolean value that can be passed to attributes.

### 8.1.11. DenseConstant

`DenseConstant` represents a dense constant value that can be passed to attributes.

### 8.1.12. view_type

`view_type` represents a type which implements the view interface, currently this is only implemented by *partition_view* but will have new implementers in future releases.

## 8.2. Operation Design Considerations

The design of **Tile IR** has a set of design considerations that apply to all operations in the dialect this section introduces some of the common design considerations that apply to all operations, or to classes of operations generically.

### 8.2.1. Explicit Broadcast

There are no implicit broadcast performed by operations in the **Tile IR** dialect all operations that require operands of the same shape must be explicitly broadcasted. For example to use the cuda_tile.offset operation to add an offset tile to a pointer, the pointer and offset must be reshaped or broadcasted to have the same shape using the cuda_tile.reshape or cuda_tile.broadcast operations.

### 8.2.2. Distinct Floating-Point and Integer Operations

Numeric operations are split across integer and floating-point types due to differences in flags such as rounding modes, `NaN` handling, and fast math.

For example, the cuda_tile.addf operation supports a rounding attribute, but the addi operation does not.

### 8.2.3. Explicit Overflow Annotations

Some operations such as cuda_tile.addi support an explicit overflow annotation that expresses the expected overflow behavior of the operation.

These attributes serve as assumptions that an implementation may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

We recommend that generators of **Tile IR** programs utilize these annotations to help the implementation reason about the overflow behavior of the operation, enabling extra optimization opportunities.

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

## 8.4. Conversions

There are no implicit type conversions in **Tile IR** thus we expose a set of explicit conversion operations for interconverting between types which have compatible representations or rules for conversion.

cuda_tile.bitcast preserves the contents of the input but allows for changing of element types, cuda_tile.exti and cuda_tile.trunci change the width of integer tiles, cuda_tile.ftoi and cuda_tile.itof convert floating-point tiles to integer tiles and vice versa, and cuda_tile.ftof converts between different floating-point types.

For more details on conversions and their rules see the individual operation's documentation.

### 8.4.1. cuda_tile.bitcast

*Bitcast a tile from one element type to another*

```
cuda_tile.bitcast %source
```

#### Parameters

- **source** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The source tile to cast.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The casted tile.

#### Description

The `bitcast` operation casts the input tile from one element type to another without modifying the underlying bits.

Only non-pointer types of the same bit width are allowed (e.g., `i32` to `f32`). Pointer types must use cuda_tile.ptr_to_int or cuda_tile.int_to_ptr instead.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.

### 8.4.2. cuda_tile.exti

*Extend the width of an integer tile*

```
cuda_tile.exti %from %signedness
```

#### Parameters

- **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to extend.
- **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned`

#### Results

- **to** (tile<i1 | i8 | i16 | i32 | i64>) - The extended tile.

#### Description

The `exti` operation converts a tile of integers of a given width to a strictly larger width. Zero-extension is used for `unsigned` integers and sign-extension is used for `signed` integers.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.

### 8.4.3. cuda_tile.ftof

*Convert between floating-point types*

```
cuda_tile.ftof %from %rounding_mode
```

#### Parameters

- **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile.
- **rounding_mode** (RoundingMode) - The rounding mode for the operation.

#### Results

- **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The result floating-point tile.

#### Description

The `ftof` operation converts a tile of a given floating-point element type into one of a different floating-point element type (for example, from `f32` to `f64`).

The source type and the result type must be different.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.

### 8.4.4. cuda_tile.ftoi

*Convert a tile from floating-point values to integer values*

```
cuda_tile.ftoi %from %signedness %rounding_mode
```

#### Parameters

- **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile.
- **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned`
- **rounding_mode** (RoundingMode) - The rounding mode for the operation.

#### Results

- **to** (tile<i1 | i8 | i16 | i32 | i64>) - The result integer tile.

#### Description

The `ftoi` operation converts a floating-point tile into an integer tile.

In contrast to a `bitcast` which is bits preserving, this preserves the numerical value of the tile, rounded towards zero to the nearest integer of the provided type.

> **Warning**
> If the input floating-point value, after being rounded, is outside the (signed or unsigned) range of the target integer type, the closest representable value is used instead. `NaN` values are converted to 0. Input `Inf` values are undefined behavior.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.

### 8.4.5. cuda_tile.itof

*Convert a tile from integer values to floating-point values*

```
cuda_tile.itof %from %signedness %rounding_mode
```

#### Parameters

- **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile.
- **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned`
- **rounding_mode** (RoundingMode) - The rounding mode for the operation.

#### Results

- **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The result floating-point tile.

#### Description

The `itof` operation converts an integer tile into a floating-point tile.

In contrast to a `bitcast` which is bits preserving, this preserves the numerical value of the tile, rounded to the nearest floating-point number of the provided type.

> **Warning**
> If the input integer value, after being rounded, is outside the range of the target floating-point type, it is converted to `Inf` for types that support that value, and `NaN` otherwise.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.

### 8.4.6. cuda_tile.int_to_ptr

*Convert a tile of integers to a tile of pointers*

```
cuda_tile.int_to_ptr %source
```

#### Parameters

- **source** (tile<i64>) - The input tile of integers.

#### Results

- **result** (ptr) - The output tile of pointers.

#### Description

The `int_to_ptr` operation converts a tile of integers to a tile of pointers.

The inverse of this operation is cuda_tile.ptr_to_int.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.

### 8.4.7. cuda_tile.ptr_to_int

*Convert a tile of pointers to a tile of integers*

```
cuda_tile.ptr_to_int %source
```

#### Parameters

- **source** (ptr) - The input tile of pointers.

#### Results

- **result** (tile<i64>) - The output tile of integers.

#### Description

The `ptr_to_int` operation converts a tile of pointer-type elements to a tile of `i64` elements.

The inverse of this operation is cuda_tile.int_to_ptr.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.

### 8.4.8. cuda_tile.ptr_to_ptr

*Reinterpret a tile of one pointer type as another*

```
cuda_tile.ptr_to_ptr %source
```

#### Parameters

- **source** (ptr) - Tile with source pointer element type.

#### Results

- **result** (ptr) - Tile with target pointer element type.

#### Description

The `ptr_to_ptr` operation casts a tile of pointers from a pointer of one element type to another element. Casts between pointer and non-pointer types are disallowed.

In order to perform those conversions, use cuda_tile.ptr_to_int or cuda_tile.int_to_ptr. These operations are distinct to enable future compiler reasoning about pointer provenance.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.

### 8.4.9. cuda_tile.trunci

*Truncates the width of an integer tile*

```
cuda_tile.trunci %from %overflow
```

#### Parameters

- **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to truncate.
- **overflow** (OverflowBehavior) - The overflow behavior for the operation.

#### Results

- **to** (tile<i1 | i8 | i16 | i32 | i64>) - The truncated integer tile.

#### Description

The `trunci` operation converts a tile of integers of a given width to a strictly smaller width, by discarding the most significant bits. The operation works without a sign attribute since this is lossy regardless of whether an integer is treated as signed or unsigned.

For more information see the OverflowBehavior attribute.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.

## 8.5. Control Flow

**Tile IR** contains a standard set of control flow operations that enable conditionals, and loops.

The operations are designed in the style of the MLIR Control Flow dialect.

A notable difference is that we allow the nesting of control flow operations for example a cuda_tile.if may appear inside a cuda_tile.loop or cuda_tile.for.

The main control structures are:

- cuda_tile.if which implements conditional branching.
- cuda_tile.loop which implements a loop with arbitrary exit conditions.
- cuda_tile.for which implements a range-based loop with a fixed number of iterations.

These operations and their supporting operations are described in the following section.

### 8.5.1. cuda_tile.assert

*Terminate kernel execution with an error message if condition is false-y*

```
cuda_tile.assert %condition %message
```

#### Parameters

- **condition** (tile<i1>) - The condition tile to check.
- **message** (String) - The error message to display if assertion fails.

#### Results

No results.

#### Description

The `assert` operation takes as `condition` a tile of `i1` values. For each value that is `0`, it prints the given error message, along with the index of the value within the tile.

If at least one value is `0`, an error is signalled to the host side. The kernel, including the tile block that failed the assertion, may keep running.

Assertions are for debugging purposes. They can affect performance and it is therefore recommended to remove them in production code.

#### Constraints

No constraints.

#### Examples

```mlir
assert %arg0, "assertion failed" : tile<i1>
```

See cuda_tile.assert_0 for the full example listing.

### 8.5.2. cuda_tile.break

*Break from loop*

```
cuda_tile.break %operands
```

#### Parameters

- **operands** (Variadic<Any>) - The operands to yield to the parent loop upon termination.

#### Results

No results.

#### Description

The `break` operation is a terminator operation of a cuda_tile.loop.

It may yield any number of `$operands` to the parent loop upon termination. The number of values yielded and the execution semantics of how they are yielded are determined by the parent loop.

The `break` operation always returns control to the innermost enclosing loop operation, even when it is nested within other control constructs such as `if` or additional loops.

#### Constraints

- The operation must terminate its parent basic block.

#### Examples

```mlir
// Break from the body of a loop.
loop {
    break
}

// Break from an if nested within the loop.
loop  {
    %condition = constant dense<1> : tile<i1>
    if %condition  {
        break
    }
    // ...
}

%initValue0 = constant dense<0.0> : tile<f32>
// Break from an if nested within the loop, while yielding values.
%results = loop iter_values(%var0 = %initValue0): tile<f32> -> tile<f32> {
    %condition = constant dense<1> : tile<i1>
    if %condition  {
        // ...
        yield
    } else {
        // %if.loopValue0 = ...
        %loopValue0 = constant dense<1.0> : tile<f32>
        break %loopValue0 : tile<f32>
    }
    %loopValue1 = constant dense<1.0> : tile<f32>
    continue %loopValue1 : tile<f32>
}
```

See cuda_tile.break_0 for the full example listing.

### 8.5.3. cuda_tile.continue

*Continue to next loop iteration*

```
cuda_tile.continue %operands
```

#### Parameters

- **operands** (Variadic<Any>) - The values to yield to the parent loop.

#### Results

No results.

#### Description

The `continue` operation represents a block terminator that returns control to a loop operation, such as cuda_tile.for and cuda_tile.loop. The operation may yield any number of `$operands` to the parent loop upon termination.

The requirements and semantics of the `continue` operation are defined by the parent loop operation, see the loop operation's description for particular semantics.

The `continue` operation always returns control to the innermost enclosing loop operation, even when it is nested within other control constructs such as `if` or additional loops.

#### Constraints

- The operation must terminate its parent basic block.

#### Examples

```mlir
  %lowerBound = constant dense<0> : tile<i32>
  %upperBound = constant dense<10> : tile<i32>
  %step = constant dense<1> : tile<i32>
  %condition = constant dense<1> : tile<i1>
  // Continue from the body of a loop.
  for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
      continue
  }

  // Continue from an if nested within the loop.
  for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
      if %condition  {
          continue
      }
      // ...
  }

// Continue from an if nested within the loop, while yielding values.
%initVar0 = constant dense<0.0> : tile<f32>
%results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
          iter_values(%var0 = %initVar0) -> (tile<f32>)
  {
      if %condition {
          // ...
          yield
      } else {
          %loopValue0 = constant dense<1.0> : tile<f32>
          continue %loopValue0 : tile<f32>
      }
      %loopValue1 = constant dense<1.0> : tile<f32>
      continue %loopValue1 : tile<f32>
  }
```

See cuda_tile.continue_0 for the full example listing.

### 8.5.4. cuda_tile.for

*For loop over integer range*

```
cuda_tile.for %lowerBound %upperBound %step %initValues
```

#### Parameters

- **lowerBound** (tile<any>) - The lower bound of the loop.
- **upperBound** (tile<any>) - The upper bound of the loop.
- **step** (tile<any>) - The step of the loop.
- **initValues** (Variadic<Any>) - The initial values for the loop carried variables.

#### Results

- **resultValues** (Variadic<Any>) - The values of the loop-carried variables after loop termination.

#### Description

The `for` operation is a structured range-based sequential loop.

The loop operation consists of (1) a range formed by `lowerBound`, `upperBound`, and `step`, (2) a set of loop-carried values which are initialized by `initValues` and updated by each iteration of the loop, and (3) a region which represents the loop body.

The iteration space is defined by the interval `[lowerBound, upperBound)` with each value separated by `step`.

`lowerBound`, `upperBound`, and `step` must be of the same type. `lowerBound` and `upperBound` specify a half-open (or exclusive) range: the range includes the `lowerBound` but does not include the `upperBound`. `step` must be positive but the bounds may be negative or zero.

The first iteration of the loop receives the induction variable initialized to the value of `lowerBound` and the loop-carried values initialized to the values of `initValues`.

The loop body is executed for each value in the range, receiving an integer induction variable incremented by `step` on each iteration and the loop-carried values which correspond to the loop-carried values yielded by the previous loop iteration.

The loop terminates when the induction variable is greater than or equal to `upperBound`. By default, signed comparison is used between the upperBound and the induction variable. To use unsigned comparison instead, specify the optional `unsigned` unit attribute.

The body of the loop must be terminated by a cuda_tile.continue that yields the next iteration's value for each loop carried variable.

The for operation produces one return value for each loop carried variable. The type of the i-th return value is that of the i-th loop carried variable and its value is the final value of the i-th loop carried variable.

> **Warning**
> - Loop carried variables can not be a tensor_view or view type.
> - `for` operations cannot terminate early and must end in a cuda_tile.continue.

#### Constraints

- The operation must define scope when stack allocations are freed automatically.
- `lowerBound`, `upperBound` and `step` must have the same shape and element type (tile<any>).
- `initValues` and `resultValues` must have the same shape and element type (Variadic<Any>).
- The operation must provide custom parsing and printing methods.
- The operation only has an effect if and only if it the region's operation have an effect.
- Each provided region must contain exactly one block.

#### Examples

```mlir
%lowerBound = constant dense<0> : tile<i32>
%upperBound = constant dense<10> : tile<i32>
%step = constant dense<1> : tile<i32>

// A simple loop iterating over an i32 range.
for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
    continue
}

%initVal0 = constant dense<0.0> : tile<f32>
// A similar loop to the above, but with a loop carried value, val0.
%results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
                    iter_values(%val00 = %initVal0) -> (tile<f32>) {
  %loopVal0 = constant dense<1.0> : tile<f32>
  continue %loopVal0 : tile<f32>
}
```

See cuda_tile.for_0 for the full example listing.

### 8.5.5. cuda_tile.if

*Conditional execution*

```
cuda_tile.if %condition
```

#### Parameters

- **condition** (tile<i1>) - The condition of the if operation.

#### Results

- **results** (Variadic<Any>) - The results of the if operation.

#### Description

The `if` operation represents an if-then-else construct.

The `if` operation consists of (1) a control operand which is a `tile<i1>` value, (2) a true branch `thenRegion` and (3) an optional false branch `elseRegion`.

The `if` operation may produce results by yielding values in each branch using cuda_tile.yield.

If yielding value(s) the types of yielded values must match and the result result type of the `if` operation will be the same as the yielded values.

If yielding values the else branch is required and must also yield a value.

The values returned will be dependent on which branch is taken.

> **Warning**
> The `if` operation has a set of additional restrictions today:
> - Results of `if` must not be a tensor_view or view type.

#### Constraints

- All regions must have zero arguments.
- The operation must provide custom parsing and printing methods.
- The operation only has an effect if and only if it the region's operation have an effect.
- Each provided region must contain exactly one block.

#### Examples

```mlir
%condition = constant dense<1> : tile<i1>

// A simple if operation that conditionally executes a region.
if %condition  {
  // ...
}

// An if operation with an "else" branch.
if %condition  {
  // ...
} else {
  // ...
}

// An if operation that returns mixed types (f32,i32)
%x, %y = if %condition -> (tile<f32>, tile<i32>) {
  %x_then = constant dense<1.0> : tile<f32>
  %y_then = constant dense<2> : tile<i32>
  yield %x_then, %y_then : tile<f32>, tile<i32>
} else {
  %x_then = constant dense<1.0> : tile<f32>
  %y_then = constant dense<42> : tile<i32>
  yield %x_then, %y_then : tile<f32>, tile<i32>
}
```

See cuda_tile.if_0 for the full example listing.

### 8.5.6. cuda_tile.loop

*Loop until a break operation*

```
cuda_tile.loop %initValues
```

#### Parameters

- **initValues** (Variadic<Any>) - The initial values of the loop.

#### Results

- **resultValues** (Variadic<Any>) - The result values of the loop.

#### Description

The `loop` operation represents an, unstructured, infinite loop that executes until a cuda_tile.break is reached.

The loop consists of a (1) a set of loop-carried values which are initialized by `initValues` and updated by each iteration of the loop, and (2) a region which represents the loop body.

The loop will execute the body of the loop until a cuda_tile.break is dynamically executed.

Each control path of the loop must be terminated by:

- a cuda_tile.continue that yields the next iteration's value for each loop carried variable.
- a cuda_tile.break that terminates the loop and yields the final loop carried values.

As long as each loop iteration is terminated by one of these operations they may be combined with other control flow operations to express different control flow patterns.

The loop operation produces one return value for each loop carried variable. The type of the i-th return value is that of the i-th loop carried variable and its value is the final value of the i-th loop carried variable.

> **Warning**
> Loop operations have a set of additional restrictions today:
> - Early returns from inside loops are not supported, a code generator must first terminate the loop and then return if they wish to end the function execution entirely.
> - Loop carried variables can not be a tensor_view or view type.

#### Constraints

- The operation must define scope when stack allocations are freed automatically.
- `initValues` and `resultValues` must have the same shape and element type (Variadic<Any>).
- The operation must provide custom parsing and printing methods.
- The operation only has an effect if and only if it the region's operation have an effect.
- Each provided region must contain exactly one block.

#### Examples

```mlir
%initValue0 = constant dense<0.0> : tile<f32>
%results = loop iter_values(%var0 = %initValue0): tile<f32> -> tile<f32> {
    %condition = constant dense<1> : tile<i1>
    if %condition  {
        // ...
        yield
    } else {
        // %if.loopValue0 = ...
        %loopValue0 = constant dense<1.0> : tile<f32>
        break %loopValue0 : tile<f32>
    }
    %loopValue1 = constant dense<1.0> : tile<f32>
    continue %loopValue1 : tile<f32>
}
```

See cuda_tile.loop_0 for the full example listing.

#### Constraints

- The operation must define scope when stack allocations are freed automatically.
- The operation must provide custom parsing and printing methods.
- The operation only has an effect if and only if it the region's operation have an effect.
- Each provided region must contain exactly one block.

#### Examples

```mlir
// A simple "while-do" loop.
loop {
    %cond = constant dense<1> : tile<i1>
    if %cond {
        continue
    }
    break
}
```

See cuda_tile.loop_0 for the full example listing.

```mlir
// A simple "do-while" loop.
loop {
    //... body of the loop.

    %cond = constant dense<1> : tile<i1>
    if %cond {
        continue
    }
    break
}
```

See cuda_tile.loop_1 for the full example listing.

```mlir
%initValue0 = constant dense<0.0> : tile<f32>
// A loop that yields carried-iteration values, returning the final values.
%results = loop iter_values(%value0 = %initValue0) : tile<f32> -> tile<f32> {
    %cond = constant dense<1> : tile<i1>
    if %cond {
        %loopValue0 = constant dense<0.0> : tile<f32>
        continue %loopValue0 : tile<f32>
    }
    break %value0 : tile<f32>
}
```

See cuda_tile.loop_2 for the full example listing.

```mlir
%initValue0 = constant dense<0> : tile<i32>
// A loop that uses loop-carried values and returns a different type.
%results = loop iter_values(%value0 = %initValue0) : tile<i32> -> tile<f32> {
    %cond = constant dense<1> : tile<i1>

    if %cond {
        %newLoopValue = constant dense<0> : tile<i32>
        continue %newLoopValue : tile<i32>
    }

    %finalReturnValue = constant dense<0.0> : tile<f32>
    break %finalReturnValue : tile<f32>
}
```

See cuda_tile.loop_3 for the full example listing.

### 8.5.7. cuda_tile.return

*Return value(s) from function*

```
cuda_tile.return %operands
```

#### Parameters

- **operands** (Variadic<Any>) - The values to return.

#### Results

No results.

#### Description

The `return` operation returns control to the caller of a function.

> **Warning**
> Today the `return` operation has restricted semantics:
> - cuda_tile.entry operations do not produce return value(s) and thus `return` may be used to terminate the execution of the kernel by invoking the operation with no operands.
> - `return` can not be directly used inside of loop bodies to terminate the the execution of the kernel.

#### Constraints

- The operation must terminate its parent basic block.

#### Examples

```mlir
entry @foo() {
  %0 = constant dense<0> : tile<i32>
  %1 = constant dense<0.0> : tile<f16>
  // ...
  return
}
```

See cuda_tile.return_0 for the full example listing.

### 8.5.8. cuda_tile.yield

*Yield a value from the block*

```
cuda_tile.yield %operands
```

#### Parameters

- **operands** (Variadic<Any>) - The operands to yield to the parent operation.

#### Results

No results.

#### Description

The `yield` operation terminates a block that must yield control back to the parent operation such as `if`, `scan`, `reduce`.

The operation may yield any number of `$operands` to the parent upon termination. The number of values yielded and the execution semantics of how they are yielded are determined by the parent operation.

> **Note**
> Unlike standard MLIR control flow dialects `yield` is not used for loop control flow, see cuda_tile.break and cuda_tile.continue for loop control flow.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- The operation must terminate its parent basic block.

#### Examples

```mlir
%condition = constant dense<true> : tile<i1>
// Yield from the body of an if conditional.
if %condition  {
    yield
}

// Yield values from within an if conditional.
%x, %y = if %condition -> (tile<f32>, tile<f32>) {
    %x_then = constant dense<0.0> : tile<f32>
    %y_then = constant dense<1.0> : tile<f32>
    yield %x_then, %y_then : tile<f32>, tile<f32>
} else {
    %x_else = constant dense<2.0> : tile<f32>
    %y_else = constant dense<3.0> : tile<f32>
    yield %x_else, %y_else : tile<f32>, tile<f32>
}
```

See cuda_tile.yield_0 for the full example listing.

## 8.6. Memory

**Tile IR** contains a set of memory operations which enable loading, storing, and manipulating memory.

There are a few families of memory operations in **Tile IR**:

- Tile of pointer based memory operations such as cuda_tile.load_ptr_tko and cuda_tile.store_ptr_tko which load and store tiles from and to global memory.
- View based memory operations such as cuda_tile.load_view_tko and cuda_tile.store_view_tko which load and store tiles from and to views.
- Atomic memory operations such as cuda_tile.atomic_rmw_tko and cuda_tile.atomic_cas_tko which perform atomic operations on global memory.

Currently all memory operations are token-ordered; the ordering between any pair of memory operations is undefined unless connected by tokens. For more discussion on token-ordered operations see section-memory_model.

> **Warning**
> Reading or writing of bound of any allocation is undefined behavior. Examples of out of bounds access are:
> - Pointer memory operations to tiles containing elements outside the allocation, for example offseting passed the end of the allocation.
> - Associating an invalid layout with a base pointer, that describes a striding or shape that over runs the allocation and then indexing into the view.
> - Indexing into a view with indices that are out of bounds.

> **Note**
> The rules of what consititues out of bounds is modified when using padded views or masking, see Type System for more details on specific types.

### 8.6.1. cuda_tile.join_tokens

*Product a new token which depends on the input tokens*

```
cuda_tile.join_tokens %tokens
```

#### Parameters

- **tokens** (Variadic<token>) - The input tokens to join.

#### Results

- **result** (token) - The joined token.

#### Description

The `join_tokens` operation produces a fresh token which depends on all input tokens. Token-ordered operations which consume the new token will then be ordered with respect to all joined tokens.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- The operation's result type may be inferred from its operands and attributes.

### 8.6.2. cuda_tile.load_ptr_tko

*Load and gather data from global memory using a pointer tile without ordering guarantees*

```
cuda_tile.load_ptr_tko %memory_ordering_semantics %memory_scope %source %mask %paddingValue %token %optimization_hints
```

#### Parameters

- **memory_ordering_semantics** (MemoryOrderingSemantics) - The memory ordering semantics for the load operation.
- **memory_scope** (MemoryScope) - The memory scope for the atomic operation.
- **source** (ptr) - The source tile of pointers.
- **mask** (tile<i1>) - The mask for the load operation.
- **paddingValue** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The padding value for the load operation.
- **token** (token) - The token for the load operation.
- **optimization_hints** (OptimizationHints) - Optimization hints for operation

#### Results

- **result** (tile) - The result of the load operation.
- **result_token** (token) - The result token of the load operation.

#### Description

This `load` OP performs a gather operation by loading a tile of data from global memory into a result tile based on a tile of pointers provided by the `source` operand.

The `source` operand is a tile of pointers, which specifies the memory locations from which the data is gathered. The operation loads this data and returns it as the `result` tile. When loading i1 values, each value is loaded from a full byte in memory. Any nonzero byte is canonicalized to 0x01, and zero bytes become 0x00.

Optionally, a `mask` operand can be provided to control the gathering of elements. If present, only the elements specified by the `mask` are loaded. The shape of the `mask` must match the shape of the `result`.

When `mask` is present one `paddingValue` can be optionally present as well. The `paddingValue` must have the same shape of the `source` tile. If it is not present, the value of masked elements are undefined.

Token-ordered operations are not constrained by program order. The compiler may reorder them (i.e. place them earlier or later in program order) unless further constrained by tokens.

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required.

- `weak` - No concurrent accesses to the source/destination location.
- `relaxed` - There may be concurrent access to the location, but this access does not establish a happens-before relationship.
- `acquire` - There may be concurrent accesses to the location. If this acquire observes a release operation, then happens before is established.

Note: The following variants are not supported by this operation: `release`, `acq_rel`.

The `memory_scope` attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

- `tl_blk` - There may be concurrent accesses from within the same tile block.
- `device` - There may be concurrent accesses from within the same device (i.e., GPU).
- `sys` - There may be concurrent accesses from anywhere within the system (i.e., all devices).

The `optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries.

The hints are specified for each architecture (e.g., `sm_100`, `sm_120`) and for each architecture the user can specify specific hints for each operation.

- `num_cta_in_cga` - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for cuda_tile.entry.
- `allow_tma` - suggest whether to use TMA for cuda_tile.load_view_tko and cuda_tile.store_view_tko.
- `latency` - latency hint for cuda_tile.load_view_tko and cuda_tile.store_view_tko.

#### Constraints

- The operation must encode variadic operand segment sizes in attributes.
- `source` type is expected a pointer type of `result` type
- shape of 'mask' must match the shape of 'source'
- type of 'paddingValue' must match the type of 'result'

#### Examples

```mlir
%mask = constant dense<1> : tile<i1>
%padding = constant dense<0.0> : tile<f32>

  // Load without token.
  %result0, %res_token0 = load_ptr_tko weak %ptr, %mask, %padding
      : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token

  // Load with token.
  %token0 = make_token : token
  %result1, %res_token1 = load_ptr_tko weak %ptr, %mask, %padding token=%token0
      : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token

  return
```

See cuda_tile.load_ptr_tko_0 for the full example listing.

### 8.6.3. cuda_tile.make_token

*Create a fresh token with no prior dependencies*

```
cuda_tile.make_token
```

#### Parameters

No parameters.

#### Results

- **result** (token) - A fresh token with no prior dependencies.

#### Description

The `make_token` operation creates a fresh token with no prior dependencies.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- The operation's result type may be inferred from its operands and attributes。

### 8.6.4. cuda_tile.store_ptr_tko

*Store and scatter data from pointer of tile to global memory without ordering guarantees*

```
cuda_tile.store_ptr_tko %memory_ordering_semantics %memory_scope %destination %value %mask %token %optimization_hints
```

#### Parameters

- **memory_ordering_semantics** (MemoryOrderingSemantics) - The memory ordering semantics.
- **memory_scope** (MemoryScope) - The optional memory scope.
- **destination** (ptr) - The destination pointer tile.
- **value** (tile) - The value tile to store.
- **mask** (tile<i1>) - The optional mask for selective storage.
- **token** (token) - The optional token for operation ordering.
- **optimization_hints** (OptimizationHints) - Optimization hints for operation

#### Results

- **result_token** (token) - The result token for synchronization.

#### Description

The `store` operation performs a scatter by storing a tile of data from a tile into global memory.

The `destination` operand is a tile of pointers indicating the global memory locations where data from the `value` tile will be stored. When storing i1 values, each value occupies a full byte in memory. Any nonzero byte is canonicalized to 0x01, and zero bytes become 0x00.

Additionally, the operation supports an optional `mask` operand, which allows selective scattering of elements. If provided, only the elements specified by the `mask` are stored. The shape of the `mask` must align with the shape of the `value` tile.

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required.

- `weak` - No concurrent accesses to the source/destination location.
- `relaxed` - There may be concurrent access to the location, but this access does not establish a happens-before relationship.
- `release` - There may be concurrent access to the location. If this release is observed with an acquire operation, then happens before is established.

Note: The following variants are not supported by this operation: `acquire`, `acq_rel`.

The `memory_scope` attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

- `tl_blk` - There may be concurrent accesses from within the same tile block.
- `device` - There may be concurrent accesses from within the same device (i.e., GPU).
- `sys` - There may be concurrent accesses from anywhere within the system (i.e., all devices).

#### Constraints

- The operation must encode variadic operand segment sizes in attributes.
- `destination` type is expected a pointer type of `value` type
- shape of 'destination' must match the shape of 'mask'
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
%mask = constant dense<1> : tile<i1>

  // Store without token.
  %res_token0 = store_ptr_tko weak %ptr, %value, %mask
      : tile<ptr<f32>>, tile<f32>, tile<i1> -> token

  // Store with token.
  %token0 = make_token : token
  %res_token1 = store_ptr_tko weak %ptr, %value, %mask token=%token0
      : tile<ptr<f32>>, tile<f32>, tile<i1> -> token

  return
```

See cuda_tile.store_ptr_tko_0 for the full example listing.

## 8.7. Floating Point

**Tile IR** contains a set of typed arithmetic operations which implement familiar arithmetic operations on floating-point types for integer operations see Integer.

All operations are implemented in a manner that is efficient for the target architecture and device family. In most common cases this means utilizing the underlying hardware's native floating-point operations. Due to **Tile IR**'s stability guarantees and higher-level programming model some types on some hardware may be emulated, see Stability for more information about the stability guarantees and information about per device behavior.

### 8.7.1. Floating-Point Arithmetic

Standard floating-point types implement the IEEE-754 standard for floating-point arithmetic. On NVIDIA hardware, certain types are non-standard and *do not* implement the IEEE-754 standard, see Element Types for more details about the different floating-point types, their precision, storage, and formats.

Supports 16-bit, 32-bit, and 64-bit floating-point data types.

### 8.7.2. Floating-Point Math

**Tile IR** contains a set of standard math library operations which implement familiar mathematical functions over tensors supporting 16-bit, 32-bit, and 64-bit floating-point data types.

> **Note**
> 32-bit and 64-bit operations typically leverage efficient hardware-specific instructions. Some 16-bit operations are emulated using wider intermediate computations, and may not offer the same performance.

> **Warning**
> There are some restrictions based on data type support which are detailed in the Type System section.

### 8.7.3. cuda_tile.absf

*Element-wise floating-point absolute value*

```
cuda_tile.absf %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input float tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The absolute value of the input tile.

#### Description

The `absf` operation computes the element-wise absolute value of the input float tile.

Element-wise floating-point arithmetic operations are performed by the target architecture's native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See Floating Point for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same shape.
- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- The operation's result type may be inferred from its operands and attributes.

### 8.7.4. cuda_tile.addf

*Element-wise floating-point addition*

```
cuda_tile.addf %lhs %rhs %rounding_mode %flush_to_zero
```

#### Parameters

- **lhs** (tile<f16 | bf16 | f32 | f64>) - The left hand side operand.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The right hand side operand.
- **rounding_mode** (RoundingMode) - The rounding mode for the operation.
- **flush_to_zero** (Flag) - If set, flushes subnormal inputs and results to sign-preserving zero.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The sum of the input tiles.

#### Description

The `addf` operation computes the element-wise addition of two tiles with floating-point element type.

The addition of individual elements is performed by the target architecture's native floating-point addition for the given element type unless otherwise specified.

Element-wise floating-point arithmetic operations are performed by the target architecture's native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See Floating Point for more details.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- The operation's result type may be inferred from its operands and attributes.

### 8.7.5. cuda_tile.ceil

*Element-wise ceiling*

```
cuda_tile.ceil %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input float tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The ceiling of the input tile.

#### Description

The `ceil` operation computes the element-wise ceiling on the input floating-point tile. The ceiling operation rounds each element up to the largest integer value that is greater than or equal to the input value.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
%result = ceil %source : tile<f32>
```

See cuda_tile.ceil_0 for the full example listing.

### 8.7.6. cuda_tile.cmpf

*Element-wise floating-point comparison*

```
cuda_tile.cmpf %comparison_predicate %comparison_ordering %lhs %rhs
```

#### Parameters

- **comparison_predicate** (ComparisonPredicate) - The comparison predicate.
- **comparison_ordering** (ComparisonOrdering) - The comparison ordering.
- **lhs** (tile<f16 | bf16 | f32 | f64>) - The left hand side operand.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The right hand side operand.

#### Results

- **result** (tile<i1>) - The result of the comparison.

#### Description

The `cmpf` operation is a generic comparison for float-like types. The operands must have the same shape and type, and this type must be a float type.

The result is `1` if the comparison is true and `0` otherwise. The comparison is performed element-wise and the element of the result indicates whether the comparison is true for the operand elements with the same indices as those of the result.

The `comparison_predicate` attribute specifies the kind of comparison to be performed.

- `equal` - Equal comparison.
- `not_equal` - Not equal comparison.
- `less_than` - Less than comparison.
- `less_than_or_equal` - Less than or equal comparison.
- `greater_than` - Greater than comparison.
- `greater_than_or_equal` - Greater than or equal comparison.

The `comparison_ordering` attribute specifies the kind of ordering to be performed in the comparison operation.

- `unordered` - Unordered comparison.
- `ordered` - Ordered comparison.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs` and `rhs` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- Result type has i1 element type and same shape as operands
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
%lhs0 = constant dense<0.0> : tile<f16>
%rhs0 = constant dense<0.0> : tile<f16>

// Custom form of scalar "ordered equal" comparison.
%x0 = cmpf equal ordered %lhs0, %rhs0 : tile<f16>

%lhs1 = constant dense<0.0> : tile<2x2xf16>
%rhs1 = constant dense<0.0> : tile<2x2xf16>

// Custom form of scalar "unordered less than" comparison.
%x2 = cmpf less_than unordered %lhs1, %rhs1 : tile<2x2xf16>

%lhs2 = constant dense<0.0> : tile<2x2xf64>
%rhs2 = constant dense<0.0> : tile<2x2xf64>
```

See cuda_tile.cmpf_0 for the full example listing.

### 8.7.7. cuda_tile.cosh

*Element-wise hyperbolic cosine*

```
cuda_tile.cosh %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input floating-point tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The hyperbolic cosine of the input tile.

#### Description

The `cosh` operation computes the element-wise hyperbolic cosine of the input tile with floating-point element type.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- The operation's result type may be inferred from its operands and attributes.

### 8.7.8. cuda_tile.cos

*Element-wise cosine*

```
cuda_tile.cos %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input float tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The cosine of the input tile.

#### Description

The `cos` operation computes the element-wise cosine of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
%in = constant dense<[0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = cos %in : tile<4xf32>
```

See cuda_tile.cos_0 for the full example listing.

### 8.7.9. cuda_tile.divf

*Element-wise floating-point division*

```
cuda_tile.divf %lhs %rhs %rounding_mode %flush_to_zero
```

#### Parameters

- **lhs** (tile<f16 | bf16 | f32 | f64>) - The dividend input floating-point tile.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The divisor input floating-point tile.
- **rounding_mode** (RoundingMode) - The rounding mode for the operation.
- **flush_to_zero** (Flag) - If set, flushes subnormal inputs and results to sign-preserving zero.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The quotient of the input tiles.

#### Description

The `divf` operation computes the element-wise quotient of two tiles with floating-point element type.

The division of individual elements is performed by the target architecture's native floating-point division for the given element type unless otherwise specified.

Element-wise floating-point arithmetic operations are performed by the target architecture's native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See Floating Point for more details.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- The operation's result type may be inferred from its operands and attributes.

### 8.7.10. cuda_tile.exp2

*Element-wise power of two*

```
cuda_tile.exp2 %source %flush_to_zero
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input floating-point tile.
- **flush_to_zero** (Flag) - If set, flushes subnormal inputs and results to sign-preserving zero.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The result of raising 2 to the power of the input tile.

#### Description

The `exp2` operation computes the element-wise power of two of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

The below table shows the supported modifiers for each data type.

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
|----------|---------|---------|----------|---------|
| `flush_to_zero` | yes | no | no | no |

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
%in = constant dense<[0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = exp2 %in : tile<4xf32>
```

See cuda_tile.exp2_0 for the full example listing.

### 8.7.11. cuda_tile.exp

*Element-wise exponential*

```
cuda_tile.exp %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input float tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The exponential of the input tile.

#### Description

The `exp` operation computes the element-wise exponential of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- The operation's result type may be inferred from its operands and attributes.

### 8.7.12. cuda_tile.floor

*Element-wise floor rounding*

```
cuda_tile.floor %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input tile to the floor operation.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The result of the floor operation.

#### Description

The `floor` operation computes the element-wise floor on the input floating-point tile rounding each element down to the largest integer that is less than or equal to the element.

Element-wise floating-point arithmetic operations are performed by the target architecture's native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See Floating Point for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
%source = constant dense<1.5> : tile<f32>
%result = floor %source : tile<f32>
```

See cuda_tile.floor_0 for the full example listing.

### 8.7.13. cuda_tile.fma

*Floating point fused multipy-add*

```
cuda_tile.fma %lhs %rhs %acc %rounding_mode %flush_to_zero
```

#### Parameters

- **lhs** (tile<f16 | bf16 | f32 | f64>) - The left hand side operand.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The right hand side operand.
- **acc** (tile<f16 | bf16 | f32 | f64>) - The accumulator operand.
- **rounding_mode** (RoundingMode) - The rounding mode for the operation.
- **flush_to_zero** (Flag) - If set, flushes subnormal inputs and results to sign-preserving zero.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The result of `lhs * rhs + acc`.

#### Description

The `fma` operation computes the fused multiply-add of three input tiles with floating-point element type. It performs the operation `lhs * rhs + acc` with a single rounding step.

The multiplication and addition of individual elements is performed by the target architecture's native floating-point fused multiply-add instruction for the given element type unless otherwise specified.

Element-wise floating-point arithmetic operations are performed by the target architecture's native floating-point instructions. If the `rounding` modifier is specified, the particular rounding mode will be applied to each element of the result. See Floating Point for more details.

The `rounding` attribute specifies the rounding mode to use for the operation.

- `nearest_even` - Round to nearest (ties to even).
- `zero` - Round towards zero (truncate).
- `negative_inf` - Round towards negative infinity.
- `positive_inf` - Round towards positive infinity.
- `approx` - Approximate rounding mode.
- `full` - Full precision rounding mode.
- `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs`, `acc` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- The operation's result type may be inferred from its operands and attributes.

### 8.7.14. cuda_tile.log10

*Element-wise base-10 logarithm*

```
cuda_tile.log10 %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input floating-point tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The base-10 logarithm of the input tile.

#### Description

The `log10` operation computes the element-wise base-10 logarithm of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
- The operation's result type may be inferred from its operands and attributes.

### 8.7.15. cuda_tile.log1p

*Element-wise logarithm of (1 + x)*

```
cuda_tile.log1p %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input floating-point tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The logarithm of (1 + source).

#### Description

The `log1p` operation computes the element-wise natural logarithm of (1 + source) for the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.16. cuda_tile.log2

*Element-wise base-2 logarithm*

```
cuda_tile.log2 %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input floating-point tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The base-2 logarithm of the input tile.

#### Description

The `log2` operation computes the element-wise base-2 logarithm of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.17. cuda_tile.log

*Element-wise natural logarithm*

```
cuda_tile.log %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input float tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The natural logarithm of the input tile.

#### Description

The `log` operation computes the element-wise natural logarithm of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.18. cuda_tile.maxf

*Element-wise floating-point maximum*

```
cuda_tile.maxf %lhs %rhs %comparison_ordering
```

#### Parameters

- **lhs** (tile<f16 | bf16 | f32 | f64>) - The left hand side operand.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The right hand side operand.
- **comparison_ordering** (ComparisonOrdering) - The comparison ordering.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The element-wise maximum of lhs and rhs.

#### Description

The `maxf` operation computes the element-wise floating-point maximum of two tiles with floating-point element type.

The comparison behavior for NaNs follows the IEEE-754 standard:

- If both operands are NaN, returns NaN.
- If one operand is NaN, returns the non-NaN operand.
- Otherwise, returns the larger of the two operands.

This matches the behavior of std::fmax in C++.

NaNs are propagated per the IEEE-754 rules for NaN handling.

The `comparison_ordering` attribute specifies the kind of ordering to be performed in the comparison operation.

- `unordered` - Unordered comparison (NaN compares as false).
- `ordered` - Ordered comparison.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.19. cuda_tile.maximumf

*Element-wise floating-point maximum (propagates NaN)*

```
cuda_tile.maximumf %lhs %rhs %comparison_ordering
```

#### Parameters

- **lhs** (tile<f16 | bf16 | f32 | f64>) - The left hand side operand.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The right hand side operand.
- **comparison_ordering** (ComparisonOrdering) - The comparison ordering.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The element-wise maximum of lhs and rhs.

#### Description

The `maximumf` operation computes the element-wise floating-point maximum of two tiles with floating-point element type.

The comparison behavior differs from maxf:

- If either operand is NaN, returns NaN.
- Otherwise, returns the larger of the two operands.

This matches the behavior of numpy.maximum and JAX's lax.max.

The `comparison_ordering` attribute specifies the kind of ordering to be performed in the comparison operation.

- `unordered` - Unordered comparison.
- `ordered` - Ordered comparison.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.20. cuda_tile.minf

*Element-wise floating-point minimum*

```
cuda_tile.minf %lhs %rhs %comparison_ordering
```

#### Parameters

- **lhs** (tile<f16 | bf16 | f32 | f64>) - The left hand side operand.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The right hand side operand.
- **comparison_ordering** (ComparisonOrdering) - The comparison ordering.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The element-wise minimum of lhs and rhs.

#### Description

The `minf` operation computes the element-wise floating-point minimum of two tiles with floating-point element type.

The comparison behavior for NaNs follows the IEEE-754 standard:

- If both operands are NaN, returns NaN.
- If one operand is NaN, returns the non-NaN operand.
- Otherwise, returns the smaller of the two operands.

This matches the behavior of std::fmin in C++.

NaNs are propagated per the IEEE-754 rules for NaN handling.

The `comparison_ordering` attribute specifies the kind of ordering to be performed in the comparison operation.

- `unordered` - Unordered comparison.
- `ordered` - Ordered comparison.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.21. cuda_tile.minimumf

*Element-wise floating-point minimum (propagates NaN)*

```
cuda_tile.minimumf %lhs %rhs %comparison_ordering
```

#### Parameters

- **lhs** (tile<f16 | bf16 | f32 | f64>) - The left hand side operand.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The right hand side operand.
- **comparison_ordering** (ComparisonOrdering) - The comparison ordering.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The element-wise minimum of lhs and rhs.

#### Description

The `minimumf` operation computes the element-wise floating-point minimum of two tiles with floating-point element type.

The comparison behavior differs from minf:

- If either operand is NaN, returns NaN.
- Otherwise, returns the smaller of the two operands.

This matches the behavior of numpy.minimum and JAX's lax.min.

The `comparison_ordering` attribute specifies the kind of ordering to be performed in the comparison operation.

- `unordered` - Unordered comparison.
- `ordered` - Ordered comparison.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.22. cuda_tile.mulf

*Element-wise floating-point multiplication*

```
cuda_tile.mulf %lhs %rhs %rounding_mode %flush_to_zero
```

#### Parameters

- **lhs** (tile<f16 | bf16 | f32 | f64>) - The left hand side operand.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The right hand side operand.
- **rounding_mode** (RoundingMode) - The rounding mode for the operation.
- **flush_to_zero** (Flag) - If set, flushes subnormal inputs and results to sign-preserving zero.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The product of lhs and rhs.

#### Description

The `mulf` operation computes the element-wise product of two tiles with floating-point element type.

The multiplication of individual elements is performed by the target architecture's native floating-point multiplication for the given element type unless otherwise specified.

The `rounding` attribute specifies the rounding mode.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.23. cuda_tile.negf

*Element-wise floating-point negation*

```
cuda_tile.negf %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input float tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The negation of the input tile.

#### Description

The `negf` operation computes the element-wise negation of the input floating-point tile.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.24. cuda_tile.powf

*Element-wise power operation*

```
cuda_tile.powf %lhs %rhs
```

#### Parameters

- **lhs** (tile<f16 | bf16 | f32 | f64>) - The base tile.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The exponent tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The result of lhs raised to the power of rhs.

#### Description

The `powf` operation computes the element-wise power function (lhs^rhs) of two tiles with floating-point element type.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.25. cuda_tile.recipf

*Element-wise reciprocal*

```
cuda_tile.recipf %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input floating-point tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The reciprocal (1/x) of the input tile.

#### Description

The `recipf` operation computes the element-wise reciprocal of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.26. cuda_tile.remf

*Element-wise floating-point remainder*

```
cuda_tile.remf %lhs %rhs %rounding_mode
```

#### Parameters

- **lhs** (tile<f16 | bf16 | f32 | f64>) - The dividend tile.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The divisor tile.
- **rounding_mode** (RoundingMode) - The rounding mode for the operation.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The remainder of lhs divided by rhs.

#### Description

The `remf` operation computes the element-wise floating-point remainder of two tiles with floating-point element type.

The remainder operation follows IEEE-754 semantics.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.27. cuda_tile.rsqrt

*Element-wise reciprocal square root*

```
cuda_tile.rsqrt %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input floating-point tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The reciprocal square root of the input tile.

#### Description

The `rsqrt` operation computes the element-wise reciprocal square root of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.28. cuda_tile.sigmoid

*Element-wise sigmoid function*

```
cuda_tile.sigmoid %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input floating-point tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The sigmoid of the input tile.

#### Description

The `sigmoid` operation computes the element-wise sigmoid function (1 / (1 + exp(-x))) of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.29. cuda_tile.sinh

*Element-wise hyperbolic sine*

```
cuda_tile.sinh %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input floating-point tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The hyperbolic sine of the input tile.

#### Description

The `sinh` operation computes the element-wise hyperbolic sine of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.30. cuda_tile.sin

*Element-wise sine*

```
cuda_tile.sin %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input float tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The sine of the input tile.

#### Description

The `sin` operation computes the element-wise sine of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.31. cuda_tile.sqrt

*Element-wise square root*

```
cuda_tile.sqrt %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input floating-point tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The square root of the input tile.

#### Description

The `sqrt` operation computes the element-wise square root of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.32. cuda_tile.subf

*Element-wise floating-point subtraction*

```
cuda_tile.subf %lhs %rhs %rounding_mode %flush_to_zero
```

#### Parameters

- **lhs** (tile<f16 | bf16 | f32 | f64>) - The left hand side operand.
- **rhs** (tile<f16 | bf16 | f32 | f64>) - The right hand side operand.
- **rounding_mode** (RoundingMode) - The rounding mode for the operation.
- **flush_to_zero** (Flag) - If set, flushes subnormal inputs and results to sign-preserving zero.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The difference of lhs and rhs.

#### Description

The `subf` operation computes the element-wise difference of two tiles with floating-point element type.

The subtraction of individual elements is performed by the target architecture's native floating-point subtraction for the given element type unless otherwise specified.

The `rounding` attribute specifies the rounding mode.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.33. cuda_tile.tanhf

*Element-wise hyperbolic tangent (alternative)

```
cuda_tile.tanhf %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input floating-point tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The hyperbolic tangent of the input tile.

#### Description

Alternative name for the hyperbolic tangent operation. See cuda_tile.tanh for details.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

### 8.7.34. cuda_tile.tanh

*Element-wise hyperbolic tangent*

```
cuda_tile.tanh %source
```

#### Parameters

- **source** (tile<f16 | bf16 | f32 | f64>) - The input float tile.

#### Results

- **result** (tile<f16 | bf16 | f32 | f64>) - The hyperbolic tangent of the input tile.

#### Description

The `tanh` operation computes the element-wise hyperbolic tangent of the input floating-point tile.

This operation is emulated in `f32` when executed on half-precision inputs (`f16` and `bf16`). See Floating Point for more details.

#### Constraints

- `source` and `result` must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).

## 8.8. Integer

**Tile IR** contains a set of typed arithmetic operations which implement familiar arithmetic operations on integer types.

### 8.8.1. Integer Arithmetic

Integer arithmetic operations support the full range of integer types: `i1`, `i8`, `i16`, `i32`, and `i64`.

### 8.8.2. cuda_tile.addi

*Element-wise integer addition*

```
cuda_tile.addi %lhs %rhs %overflow_behavior
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand.
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand.
- **overflow_behavior** (OverflowBehavior) - The overflow behavior for the operation.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The sum of lhs and rhs.

#### Description

The `addi` operation computes the element-wise sum of two tiles with integer element type.

The addition of individual elements is performed by the target architecture's native integer addition for the given element type unless otherwise specified.

If an overflow occurs, the `overflow` modifier determines the behavior.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type (tile<i1 | i8 | i16 | i32 | i64>).

### 8.8.2. cuda_tile.absi

*Element-wise integer absolute value*

```
cuda_tile.absi %source
```

#### Parameters

- **source** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The absolute value of the input tile.

#### Description

The `absi` operation computes the absolute value of the input integer tile.

The input tile is always interpreted as a signed integer. The output tile is always interpreted as an unsigned integer.

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow.

#### Constraints

- `source` and `result` must have the same shape and element type.

### 8.85. cuda_tile.divi

*Element-wise integer division*

```
cuda_tile.divi %lhs %rhs %signedness
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The dividend operand.
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The divisor operand.
- **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned`.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The quotient of lhs divided by rhs.

#### Description

The `divi` operation computes the element-wise quotient of two tiles with integer element types.

The `signedness` attribute specifies the signedness of operand(s):
- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

Division by zero produces undefined results.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type.

### 8.8.6. cuda_tile.maxi

*Element-wise integer maximum*

```
cuda_tile.maxi %lhs %rhs %signedness
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand.
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand.
- **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned`.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The element-wise maximum of lhs and rhs.

#### Description

The `maxi` operation computes the element-wise maximum between the two input tiles with integer element types.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type.

### 8.8.7. cuda_tile.mini

*Element-wise integer minimum*

```
cuda_tile.mini %lhs %rhs %signedness
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand.
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand.
- **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned`.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The element-wise minimum of lhs and rhs.

#### Description

The `mini` operation computes the element-wise minimum between the two input tiles with integer element types.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type.

### 8.8.8. cuda_tile.mulhi

*Element-wise multiply high (unsigned)*

```
cuda_tile.mulhi %lhs %rhs
```

#### Parameters

- **lhs** (tile<i8 | i16 | i32 | i64>) - The left hand side operand.
- **rhs** (tile<i8 | i16 | i32 | i64>) - The right hand side operand.

#### Results

- **result** (tile<i8 | i16 | i32 | i64>) - The high bits of the multiplication result.

#### Description

The `mulhi` operation produces the most significant N bits of the 2N-bit product of two N-bit integer tiles. For `i64`, this is the most significant 64 bits of the full 128-bit product; for `i8`, it is the most significant 8 bits of the full 16-bit product; etc.

This is in contrast to `muli`, which produces the lower N bits of the 2N-bit product.

The `mulhi` operation is only defined for unsigned integers.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type.

### 8.8.9. cuda_tile.muli

*Element-wise integer multiplication*

```
cuda_tile.muli %lhs %rhs %overflow
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand.
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand.
- **overflow** (OverflowBehavior) - The overflow behavior for the operation.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The product of lhs and rhs.

#### Description

The `muli` operation computes the element-wise product of two tiles with integer element types.

The multiplication of individual elements is performed by the target architecture's native integer multiplication for the given element type unless otherwise specified.

If an overflow occurs, the `overflow` modifier determines the behavior.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type.

### 8.8.10. cuda_tile.negsi

*Element-wise integer negation*

```
cuda_tile.negsi %source
```

#### Parameters

- **source** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The negation of the input tile.

#### Description

The `negsi` operation computes the element-wise negation of the input integer tile.

#### Constraints

- `source` and `result` must have the same shape and element type.

### 8.8.11. cuda_tile.remi

*Element-wise integer remainder*

```
cuda_tile.remi %lhs %rhs %signedness
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The dividend operand.
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The divisor operand.
- **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned`.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The remainder of lhs divided by rhs.

#### Description

The `remi` operation computes the element-wise remainder of two tiles with integer element types.

The `signedness` attribute specifies the signedness of operand(s):
- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

Division by zero produces undefined results.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type.

### 8.8.12. cuda_tile.shli

*Element-wise left shift*

```
cuda_tile.shli %value %amount
```

#### Parameters

- **value** (tile<i1 | i8 | i16 | i32 | i64>) - The value to shift.
- **amount** (tile<i8 | i16 | i32 | i64>) - The amount to shift by.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The result of shifting value left by amount.

#### Description

The `shli` operation computes the element-wise left shift of the input tile by the specified amount.

#### Constraints

- `value` and `result` must have the same shape and element type.

### 8.8.13. cuda_tile.shri

*Element-wise right shift*

```
cuda_tile.shri %value %amount %signedness
```

#### Parameters

- **value** (tile<i1 | i8 | i16 | i32 | i64>) - The value to shift.
- **amount** (tile<i8 | i16 | i32 | i64>) - The amount to shift by.
- **signedness** (Signedness) - Interpret as `signed` (arithmetic) or `unsigned` (logical) shift.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The result of shifting value right by amount.

#### Description

The `shri` operation computes the element-wise right shift of the input tile by the specified amount.

The `signedness` attribute determines the type of right shift:
- `signed` - Arithmetic right shift (sign bit is preserved).
- `unsigned` - Logical right shift (zeros fill from the left).

#### Constraints

- `value` and `result` must have the same shape and element type.

### 8.8.14. cuda_tile.subi

*Element-wise integer subtraction*

```
cuda_tile.subi %lhs %rhs %overflow
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand.
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand.
- **overflow** (OverflowBehavior) - The overflow behavior for the operation.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The difference of lhs and rhs.

#### Description

The `subi` operation computes the element-wise subtraction of two tiles with integer element types.

The subtraction of individual elements is performed by the target architecture's native integer subtraction for the given element type unless otherwise specified.

If an overflow occurs, the `overflow` modifier determines the behavior.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type.

## 8.9. Bitwise

**Tile IR** contains bitwise operations for manipulating integer tiles at the bit level.

### 8.9.1. cuda_tile.andi

*Element-wise bitwise AND*

```
cuda_tile.andi %lhs %rhs
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand.
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The bitwise AND of lhs and rhs.

#### Description

The `andi` operation computes the element-wise bitwise AND of two tiles with integer element types.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type.

### 8.9.2. cuda_tile.ori

*Element-wise bitwise OR*

```
cuda_tile.ori %lhs %rhs
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand.
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The bitwise OR of lhs and rhs.

#### Description

The `ori` operation computes the element-wise bitwise OR of two tiles with integer element types.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type.

### 8.9.3. cuda_tile.xori

*Element-wise bitwise XOR*

```
cuda_tile.xori %lhs %rhs
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand.
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The bitwise XOR of lhs and rhs.

#### Description

The `xori` operation computes the element-wise bitwise XOR of two tiles with integer element types.

#### Constraints

- `lhs`, `rhs` and `result` must have the same shape and element type.

### 8.9.4. cuda_tile.noti

*Element-wise bitwise NOT*

```
cuda_tile.noti %source
```

#### Parameters

- **source** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile.

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The bitwise NOT of the input tile.

#### Description

The `noti` operation computes the element-wise bitwise NOT of the input integer tile.

#### Constraints

- `source` and `result` must have the same shape and element type.

### 8.9.5. cuda_tile.popcnt

*Element-wise population count*

```
cuda_tile.popcnt %source
```

#### Parameters

- **source** (tile<i8 | i16 | i32 | i64>) - The input integer tile.

#### Results

- **result** (tile<i8 | i16 | i32 | i64>) - The population count of each element.

#### Description

The `popcnt` operation counts the number of set bits (1s) in each element of the input tile.

#### Constraints

- `source` and `result` must have the same shape and element type.

### 8.9.6. cuda_tile.clz

*Element-wise count leading zeros*

```
cuda_tile.clz %source
```

#### Parameters

- **source** (tile<i8 | i16 | i32 | i64>) - The input integer tile.

#### Results

- **result** (tile<i8 | i16 | i32 | i64>) - The count of leading zeros in each element.

#### Description

The `clz` operation counts the number of leading zeros in the binary representation of each element.

#### Constraints

- `source` and `result` must have the same shape and element type.

### 8.9.7. cuda_tile.ctz

*Element-wise count trailing zeros*

```
cuda_tile.ctz %source
```

#### Parameters

- **source** (tile<i8 | i16 | i32 | i64>) - The input integer tile.

#### Results

- **result** (tile<i8 | i16 | i32 | i64>) - The count of trailing zeros in each element.

#### Description

The `ctz` operation counts the number of trailing zeros in the binary representation of each element.

#### Constraints

- `source` and `result` must have the same shape and element type.

### 8.9.8. cuda_tile.brev

*Element-wise bit reversal*

```
cuda_tile.brev %source
```

#### Parameters

- **source** (tile<i8 | i16 | i32 | i64>) - The input integer tile.

#### Results

- **result** (tile<i8 | i16 | i32 | i64>) - The bit-reversed input tile.

#### Description

The `brev` operation reverses the bits in each element of the input tile.

#### Constraints

- `source` and `result` must have the same shape and element type.

## 8.10. Atomics

**Tile IR** provides atomic operations for thread-safe memory manipulation.

### 8.10.1. cuda_tile.atomic_cas_tko

*Atomic compare-and-swap*

```
cuda_tile.atomic_cas_tko %memory_ordering_semantics %memory_scope %ptr %expected %desired %mask %token
```

#### Parameters

- **memory_ordering_semantics** (MemoryOrderingSemantics) - Memory ordering for the operation.
- **memory_scope** (MemoryScope) - Memory scope for the operation.
- **ptr** (ptr) - Pointer to the memory location.
- **expected** (tile) - The expected value at the memory location.
- **desired** (tile) - The value to write if comparison succeeds.
- **mask** (tile<i1>) - Optional mask for selective operation.
- **token** (token) - Optional token for operation ordering.

#### Results

- **result** (tile) - The value found at the memory location before the operation.
- **result_token** (token) - Result token for synchronization.

#### Description

The `atomic_cas_tko` operation atomically compares the value at `ptr` with `expected`, and if they match, writes `desired` to that location.

#### Constraints

- `ptr` must be a pointer type.
- `expected`, `desired`, and `result` must have the same type.

### 8.10.2. cuda_tile.atomic_rmw_tko

*Atomic read-modify-write with scope*

```
cuda_tile.atomic_rmw_tko %memory_ordering_semantics %memory_scope %atomic_op %ptr %value %mask %token
```

#### Parameters

- **memory_ordering_semantics** (MemoryOrderingSemantics) - Memory ordering for the operation.
- **memory_scope** (MemoryScope) - Memory scope for the operation.
- **atomic_op** (AtomicRMWKind) - The atomic operation to perform.
- **ptr** (ptr) - Pointer to the memory location.
- **value** (tile) - The value for the atomic operation.
- **mask** (tile<i1>) - Optional mask for selective operation.
- **token** (token) - Optional token for operation ordering.

#### Results

- **result** (tile) - The original value at the memory location.
- **result_token** (token) - Result token for synchronization.

#### Description

The `atomic_rmw_tko` operation performs an atomic read-modify-write operation on the memory location pointed to by `ptr`.

The `atomic_op` can be one of:
- `add` - Atomic addition
- `min` - Atomic minimum
- `max` - Atomic maximum
- `and` - Atomic bitwise AND
- `or` - Atomic bitwise OR
- `xor` - Atomic bitwise XOR
- `exchange` - Atomic exchange

#### Constraints

- `ptr` must be a pointer type.
- `value` and `result` must have the same type.

### 8.10.3. cuda_tile.atomic_load_tko

*Atomic load with ordering*

```
cuda_tile.atomic_load_tko %memory_ordering_semantics %memory_scope %ptr %mask %token
```

#### Parameters

- **memory_ordering_semantics** (MemoryOrderingSemantics) - Memory ordering for the load.
- **memory_scope** (MemoryScope) - Memory scope for the operation.
- **ptr** (ptr) - Pointer to the memory location.
- **mask** (tile<i1>) - Optional mask for selective loading.
- **token** (token) - Optional token for operation ordering.

#### Results

- **result** (tile) - The loaded value.
- **result_token** (token) - Result token for synchronization.

#### Description

The `atomic_load_tko` operation performs an atomic load from memory.

#### Constraints

- `ptr` must be a pointer type.

### 8.10.4. cuda_tile.atomic_store_tko

*Atomic store with ordering*

```
cuda_tile.atomic_store_tko %memory_ordering_semantics %memory_scope %ptr %value %mask %token
```

#### Parameters

- **memory_ordering_semantics** (MemoryOrderingSemantics) - Memory ordering for the store.
- **memory_scope** (MemoryScope) - Memory scope for the operation.
- **ptr** (ptr) - Pointer to the memory location.
- **value** (tile) - The value to store.
- **mask** (tile<i1>) - Optional mask for selective storing.
- **token** (token) - Optional token for operation ordering.

#### Results

- **result_token** (token) - Result token for synchronization.

#### Description

The `atomic_store_tko` operation performs an atomic store to memory.

#### Constraints

- `ptr` must be a pointer type.

## 8.11. Views

Views are a structured way to interact with tensors in memory. They are described in both the types section Tensor View and the semantics section Views. Views are the primary way to interact with global memory in Tile IR. A common pattern is to construct a Tensor View from a pointer with `cuda_tile.make_tensor_view` and then use the `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko` operations to read and write to them. For larger tensors, loading the entire tensor is not efficient and therefore we have a sub-view Partition View which allows a user to tile a tensor_view.

## 8.11.1. cuda_tile.get_index_space_shape

Return index space dimension size

```
cuda_tile.get_index_space_shape %src
```

### Parameters

- **src** (`view_type`) - The source view type.

### Results

- **result** (`Variadic<tile<any>>`) - The shape of the index space, each value representing the size of the corresponding dimension.

### Description

The `get_index_space_shape` operation returns the shape of the index space of `src`. The result types must be the same as the view’s index type, and the number of results must be the same as the view’s index rank. If the index space shape sizes do not fit within the provided type, behavior is undefined.

### Constraints

The operation is pure and does not perform any memory side effects.

## 8.11.2. cuda_tile.get_tensor_shape

Returns the shape of a tensor view

```
cuda_tile.get_tensor_shape %src
```

### Parameters

- **src** (`tensor_view`) - The source tensor view.

### Results

- **result** (`Variadic<tile<any>>`) - The shape of the tensor, each value representing the size of the corresponding dimension.

### Description

The `get_tensor_shape` operation returns the shape of the tensor backing the provided `tensor_view`. If the tensor shape sizes do not fit within the provided type, behavior is undefined.

### Constraints

The operation is pure and does not perform any memory side effects.

## 8.11.3. cuda_tile.load_view_tko

Load a tile from a tile view

```
cuda_tile.load_view_tko %memory_ordering_semantics %memory_scope %view %index %token %optimization_hints
```

### Parameters

- **memory_ordering_semantics** (`MemoryOrderingSemantics`) - The memory ordering semantics for the load operation.
- **memory_scope** (`MemoryScope`) - The memory scope for the atomic operation.
- **view** (`view_type`) - The view from which the tile will be loaded.
- **index** (`Variadic<tile<any>>`) - The n-dimensional index of the desired element to load from the view.
- **token** (`token`) - The optional token for the load operation.
- **optimization_hints** (`OptimizationHints`) - Optimization hints for operation.

### Results

- **tile** (`tile`) - The loaded tile.
- **result_token** (`token`) - The result token.

### Description

The `load_view_tko` operation loads a tile from a tile view. A view is mapping from view-space indices to a particular element in the view, each view type has a defined mapping from view-space indices to tiles produced from elements of the view. For example, the Partition View partitions a Tensor View into a grid of equally sized tiles. The view indices one of the partitioned tiles in the grid.

For a given view the rank of the indices must match the rank of the view’s index space. The space of valid indices depends on which view is passed to the operation. For example the index space of a Partition View is equal to the rank of the partitioned tiles. Out of bounds accesses are handling according to the semantics of the tile view.

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required. For example, `weak` ordering allows the compiler to assume that there are no concurrent accesses to any accessed location. For more information, refer to the memory model section of the specification.

- **weak** - No concurrent accesses to the source/destination location.
- **relaxed** - There may be concurrent access to the location, but this access does not establish a happens-before relationship.
- **acquire** - There may be concurrent accesses to the location. If this acquire observes a release operation, then happens before is established.
  - *Note: The following variants are not supported by this operation: release, acq_rel.*

The `memory_scope` attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

- **tl_blk** - There may be concurrent accesses from within the same tile block.
- **device** - There may be concurrent accesses from within the same device (i.e., GPU).
- **sys** - There may be concurrent accesses from anywhere within the system (i.e., all devices).

The `optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries. The hints are specified for each architecture (e.g., `sm_100`, `sm_120`) and for each architecture the user can specify specific hints for each operation.

- **num_cta_in_cga** - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for `cuda_tile.entry`.
- **allow_tma** - suggest whether to use TMA for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.
- **latency** - latency hint for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.

For example they can be annotated as:

```
optimization_hints =< sm_100 = { num_cta_in_cga = 8 }, sm_120 = { num_cta_in_cga = 16 } >
```

### Constraints

The operation must encode variadic operand segment sizes in attributes.

### Examples

```
%tensor_view = make_tensor_view %ptr, shape = [8192, 128], strides = [128, 1] : tensor_view<8192x128xf32, strides = [128, 1]>

// This example uses the PartitionView on a 8192x128xf32 tensor_view,
// dividing the tensor_view in tiles of 64x64.
%view = make_partition_view %tensor_view : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>>

%c0 = constant dense<0> : tile<i32>
%c1 = constant dense<1> : tile<i32>

// Load a tile at index (0, 0) in the view's index space.
// For this PartitionView, this is the rectangular tile such that
// X=[0,64) and Y=[0,64), in the coordinates of tiles.
%tile0, %res_token0 = load_view_tko weak %view [%c0, %c0] : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> tile<64x64xf32>, token

// Load a tile at index (0, 1) in the view's index space.
// For this PartitionView, this is the rectangular tile such that
// X=[0,64) and Y=[64,128), in the coordinates of tiles.
%tile1, %res_token1 = load_view_tko weak %view [%c0, %c1] : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> tile<64x64xf32>, token

// Same example as above but with memory token as input.
%token = make_token : token
%tile2, %res_token2 = load_view_tko weak %view [%c0, %c1] token = %token : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> tile<64x64xf32>, token

// Loads a tile at the dynamic index (%index, %index) in the view's index space.
%tile3, %res_token3 = load_view_tko weak %view [%index, %index] : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> tile<64x64xf32>, token
```

See `cuda_tile.load_view_tko_0` for the full example listing.

## 8.11.4. cuda_tile.make_partition_view

Create a partition view from a tensor view

```
cuda_tile.make_partition_view %tensor_view
```

### Parameters

- **tensor_view** (`tensor_view`) - The source tensor view to create a partition view from.

### Results

- **result** (`partition_view`) - The created partition view.

### Description

The `make_partition_view` operation creates a `partition_view` from a `tensor_view`. For more details about partition views see Partition View. The operation uses the type constraints of the input tensor view and the annotated return type to perform the partitioning.

The tensor view’s type contains its physical layout in the form of shapes and strides and the partition view containts the logical size of a single tile. The resulting partition view can be loaded from using `cuda_tile.load_view_tko` and stored to using `cuda_tile.store_view_tko`. The view memory options act on the computed index space of the partition view see Tensor View and Partition View for detailed semantics.

### Constraints

The operation is conditionally speculatable based on the specific operands and attributes. The operation may be speculatively executed without side effects. The operation is pure and does not perform any memory side effects.

### Examples

```
%tensor_view0 = make_tensor_view %ptr, shape = [8192, 8192, 64], strides = [524288, 64, 1] : tensor_view<8192x8192x64xf32, strides = [524288, 64, 1]>

// Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
// the provided tensor_view.
make_partition_view %tensor_view0 : partition_view<tile = (1024x1x32), tensor_view<8192x8192x64xf32, strides = [524288, 64, 1]>>

%s0 = constant dense<8192> : tile<i32>
%str0 = constant dense<524288> : tile<i32>
%tensor_view1 = make_tensor_view %ptr, shape = [%s0, 8192, 64], strides = [%str0, 64, 1] : tensor_view<?x8192x64xf32, strides = [?, 64, 1]>

// Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
// the provided tensor_view, with masking. The provided tensor_view has a
// dynamically-sized dimension.
make_partition_view %tensor_view1 : partition_view<tensor_view<?x8192x64xf32, strides = [?, 64, 1]>>
```

See `cuda_tile.make_partition_view_0` for the full example listing.

## 8.11.5. cuda_tile.make_tensor_view

Create :code:`tensor_view` from a pointer to global memory

```
cuda_tile.make_tensor_view %base %dynamicShape %dynamicStrides
```

### Parameters

- **base** (`tile<ptr>`) - The scalar base pointer to a portion of global memory.
- **dynamicShape** (`Variadic<tile<any>>`) - The array of values representing the shape of the view, may be fully dynamic.
- **dynamicStrides** (`Variadic<tile<any>>`) - The array of values representing the strides of the view, may be fully dynamic.

### Results

- **result** (`tensor_view`) - The constructed tensor_view.

### Description

The `make_tensor_view` operation constructs a `tensor_view` from a global memory pointer, a dynamic shape and dynamic strides. See Tensor View for more details. The constructor supports taking dynamic arrays for shapes and strides as part of the constructor enabling workloads to take global memory tensors of dynamic shape and strides.

If these arguments are static they will be statically reflected in the type of the resulting `tensor_view`, if they are dynamic they will appear as `?` in the type. See below for concrete examples.

If shapes or strides are larger than the indexBitwidth of the `tensor_view`, behavior is undefined on the creation of the `tensor_view`.

### Constraints

The operation must encode variadic operand segment sizes in attributes. The operation is pure and does not perform any memory side effects.

### Examples

```
// tensor_view to a scalar tile of f32
%a0 = make_tensor_view %base, shape = [], strides = [] : tensor_view<f32>

// tensor_view to a tile of static shape and strides
%a1 = make_tensor_view %base, shape = [32, 32], strides = [32, 1] : tensor_view<32x32xf32, strides = [32, 1]>

%sh0 = constant dense<32> : tile<i32>
%sh1 = constant dense<32> : tile<i32>
%st0 = constant dense<32> : tile<i32>
%st1 = constant dense<1> : tile<i32>

// tensor_view to a tile with partially dynamic shape and strides
// all dynamic values must be of the same type, here tile<i32>
%a2 = make_tensor_view %base, shape = [%sh0, %sh1], strides = [%st0, %st1] : tile<i32> -> tensor_view<?x?xf32, strides = [?, ?]>
```

See `cuda_tile.make_tensor_view_0` for the full example listing.

## 8.11.6. cuda_tile.store_view_tko

Stores a tile into a tile view

```
cuda_tile.store_view_tko %memory_ordering_semantics %memory_scope %tile %view %index %token %optimization_hints
```

### Parameters

- **memory_ordering_semantics** (`MemoryOrderingSemantics`) - The memory scope for the store operation.
- **memory_scope** (`MemoryScope`) - The memory scope for the store operation.
- **tile** (`tile`) - The tile to store.
- **view** (`view_type`) - The view to store the tile to.
- **index** (`Variadic<tile<any>>`) - The indices of the desired target tile within the view.
- **token** (`token`) - The optional token for operation ordering.
- **optimization_hints** (`OptimizationHints`) - Optimization hints for operation.

### Results

- **result_token** (`token`) - The result token for synchronization.

### Description

The `store_view_tko` operation stores a tile to a view indexing into a tile view. A view is mapping from view-space indices to a particular element in the view, each view type has a defined mapping from view-space indices to tiles produced from elements of the view. For example, the Partition View partitions a Tensor View into a grid of equally sized tiles. The view indices one of the partitioned tiles in the grid.

For a given view the rank of the indices must match the rank of the view’s index space. The space of valid indices depends on which view is passed to the operation. For example the index space of a Partition View is equal to the rank of the partitioned tiles. The index space of the view is computed a function of the requested tile size and the shape of the view.

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required. For example, `weak` ordering allows the compiler to assume that there are no concurrent accesses to any accessed location. For more information, refer to the memory model section of the specification.

- **weak** - No concurrent accesses to the source/destination location.
- **relaxed** - There may be concurrent access to the location, but this access does not establish a happens-before relationship.
- **release** - There may be concurrent access to the location. If this release is observed with an acquire operation, then happens before is established.
  - *Note: The following variants are not supported by this operation: acquire, acq_rel.*

The `memory_scope` attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

- **tl_blk** - There may be concurrent accesses from within the same tile block.
- **device** - There may be concurrent accesses from within the same device (i.e., GPU).
- **sys** - There may be concurrent accesses from anywhere within the system (i.e., all devices).

The `optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries. The hints are specified for each architecture (e.g., `sm_100`, `sm_120`) and for each architecture the user can specify specific hints for each operation.

- **num_cta_in_cga** - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for `cuda_tile.entry`.
- **allow_tma** - suggest whether to use TMA for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.
- **latency** - latency hint for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.

For example they can be annotated as:

```
optimization_hints =< sm_100 = { num_cta_in_cga = 8 }, sm_120 = { num_cta_in_cga = 16 } >
```

### Constraints

The operation must encode variadic operand segment sizes in attributes. The operation’s result type may be inferred from its operands and attributes.

### Examples

```
%tensor_view = make_tensor_view %ptr, shape = [8192, 128], strides = [128, 1] : tensor_view<8192x128xf32, strides = [128, 1]>

// This example uses the PartitionView on a 8192x128xf32 tensor_view,
// dividing the tensor_view in tiles of 64x64.
%view = make_partition_view %tensor_view : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>>

%c0 = constant dense<0> : tile<i32>
%c1 = constant dense<1> : tile<i32>
%tile = constant dense<0.0> : tile<64x64xf32>

// Store a tile at index (0, 0) in the view's index space.
// For this TilePartitionView, this is the rectangular tile such that
// X=[0,64) and Y=[0,64), in the coordinates of tiles.
%res_token0 = store_view_tko weak %tile, %view [%c0, %c0] : tile<64x64xf32>, partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> token

// Store a tile at index (0, 1) in the view's index space.
// For this PartitionView, this is the rectangular tile such that
// X=[0,64) and Y=[64,128), in the coordinates of tiles.
%res_token1 = store_view_tko weak %tile, %view [%c0, %c1] : tile<64x64xf32>, partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> token

// Same example as above but with input token.
%token = make_token : token
%res_token2 = store_view_tko weak %tile, %view [%c0, %c1] token = %token : tile<64x64xf32>, partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> token
```

See `cuda_tile.store_view_tko_0` for the full example listing.
## 8.12. Miscellaneous

**Tile IR** contains various utility operations that don't fit into other categories.

### 8.12.1. cuda_tile.unrealized_conversion_cast

*Unrealized conversion operation*

```
cuda_tile.unrealized_conversion_cast %inputs
```

#### Parameters

- **inputs** (Variadic<Any>) - The inputs to convert.

#### Results

- **outputs** (Variadic<Any>) - The converted outputs.

#### Description

The `unrealized_conversion_cast` operation is a placeholder for conversions between types that aren't directly supported. This operation is used internally by the compiler and should not be used directly in user code.

#### Constraints

- The operation is conditionally speculatable based on the specific operands and attributes.
- The operation is pure and does not perform any memory side effects.

### 8.12.2. cuda_tile.print

*Print operation for debugging*

```
cuda_tile.print %format %values
```

#### Parameters

- **format** (String) - The format string for printing.
- **values** (Variadic<Any>) - The values to print.

#### Results

No results.

#### Description

The `print` operation is a debug utility that prints formatted output. It should not be used in production code. 

Format specifiers:
- `%d` - Integer
- `%f` - Floating-point
- `%s` - String
- `%%` - Literal percent sign

#### Examples

```mlir
print "Value: %d\n", %val : tile<i32>
print "Float: %f\n", %fval : tile<f32>
```

### 8.12.3. cuda_tile.assume

*Assume operation for optimization hints*

```
cuda_tile.assume %condition
```

#### Parameters

- **condition** (tile<i1>) - The condition to assume true.

#### Results

No results.

#### Description

The `assume` operation provides a hint to the compiler that the given condition is always true. This can enable optimizations but if the assumption is violated at runtime, the behavior is undefined.

#### Constraints

- The operation is conditionally speculatable based on the specific operands and attributes.

### 8.12.4. cuda_tile.barrier

*Synchronization barrier*

```
cuda_tile.barrier %scope %token
```

#### Parameters

- **scope** (BarrierScope) - The scope of the barrier.
- **token** (token) - Optional token for operation ordering.

#### Results

- **result_token** (token) - Result token for synchronization.

#### Description

The `barrier` operation provides a synchronization point for threads within the specified scope.

Barrier scopes:
- `warp` - Synchronize threads within a warp
- `block` - Synchronize threads within a block
- `cluster` - Synchronize threads within a cluster

#### Constraints

- All threads in the scope must execute the barrier or the behavior is undefined.