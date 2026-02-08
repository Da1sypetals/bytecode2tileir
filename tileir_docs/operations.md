# 8. Operations

This section describes a complete and categorized list of all **Tile IR** instructions names, signatures, and semantics.

## 8.1. Meta Types

Operations have arguments which are **Tile IR** values with **Tile IR** types but many operations have immediate or static arguments which correspond to attributes in the MLIR dialect. These **meta types** are not representable in the **Tile IR** type system but are used to construct **Tile IR** programs and only present at compile time. Operations in the specification are described abstractly in both the **Tile IR** IR and bytecode independent of the MLIR or bytecode encoding. For each of these types we provide a definition of them below and link to them from each operation.

> **Note**
> 
> The convention is that the meta types are capitalized and **Tile IR** types are snake cased.
> 
> The convention is that the meta types are capitalized and the native **Tile IR** types are camel cased are snake cased.

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

`view_type` represents a type which implements the view interface, currently this is only implemented by partition_view but will have new implementers in future releases.

## 8.2. Operation Design Considerations

The design of **Tile IR** has a set of design considerations that apply to all operations in the dialect this section introduces some of the common design considerations that apply to all operations, or to classes of operations generically.

### 8.2.1. Explicit Broadcast

There are no implicit broadcast performed by operations in the **Tile IR** dialect all operations that require operands of the same shape must be explicitly broadcasted. For example to use the `cuda_tile.offset` operation to add an offset tile to a pointer, the pointer and offset must be reshaped or broadcasted to have the same shape using the `cuda_tile.reshape` or `cuda_tile.broadcast` operations.

### 8.2.2. Distinct Floating-Point and Integer Operations

Numeric operations are split across integer and floating-point types due to differences in flags such as rounding modes, `NaN` handling, and fast math.

For example, the `cuda_tile.addf` operation supports a rounding attribute, but the addi operation does not.

### 8.2.3. Explicit Overflow Annotations

Some operations such as `cuda_tile.addi` support an explicit overflow annotation that expresses the expected overflow behavior of the operation.

These attributes serve as assumptions that an implementation may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

We recommend that generators of **Tile IR** programs utilize these annotations to help the implementation reason about the overflow behavior of the operation, enabling extra optimization opportunities.

## 8.3. Core

### 8.3.1. cuda_tile.broadcast

*Broadcast tile to new shape*

```
cuda_tile.broadcast %source
```

#### Parameters

* **source** (tile) - The tile to broadcast. 13.1

#### Results

* **result** (tile) - The broadcasted tile. 13.1

#### Description

The `broadcast` operation expands each unary (`1`) dimension in the input tile by duplicating the data along that dimension.

Expansion happens only for dimensions of size one that are stretched or "copied" to match the size of the dimension implied by the result type of the operation. The operation does not change the rank of the source tile. Any change to the rank of the source tile must be made using reshape-like operations before broadcasting.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.
* `source` and `result` must have the same element type (tile).
* `source` and `result` must have the same rank.

### 8.3.2. cuda_tile.cat

*Concatenate tiles along specified dimension*

```
cuda_tile.cat %lhs %rhs %dim
```

#### Parameters

* **lhs** (tile) - The left hand side operand. 13.1
* **rhs** (tile) - The right hand side operand. 13.1
* **dim** (i64) - The dimension along which to concatenate. 13.1

#### Results

* **result** (tile) - The concatenated result tile. 13.1

#### Description

The `cat` operation concatenates the two input tiles. The input tiles must have the same shape in all but the concatenating dimension. Concatenation happens along the dimension specified by the the attribute `dim` the resulting dimension is the sum of the the two input tiles concatenating dimension.

\[\begin{split}\text{cat}(x, y, dim_{cat})[ \vec{i} ] = \begin{cases} x[..., i_{cat}, ..., i_n] & \text{if } i_{cat} < d_{cat} \\ y[..., i_{cat} - d_{cat}, ..., i_n] & \text{if } i_{cat} \geq d_{cat} \end{cases}\end{split}\]

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.
* `lhs`, `rhs` and `result` must have the same rank.
* `lhs`, `rhs` and `result` must have the same element type (tile).

#### Examples

```
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

### 8.3.3. cuda_tile.constant

*Construct a constant tile*

```
cuda_tile.constant %value
```

#### Parameters

* **value** (DenseConstant) - The constant value to create. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The constant tile. 13.1

#### Description

The `constant` operation creates a tile initialized by `$value`.

There are two main forms of using the operation:

* One where the value is a single constant specified by `dense<c>` and the tile is filled with identical values for all elements.
* One where the value is a list of constants specified by `dense<[c0, c1, c2, ...]>` and the constant value's shape must match the tile's shape.

The annotated type of the tile constrains its rank, shape, and element type.

#### Constraints

* The operation has no operands and may be constant folded.
* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.
* `value` and `result` must have the same shape and element type (DenseConstant).
* The operation's result type may be inferred from its operands and attributes.

## 8.4. Conversions

There are no implicit type conversions in **Tile IR** thus we expose a set of explicit conversion operations for interconverting between types which have compatible representations or rules for conversion.

`cuda_tile.bitcast` preserves the contents of the input but allows for changing of element types, `cuda_tile.exti` and `cuda_tile.trunci` change the width of integer tiles, `cuda_tile.ftoi` and `cuda_tile.itof` convert floating-point tiles to integer tiles and vice versa, and `cuda_tile.ftof` converts between different floating-point types.

For more details on conversions and their rules see the individual operation's documentation.

### 8.4.1. cuda_tile.bitcast

*Bitcast a tile from one element type to another*

```
cuda_tile.bitcast %source
```

#### Parameters

* **source** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The source tile to cast. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The casted tile. 13.1

#### Description

The `bitcast` operation casts the input tile from one element type to another without modifying the underlying bits.

Only non-pointer types of the same bit width are allowed (e.g., `i32` to `f32`). Pointer types must use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr` instead.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.2. cuda_tile.exti

*Extend the width of an integer tile*

```
cuda_tile.exti %from %signedness
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to extend. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The extended integer tile. 13.1

#### Description

The `exti` operation converts a tile of integers of a given width to a strictly larger width. Zero-extension is used for `unsigned` integers and sign-extension is used for `signed` integers.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.3. cuda_tile.ftof

*Convert between floating-point types*

```
cuda_tile.ftof %from %rounding_mode
```

#### Parameters

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The result floating-point tile. 13.1

#### Description

The `ftof` operation converts a tile of a given floating-point element type into one of a different floating-point element type (for example, from `f32` to `f64`).

The source type and the result type must be different.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.4. cuda_tile.ftoi

*Convert a tile from floating-point values to integer values*

```
cuda_tile.ftoi %from %signedness %rounding_mode
```

#### Parameters

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The result integer tile. 13.1

#### Description

The `ftoi` operation converts a floating-point tile into an integer tile.

In contrast to a `bitcast` which is bits preserving, this preserves the numerical value of the tile, rounded towards zero to the nearest integer of the provided type.

> **Warning**
> 
> If the input floating-point value, after being rounded, is outside the (signed or unsigned) range of the target integer type, the closest representable value is used instead. `NaN` values are converted to 0. Input `Inf` values are undefined behavior.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.5. cuda_tile.itof

*Convert integer to floating-point*

```
cuda_tile.itof %from %signedness %rounding_mode
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The converted floating-point tile. 13.1

#### Description

The `itof` operation converts an integer tile into a float tile. In contrast to a bitcast, this preserves the numerical value of the tile, rounded to the nearest floating-point number of the provided type.

> **Warning**
> 
> If the input integer value, after being rounded, is outside the range of the target floating-point type, it is converted to `Inf` for types that support that value, and `NaN` otherwise.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.6. cuda_tile.int_to_ptr

*Convert a tile of integers to a tile of pointers*

```
cuda_tile.int_to_ptr %source
```

#### Parameters

* **source** (tile<i64>) - The input tile of integers. 13.1

#### Results

* **result** (ptr) - The output tile of pointers. 13.1

#### Description

The `int_to_ptr` operation converts a tile of integers to a tile of pointers.

The inverse of this operation is `cuda_tile.ptr_to_int`.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.7. cuda_tile.ptr_to_int

*Convert a tile of pointers to a tile of integers*

```
cuda_tile.ptr_to_int %source
```

#### Parameters

* **source** (ptr) - The input tile of pointers. 13.1

#### Results

* **result** (tile<i64>) - The output tile of integers. 13.1

#### Description

The `ptr_to_int` operation converts a tile of pointer-type elements to a tile of `i64` elements.

The inverse of this operation is `cuda_tile.int_to_ptr`.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.8. cuda_tile.ptr_to_ptr

*Reinterpret a tile of one pointer type as another*

```
cuda_tile.ptr_to_ptr %source
```

#### Parameters

* **source** (ptr) - Tile with source pointer element type. 13.1

#### Results

* **result** (ptr) - Tile with target pointer element type. 13.1

#### Description

The `ptr_to_ptr` operation casts a tile of pointers from a pointer of one element type to another element. Casts between pointer and non-pointer types are disallowed.

In order to perform those conversions, use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr`. These operations are distinct to enable future compiler reasoning about pointer provenance.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.9. cuda_tile.trunci

*Truncates the width of an integer tile*

```
cuda_tile.trunci %from %overflow
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to truncate. 13.1
* **overflow** (IntegerOverflow) - The overflow behavior of the operation. 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The truncated integer tile. 13.1

#### Description

The `trunci` operation converts a tile of integers of a given element type to one with a strictly smaller width.

The optional overflow attribute specifies whether an overflow can occur when interpreting the operand as a signed and/or unsigned integer. In case of "no signed wrap", all truncated bits must have the same value as the most significant bit of the truncated result. In case of "no unsigned wrap", the truncated bits must be zero.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

* `none` - The compiler makes no assumptions regarding overflow behavior.
* `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
* `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
* `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

#### Examples

```
%c0 = constant dense<0> : tile<i32>
%c1 = constant dense<1> : tile<i64>
%c2 = constant dense<[0, 1, 2, 3]> : tile<4xi32>
%c3 = constant dense<0.0> : tile<2x4xf32>
%c4 = constant dense<[0.0, 1.0, 2.0, 3.0]> : tile<4xf64>
```

### 8.3.4. cuda_tile.entry

*Define a tile kernel*

```
cuda_tile.entry %sym_name %function_type %arg_attrs %res_attrs %optimization_hints
```

#### Parameters

* **sym_name** (Symbol) - The name of the function. 13.1
* **function_type** (Type) - The type of the function. 13.1
* **arg_attrs** (Attributes) - The argument attributes of the function: none of these are supported by CUDA Tile IR at the moment. 13.1
* **res_attrs** (Attributes) - The result attributes of the function: none of these are supported by CUDA Tile IR at the moment. 13.1
* **optimization_hints** (OptimizationHints) - Compiler architecture-specific optimization hints 13.1

#### Results

No results.

#### Description

The `entry` operation defines a tile kernel; a kernel is a function that can serve as the program entry point. It has a unique name per-module. A kernel can not return any value. It must be launched from the host side using `cuLaunchKernel` or similar CUDA runtime API functions.

Tile kernels require that the user specifies the 3-d grid dimensions at launch which defines the number of tile blocks (or kernel instances) that will execute the kernel in parallel.

For detailed semantics of tile kernels see Tile Kernel.

The `optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries.

The hints are specified for each architecture (e.g., `sm_100`, `sm_120`) and for each architecture the user can specify specific hints for each operation.

* `num_cta_in_cga` - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for `cuda_tile.entry`.
* `allow_tma` - suggest whether to use TMA for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.
* `latency` - latency hint for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.

For example they can be annotated as:

```
optimization_hints=<
  sm_100 = {num_cta_in_cga = 8},
  sm_120 = {num_cta_in_cga = 16}
>
```

#### Constraints

* The operation must be a symbol in the global symbol table.
* The operation must implement callable target interface.
* The operation must implement function-like behavior interface.
* The region must not capture SSA values defined above the operation.
* The operation must provide custom parsing and printing methods.
* Each provided region must contain exactly one block.

### 8.3.5. cuda_tile.extract

*Extract a subtile from a tile*

```
cuda_tile.extract %source %indices
```

#### Parameters

* **source** (tile) - The source tile to extract from. 13.1
* **indices** (Variadic<tile<i32>>) - The indices of the slice to extract. 13.1

#### Results

* **result** (tile) - The extracted subtile. 13.1

#### Description

The extract operation extracts a subtile from the given source tile.

The shape of the result tile must divide the shape of the source tile evenly e.g., `tile<4xf32>` is a valid extraction from `tile<8xf32>`, but `tile<3xf32>` is not.

The `$indices` indicate the number of the slice to extract, but importantly not the offsets used to construct the subtile for extraction. The semantics of extract means that only full size slices can be extracted.

Slices of a source tile with the same shape are non-overlapping by definition for unique indices.

> **Warning**
> 
> If the indices specify a non-existent (i.e., out-of-bounds) slice, the behavior of the operation is undefined.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.
* `source` and `result` must have the same rank.

#### Examples

```
// Extract a subtile from %t at dim_0 = [4;8) and dim_1 = [4;6).
%c1 = constant dense<1> : tile<i32>
%c2 = constant dense<2> : tile<i32>
%t = constant dense<0.0> : tile<32x8xf32>
// Valid indices are: [ {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3} ]
%0 = extract %t[%c1, %c2]
    : tile<32x8xf32> -> tile<4x2xf32>
```

### 8.3.6. cuda_tile.get_global

*Get a pointer to a global variable*

```
cuda_tile.get_global %name
```

#### Parameters

* **name** (Symbol) - The name of the global variable. 13.1

#### Results

* **result** (tile<ptr>) - The result of the get_global operation. 13.1

#### Description

The `get_global` operation returns a pointer to the specified `global` variable. A global variable is a form of static global memory allocation that can be declared using the `cuda_tile.global` operation.

The element type of the returned pointer will be of the same type as the element type of the declared global variable.

For detailed semantics of global variables see Global Variable.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

#### Examples

```
global @val dense<[0.1, 0.2, 0.3, 0.4]> : tile<4xf32>

entry @example() {
  %ptr = get_global @val : tile<ptr<f32>>
  return
}
```

### 8.3.7. cuda_tile.get_num_tile_blocks

*Get total number of tile blocks*

```
cuda_tile.get_num_tile_blocks
```

#### Parameters

No parameters.

#### Results

* **gridSize_x** (tile<i32>) - The number of tile blocks in dimension `x`. 13.1
* **gridSize_y** (tile<i32>) - The number of tile blocks in dimension `y`. 13.1
* **gridSize_z** (tile<i32>) - The number of tile blocks in dimension `z`. 13.1

#### Description

The `get_num_tile_blocks` operation queries the total number of tile blocks in the form of a 3-tuple specifying the extent of each grid dimension.

A tile `id` is a coordinate in 3-space and therefore the must also be a 3-tuple containing the extent of each dimension: `x`, `y` and `z`.

When launching 1- or 2-dimensional grids, the unspecified dimensions will have a cardinality of 1.

For example if the grid used to launch the kernel is `(1024, 1024)` then the result of this operation will be `(1024, 1024, 1)`.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.
* The operation's result type may be inferred from its operands and attributes.

## 8.4. Conversions

There are no implicit type conversions in **Tile IR** thus we expose a set of explicit conversion operations for interconverting between types which have compatible representations or rules for conversion.

`cuda_tile.bitcast` preserves the contents of the input but allows for changing of element types, `cuda_tile.exti` and `cuda_tile.trunci` change the width of integer tiles, `cuda_tile.ftoi` and `cuda_tile.itof` convert floating-point tiles to integer tiles and vice versa, and `cuda_tile.ftof` converts between different floating-point types.

For more details on conversions and their rules see the individual operation's documentation.

### 8.4.1. cuda_tile.bitcast

*Bitcast a tile from one element type to another*

```
cuda_tile.bitcast %source
```

#### Parameters

* **source** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The source tile to cast. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The casted tile. 13.1

#### Description

The `bitcast` operation casts the input tile from one element type to another without modifying the underlying bits.

Only non-pointer types of the same bit width are allowed (e.g., `i32` to `f32`). Pointer types must use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr` instead.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.2. cuda_tile.exti

*Extend the width of an integer tile*

```
cuda_tile.exti %from %signedness
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to extend. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The extended integer tile. 13.1

#### Description

The `exti` operation converts a tile of integers of a given width to a strictly larger width. Zero-extension is used for `unsigned` integers and sign-extension is used for `signed` integers.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.3. cuda_tile.ftof

*Convert between floating-point types*

```
cuda_tile.ftof %from %rounding_mode
```

#### Parameters

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The result floating-point tile. 13.1

#### Description

The `ftof` operation converts a tile of a given floating-point element type into one of a different floating-point element type (for example, from `f32` to `f64`).

The source type and the result type must be different.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.4. cuda_tile.ftoi

*Convert a tile from floating-point values to integer values*

```
cuda_tile.ftoi %from %signedness %rounding_mode
```

#### Parameters

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The result integer tile. 13.1

#### Description

The `ftoi` operation converts a floating-point tile into an integer tile.

In contrast to a `bitcast` which is bits preserving, this preserves the numerical value of the tile, rounded towards zero to the nearest integer of the provided type.

> **Warning**
> 
> If the input floating-point value, after being rounded, is outside the (signed or unsigned) range of the target integer type, the closest representable value is used instead. `NaN` values are converted to 0. Input `Inf` values are undefined behavior.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.5. cuda_tile.itof

*Convert integer to floating-point*

```
cuda_tile.itof %from %signedness %rounding_mode
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The converted floating-point tile. 13.1

#### Description

The `itof` operation converts an integer tile into a float tile. In contrast to a bitcast, this preserves the numerical value of the tile, rounded to the nearest floating-point number of the provided type.

> **Warning**
> 
> If the input integer value, after being rounded, is outside the range of the target floating-point type, it is converted to `Inf` for types that support that value, and `NaN` otherwise.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.6. cuda_tile.int_to_ptr

*Convert a tile of integers to a tile of pointers*

```
cuda_tile.int_to_ptr %source
```

#### Parameters

* **source** (tile<i64>) - The input tile of integers. 13.1

#### Results

* **result** (ptr) - The output tile of pointers. 13.1

#### Description

The `int_to_ptr` operation converts a tile of integers to a tile of pointers.

The inverse of this operation is `cuda_tile.ptr_to_int`.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.7. cuda_tile.ptr_to_int

*Convert a tile of pointers to a tile of integers*

```
cuda_tile.ptr_to_int %source
```

#### Parameters

* **source** (ptr) - The input tile of pointers. 13.1

#### Results

* **result** (tile<i64>) - The output tile of integers. 13.1

#### Description

The `ptr_to_int` operation converts a tile of pointer-type elements to a tile of `i64` elements.

The inverse of this operation is `cuda_tile.int_to_ptr`.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.8. cuda_tile.ptr_to_ptr

*Reinterpret a tile of one pointer type as another*

```
cuda_tile.ptr_to_ptr %source
```

#### Parameters

* **source** (ptr) - Tile with source pointer element type. 13.1

#### Results

* **result** (ptr) - Tile with target pointer element type. 13.1

#### Description

The `ptr_to_ptr` operation casts a tile of pointers from a pointer of one element type to another element. Casts between pointer and non-pointer types are disallowed.

In order to perform those conversions, use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr`. These operations are distinct to enable future compiler reasoning about pointer provenance.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.9. cuda_tile.trunci

*Truncates the width of an integer tile*

```
cuda_tile.trunci %from %overflow
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to truncate. 13.1
* **overflow** (IntegerOverflow) - The overflow behavior of the operation. 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The truncated integer tile. 13.1

#### Description

The `trunci` operation converts a tile of integers of a given element type to one with a strictly smaller width.

The optional overflow attribute specifies whether an overflow can occur when interpreting the operand as a signed and/or unsigned integer. In case of "no signed wrap", all truncated bits must have the same value as the most significant bit of the truncated result. In case of "no unsigned wrap", the truncated bits must be zero.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

* `none` - The compiler makes no assumptions regarding overflow behavior.
* `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
* `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
* `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

#### Examples

```
entry @example() {
  %x, %y, %z = get_num_tile_blocks : tile<3xi32>
  print "x: %, y: %, z: %\n", %x, %y, %z : tile<i32>, tile<i32>, tile<i32>
}
```

### 8.3.8. cuda_tile.get_tile_block_id

*Get the currently executing tile block coordinates*

```
cuda_tile.get_tile_block_id
```

#### Parameters

No parameters.

#### Results

* **blockId_x** (tile<i32>) - The tile block ID for dimension `x`. 13.1
* **blockId_y** (tile<i32>) - The tile block ID for dimension `y`. 13.1
* **blockId_z** (tile<i32>) - The tile block ID for dimension `z`. 13.1

#### Description

`get_tile_block_id` returns a 3-d tile block coordinates (or ID) of the currently executing tile block.

A tile ID has three dimensions: `x`, `y`, and `z`. This operation returns all three of them simultaneously. The value of each dimension returned by this operation is between `0` (including) and the value returned by `get_num_tile_blocks` for the respective axis (excluding), represented by the inclusive interval `[0, get_num_tile_blocks(dim) - 1]` . Grid dimensions unspecified at kernel launch (i.e., a 1-d or 2-d grid) will always be `0` for all tile blocks.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.
* The operation's result type may be inferred from its operands and attributes.

## 8.4. Conversions

There are no implicit type conversions in **Tile IR** thus we expose a set of explicit conversion operations for interconverting between types which have compatible representations or rules for conversion.

`cuda_tile.bitcast` preserves the contents of the input but allows for changing of element types, `cuda_tile.exti` and `cuda_tile.trunci` change the width of integer tiles, `cuda_tile.ftoi` and `cuda_tile.itof` convert floating-point tiles to integer tiles and vice versa, and `cuda_tile.ftof` converts between different floating-point types.

For more details on conversions and their rules see the individual operation's documentation.

### 8.4.1. cuda_tile.bitcast

*Bitcast a tile from one element type to another*

```
cuda_tile.bitcast %source
```

#### Parameters

* **source** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The source tile to cast. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The casted tile. 13.1

#### Description

The `bitcast` operation casts the input tile from one element type to another without modifying the underlying bits.

Only non-pointer types of the same bit width are allowed (e.g., `i32` to `f32`). Pointer types must use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr` instead.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.2. cuda_tile.exti

*Extend the width of an integer tile*

```
cuda_tile.exti %from %signedness
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to extend. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The extended integer tile. 13.1

#### Description

The `exti` operation converts a tile of integers of a given width to a strictly larger width. Zero-extension is used for `unsigned` integers and sign-extension is used for `signed` integers.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.3. cuda_tile.ftof

*Convert between floating-point types*

```
cuda_tile.ftof %from %rounding_mode
```

#### Parameters

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The result floating-point tile. 13.1

#### Description

The `ftof` operation converts a tile of a given floating-point element type into one of a different floating-point element type (for example, from `f32` to `f64`).

The source type and the result type must be different.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.4. cuda_tile.ftoi

*Convert a tile from floating-point values to integer values*

```
cuda_tile.ftoi %from %signedness %rounding_mode
```

#### Parameters

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The result integer tile. 13.1

#### Description

The `ftoi` operation converts a floating-point tile into an integer tile.

In contrast to a `bitcast` which is bits preserving, this preserves the numerical value of the tile, rounded towards zero to the nearest integer of the provided type.

> **Warning**
> 
> If the input floating-point value, after being rounded, is outside the (signed or unsigned) range of the target integer type, the closest representable value is used instead. `NaN` values are converted to 0. Input `Inf` values are undefined behavior.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.5. cuda_tile.itof

*Convert integer to floating-point*

```
cuda_tile.itof %from %signedness %rounding_mode
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The converted floating-point tile. 13.1

#### Description

The `itof` operation converts an integer tile into a float tile. In contrast to a bitcast, this preserves the numerical value of the tile, rounded to the nearest floating-point number of the provided type.

> **Warning**
> 
> If the input integer value, after being rounded, is outside the range of the target floating-point type, it is converted to `Inf` for types that support that value, and `NaN` otherwise.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.6. cuda_tile.int_to_ptr

*Convert a tile of integers to a tile of pointers*

```
cuda_tile.int_to_ptr %source
```

#### Parameters

* **source** (tile<i64>) - The input tile of integers. 13.1

#### Results

* **result** (ptr) - The output tile of pointers. 13.1

#### Description

The `int_to_ptr` operation converts a tile of integers to a tile of pointers.

The inverse of this operation is `cuda_tile.ptr_to_int`.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.7. cuda_tile.ptr_to_int

*Convert a tile of pointers to a tile of integers*

```
cuda_tile.ptr_to_int %source
```

#### Parameters

* **source** (ptr) - The input tile of pointers. 13.1

#### Results

* **result** (tile<i64>) - The output tile of integers. 13.1

#### Description

The `ptr_to_int` operation converts a tile of pointer-type elements to a tile of `i64` elements.

The inverse of this operation is `cuda_tile.int_to_ptr`.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.8. cuda_tile.ptr_to_ptr

*Reinterpret a tile of one pointer type as another*

```
cuda_tile.ptr_to_ptr %source
```

#### Parameters

* **source** (ptr) - Tile with source pointer element type. 13.1

#### Results

* **result** (ptr) - Tile with target pointer element type. 13.1

#### Description

The `ptr_to_ptr` operation casts a tile of pointers from a pointer of one element type to another element. Casts between pointer and non-pointer types are disallowed.

In order to perform those conversions, use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr`. These operations are distinct to enable future compiler reasoning about pointer provenance.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.9. cuda_tile.trunci

*Truncates the width of an integer tile*

```
cuda_tile.trunci %from %overflow
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to truncate. 13.1
* **overflow** (IntegerOverflow) - The overflow behavior of the operation. 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The truncated integer tile. 13.1

#### Description

The `trunci` operation converts a tile of integers of a given element type to one with a strictly smaller width.

The optional overflow attribute specifies whether an overflow can occur when interpreting the operand as a signed and/or unsigned integer. In case of "no signed wrap", all truncated bits must have the same value as the most significant bit of the truncated result. In case of "no unsigned wrap", the truncated bits must be zero.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

* `none` - The compiler makes no assumptions regarding overflow behavior.
* `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
* `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
* `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.3.9. cuda_tile.global

*Allocate static global memory*

```
cuda_tile.global %sym_name %value %alignment
```

#### Parameters

* **sym_name** (Symbol) - The name of the global variable. 13.1
* **value** (DenseConstant) - The value to initialize the allocation with. 13.1
* **alignment** (i64) - The alignment of the buffer. 13.1

#### Results

No results.

#### Description

The `global` operation statically allocates a mutable 1-dimensional location in global memory and initializes it using `value`. The initialization of the allocation is performed at CUDA module load time. The lifetime of the allocation is the same as the lifetime of the module.

The allocation may be read or written to by first using `cuda_tile.get_global` to obtain a pointer to the the memory and then read using `cuda_tile.load_ptr_tko` or written to using `cuda_tile.store_ptr_tko`.

The initial values are stored in memory in linear order, so the pointer returned by `cuda_tile.get_global` points to the first element, and offsetting the pointer by x would allow to load element at position x.

`global` operations must be directly nested within the **Tile IR** module. They cannot be defined inside functions. As globals are defined at the module scope their names are globally unique symbols and must not collide with any other symbol in the module.

For more detailed semantics of global variables see Global Variable.

#### Constraints

* The operation must be a symbol in the global symbol table.

#### Examples

```
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

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The result of the iota operation. 13.1

#### Description

The `iota` operation generates a 1-d tile with a sequence of integer values. The starting value is `0` and the stride is `1`. If the shape of the result tile is `(n)`, then the generated values are `[0, n - 1]`.

> **Note**
> 
> The number of elements in the result tile must not exceed the maximum value that the element type can express.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.3.11. cuda_tile.module

*Top-level module containing a series of defined items.*

```
cuda_tile.module %sym_name
```

#### Parameters

* **sym_name** (Symbol) - The name of the module. 13.1

#### Results

No results.

#### Description

A `module` operation represents a single compilation unit and contains zero or more items (global variables, functions, or kernels).

For detailed description of the semantics of modules, and the full definition of each item type see Modules.

The `module` operation is the top-level operation in a **Tile IR** module and must contain only **Tile IR** operations and no other dialects.

#### Constraints

* The region must not capture SSA values defined above the operation.
* The operation must provide custom parsing and printing methods.
* All regions must have zero arguments.
* Each provided region must contain exactly one block.
* The operation must define a symbol scope.
* The region must not require explicit terminator operations.
* The operation must specify whether regions are SSACFG or Graph kind.
* The operation must contain only dataflow graph regions.

### 8.3.12. cuda_tile.offset

*Offsets a tile of pointers*

```
cuda_tile.offset %ptr %offset
```

#### Parameters

* **ptr** (ptr) - The base pointer tile to advance. 13.1
* **offset** (tile<i1 | i8 | i16 | i32 | i64>) - The offset tile to add to the pointer. 13.1

#### Results

* **result** (ptr) - The resulting pointer tile after advancement. 13.1

#### Description

`offset` advances a tile of pointers. It takes `ptr` as base and `offset` as increment, and performs element-wise addition of `ptr` by `offset`:

```
result[i,j] = ptr[i,j] + offset[i,j] * bitwidth
```

`ptr` is interpreted as an unsigned integer. `offset` is interpreted as a signed integer. `bitwidth` is the storage bitwidth of the pointee type. The multiplication must not overflow (wrap-around) in a signed sense. The addition must not overflow (wrap-around) in an unsigned sense. In case of an overflow, the result is undefined.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.
* The operation must apply element-wise to its operands.
* `ptr`, `offset` and `result` must have the same shape.
* `result` and `ptr` must have the same shape and element type (ptr).
* The operation's result type may be inferred from its operands and attributes.

## 8.4. Conversions

There are no implicit type conversions in **Tile IR** thus we expose a set of explicit conversion operations for interconverting between types which have compatible representations or rules for conversion.

`cuda_tile.bitcast` preserves the contents of the input but allows for changing of element types, `cuda_tile.exti` and `cuda_tile.trunci` change the width of integer tiles, `cuda_tile.ftoi` and `cuda_tile.itof` convert floating-point tiles to integer tiles and vice versa, and `cuda_tile.ftof` converts between different floating-point types.

For more details on conversions and their rules see the individual operation's documentation.

### 8.4.1. cuda_tile.bitcast

*Bitcast a tile from one element type to another*

```
cuda_tile.bitcast %source
```

#### Parameters

* **source** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The source tile to cast. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The casted tile. 13.1

#### Description

The `bitcast` operation casts the input tile from one element type to another without modifying the underlying bits.

Only non-pointer types of the same bit width are allowed (e.g., `i32` to `f32`). Pointer types must use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr` instead.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.2. cuda_tile.exti

*Extend the width of an integer tile*

```
cuda_tile.exti %from %signedness
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to extend. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The extended integer tile. 13.1

#### Description

The `exti` operation converts a tile of integers of a given width to a strictly larger width. Zero-extension is used for `unsigned` integers and sign-extension is used for `signed` integers.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.3. cuda_tile.ftof

*Convert between floating-point types*

```
cuda_tile.ftof %from %rounding_mode
```

#### Parameters

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The result floating-point tile. 13.1

#### Description

The `ftof` operation converts a tile of a given floating-point element type into one of a different floating-point element type (for example, from `f32` to `f64`).

The source type and the result type must be different.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.4. cuda_tile.ftoi

*Convert a tile from floating-point values to integer values*

```
cuda_tile.ftoi %from %signedness %rounding_mode
```

#### Parameters

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The result integer tile. 13.1

#### Description

The `ftoi` operation converts a floating-point tile into an integer tile.

In contrast to a `bitcast` which is bits preserving, this preserves the numerical value of the tile, rounded towards zero to the nearest integer of the provided type.

> **Warning**
> 
> If the input floating-point value, after being rounded, is outside the (signed or unsigned) range of the target integer type, the closest representable value is used instead. `NaN` values are converted to 0. Input `Inf` values are undefined behavior.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.5. cuda_tile.itof

*Convert integer to floating-point*

```
cuda_tile.itof %from %signedness %rounding_mode
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The converted floating-point tile. 13.1

#### Description

The `itof` operation converts an integer tile into a float tile. In contrast to a bitcast, this preserves the numerical value of the tile, rounded to the nearest floating-point number of the provided type.

> **Warning**
> 
> If the input integer value, after being rounded, is outside the range of the target floating-point type, it is converted to `Inf` for types that support that value, and `NaN` otherwise.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.6. cuda_tile.int_to_ptr

*Convert a tile of integers to a tile of pointers*

```
cuda_tile.int_to_ptr %source
```

#### Parameters

* **source** (tile<i64>) - The input tile of integers. 13.1

#### Results

* **result** (ptr) - The output tile of pointers. 13.1

#### Description

The `int_to_ptr` operation converts a tile of integers to a tile of pointers.

The inverse of this operation is `cuda_tile.ptr_to_int`.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.7. cuda_tile.ptr_to_int

*Convert a tile of pointers to a tile of integers*

```
cuda_tile.ptr_to_int %source
```

#### Parameters

* **source** (ptr) - The input tile of pointers. 13.1

#### Results

* **result** (tile<i64>) - The output tile of integers. 13.1

#### Description

The `ptr_to_int` operation converts a tile of pointer-type elements to a tile of `i64` elements.

The inverse of this operation is `cuda_tile.int_to_ptr`.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.8. cuda_tile.ptr_to_ptr

*Reinterpret a tile of one pointer type as another*

```
cuda_tile.ptr_to_ptr %source
```

#### Parameters

* **source** (ptr) - Tile with source pointer element type. 13.1

#### Results

* **result** (ptr) - Tile with target pointer element type. 13.1

#### Description

The `ptr_to_ptr` operation casts a tile of pointers from a pointer of one element type to another element. Casts between pointer and non-pointer types are disallowed.

In order to perform those conversions, use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr`. These operations are distinct to enable future compiler reasoning about pointer provenance.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.9. cuda_tile.trunci

*Truncates the width of an integer tile*

```
cuda_tile.trunci %from %overflow
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to truncate. 13.1
* **overflow** (IntegerOverflow) - The overflow behavior of the operation. 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The truncated integer tile. 13.1

#### Description

The `trunci` operation converts a tile of integers of a given element type to one with a strictly smaller width.

The optional overflow attribute specifies whether an overflow can occur when interpreting the operand as a signed and/or unsigned integer. In case of "no signed wrap", all truncated bits must have the same value as the most significant bit of the truncated result. In case of "no unsigned wrap", the truncated bits must be zero.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

* `none` - The compiler makes no assumptions regarding overflow behavior.
* `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
* `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
* `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.3.13. cuda_tile.permute

*Permute tile dimensions*

```
cuda_tile.permute %source %permutation
```

#### Parameters

* **source** (tile) - The input tile. 13.1
* **permutation** (Array<i32>) - The permutation of the dimensions. 13.1

#### Results

* **result** (tile) - The permuted tile. 13.1

#### Description

Permute the dimensions of the input tile `source` according to the `permutation` array. The `permutation` array is a list of integers that specify the new order of the dimensions.

For example, if the input tile has shape `[2, 4, 8]`, and the permutation is `[2, 0, 1]`, the output tile will have shape `[8, 2, 4]`.

This operation logically is a change in the indexing of the tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.
* `source` and `result` must have the same element type (tile).
* `source` and `result` must have the same rank.

#### Examples

```
%arg0 = constant dense<0.0> : tile<2x4x8xf16>
%0 = permute %arg0 [2, 0, 1] : tile<2x4x8xf16> -> tile<8x2x4xf16>
```

### 8.3.14. cuda_tile.reduce

*Variadic tile reduction across dimensions*

```
cuda_tile.reduce %operands %dim %identities
```

#### Parameters

* **operands** (Variadic<tile>) - The tiles to reduce. 13.1
* **dim** (i32) - The index dimension that needs to be reduced. 13.1
* **identities** (Array) - The reduction identities for each operand. 13.1

#### Results

* **results** (Variadic<tile>) - The reduced tiles. 13.1

#### Description

Applies a reduction function `body` to `operands` and `identities` along dimensions `dimensions` and produces new `results` tile values. The order of reduction is implementation-defined but the result is deterministic.

Argument explained:

* `operands` are the tiles to reduce.
* `identities` are the reduction identities for each operand. Identity at position i binds with the operand at the same position. Identities are properties of the reduction function in the `body`. For example, the identity of a min reduction is +inf, while the identity of a sum is 0.
* `dim` is the index of the dimension to be reduced.
* `body` is a region carrying the reduction(s) semantics. Each operation within the region must be a cuda_tile operation with 0-rank cuda_tile tile types. Region arguments are bound to operands in the following way: `[operand_0_current_iter, operand_0_prev_iter, operand_1_current_iter, operand_1_prev_iter…]`. `operand_i_current_iter` is the current element to reduce from operand at index i. `operand_i_prev_iter` is the accumulator that might be an element of the same operand at index i, the result of the previous reduction step or the identity value associated with `operand_i_current_iter`.

#### Constraints

* The operation must provide custom parsing and printing methods.
* The operation only has an effect if and only if it the region's operation have an effect.
* All operands must have identical shapes.
* Each provided region must contain exactly one block.

### 8.3.15. cuda_tile.reshape

*Reshape tile dimensions*

```
cuda_tile.reshape %source
```

#### Parameters

* **source** (tile) - The source tile to reshape. 13.1

#### Results

* **result** (tile) - The reshaped tile. 13.1

#### Description

The `reshape` operation changes the shape of the `source` operand. `reshape` is only a change in the indexing of the tile. The number of elements and element type must remain unchanged.

0-d tiles (i.e., scalars) contain precisely one element and thus are the one exception where a 0-d tile can be reshaped to shape where the `size(shape) == 1`.

Conceptually reshaping a tile is equivalent to first creating a 1-d tile from the data of the source assuming a row-major layout and then converting the 1-d tile into the new shape in a row-major layout.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.
* `source` and `result` must have the same element type (tile).

#### Examples

```
%cst = constant dense<0> : tile<i8>
%0 = reshape %cst
    : tile<i8> -> tile<1x1x1xi8>

%t = constant dense<0.0> : tile<8x2xf32>
%1 = reshape %t
    : tile<8x2xf32> -> tile<2x2x4x1xf32>
```

```
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

### 8.3.16. cuda_tile.scan

*A parallel prefix sum operation*

```
cuda_tile.scan %operands %dim %reverse %identities
```

#### Parameters

* **operands** (Variadic<tile>) - The a set of tiles to scan. 13.1
* **dim** (i32) - The index of the dimension along which to scan. 13.1
* **reverse** (bool) - Whether to scan in reverse order. 13.1
* **identities** (Array) - The identities of the scan operation. 13.1

#### Results

* **results** (Variadic<tile>) - The resulting tiles from the scan operation. 13.1

#### Description

Applies a scan function `body` to `operands` and `identities` along dimension `dim` and produces new `results` tile values. The scan operation maintains a carry value that is updated as it processes elements along the specified dimension. For each element, the scan function combines the current element with the carry value to produce both a result and an updated carry. The order of scan is implementation-defined but the result is deterministic.

`identities` are the scan identities for each operand. Identity at position i binds with the operand at the same position. Identities are properties of the scan function in the `body`. For example, the identity of a min scan is +inf, while the identity of a sum is 0.

`body` is a region carrying the scan semantics. Each operation within the region must be a cuda_tile operation with 0-rank cuda_tile tile types. Region arguments are bound to operands in the following way: `[operand_0_current_iter, operand_0_prev_iter, operand_1_current_iter, operand_1_prev_iter...]`. `operand_i_current_iter` is the current element to scan from operand at index `i`. `operand_i_prev_iter` is the accumulator that might be an element of the same operand at index `i`, the result of the previous scan step or the identity value associated with `operand_i_current_iter`.

> **Warning**
> 
> The current implementation only supports single tile input.

#### Constraints

* The operation must provide custom parsing and printing methods.
* The operation only has an effect if and only if it the region's operation have an effect.
* All operands must have identical shapes.
* Each provided region must contain exactly one block.

#### Examples

```
%input = constant dense<0.0> : tile<8x16xf32>
%result = scan %input dim=1 reverse=false identities=[1.0 : f32] : tile<8x16xf32> -> tile<8x16xf32>
(%acc: tile<f32>, %elem: tile<f32>) {
  %prod = mulf %acc, %elem rounding<nearest_even>: tile<f32>
  yield %prod : tile<f32>
}
```

### 8.3.17. cuda_tile.select

*Select values based on condition*

```
cuda_tile.select %cond %val_if_true %val_if_false
```

#### Parameters

* **cond** (tile<i1>) - The condition tile. 13.1
* **val_if_true** (tile) - The value if true tile. 13.1
* **val_if_false** (tile) - The value if false tile. 13.1

#### Results

* **result** (tile) - The tile of selected values. 13.1

#### Description

The `select` op chooses values based on the binary conditions supplied as the `cond` operand. The `val_if_true` operand contains the value(s) to use if the condition is 1. The `val_if_false` operand contains the value(s) to use if the condition is 0. The choice is made element-wise according to the values in the condition tile.

All tiles must have the same shape. The tiles `val_if_true`, `val_if_false`, and the result must have the same element type. The `cond` tile must be a tile of `i1` values.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.
* `val_if_true`, `val_if_false` and `result` must have the same shape and element type (tile).
* The operation's result type may be inferred from its operands and attributes.

## 8.4. Conversions

There are no implicit type conversions in **Tile IR** thus we expose a set of explicit conversion operations for interconverting between types which have compatible representations or rules for conversion.

`cuda_tile.bitcast` preserves the contents of the input but allows for changing of element types, `cuda_tile.exti` and `cuda_tile.trunci` change the width of integer tiles, `cuda_tile.ftoi` and `cuda_tile.itof` convert floating-point tiles to integer tiles and vice versa, and `cuda_tile.ftof` converts between different floating-point types.

For more details on conversions and their rules see the individual operation's documentation.

### 8.4.1. cuda_tile.bitcast

*Bitcast a tile from one element type to another*

```
cuda_tile.bitcast %source
```

#### Parameters

* **source** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The source tile to cast. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The casted tile. 13.1

#### Description

The `bitcast` operation casts the input tile from one element type to another without modifying the underlying bits.

Only non-pointer types of the same bit width are allowed (e.g., `i32` to `f32`). Pointer types must use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr` instead.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.2. cuda_tile.exti

*Extend the width of an integer tile*

```
cuda_tile.exti %from %signedness
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to extend. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The extended integer tile. 13.1

#### Description

The `exti` operation converts a tile of integers of a given width to a strictly larger width. Zero-extension is used for `unsigned` integers and sign-extension is used for `signed` integers.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.3. cuda_tile.ftof

*Convert between floating-point types*

```
cuda_tile.ftof %from %rounding_mode
```

#### Parameters

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The result floating-point tile. 13.1

#### Description

The `ftof` operation converts a tile of a given floating-point element type into one of a different floating-point element type (for example, from `f32` to `f64`).

The source type and the result type must be different.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.4. cuda_tile.ftoi

*Convert a tile from floating-point values to integer values*

```
cuda_tile.ftoi %from %signedness %rounding_mode
```

#### Parameters

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The result integer tile. 13.1

#### Description

The `ftoi` operation converts a floating-point tile into an integer tile.

In contrast to a `bitcast` which is bits preserving, this preserves the numerical value of the tile, rounded towards zero to the nearest integer of the provided type.

> **Warning**
> 
> If the input floating-point value, after being rounded, is outside the (signed or unsigned) range of the target integer type, the closest representable value is used instead. `NaN` values are converted to 0. Input `Inf` values are undefined behavior.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.5. cuda_tile.itof

*Convert integer to floating-point*

```
cuda_tile.itof %from %signedness %rounding_mode
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

#### Results

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The converted floating-point tile. 13.1

#### Description

The `itof` operation converts an integer tile into a float tile. In contrast to a bitcast, this preserves the numerical value of the tile, rounded to the nearest floating-point number of the provided type.

> **Warning**
> 
> If the input integer value, after being rounded, is outside the range of the target floating-point type, it is converted to `Inf` for types that support that value, and `NaN` otherwise.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.6. cuda_tile.int_to_ptr

*Convert a tile of integers to a tile of pointers*

```
cuda_tile.int_to_ptr %source
```

#### Parameters

* **source** (tile<i64>) - The input tile of integers. 13.1

#### Results

* **result** (ptr) - The output tile of pointers. 13.1

#### Description

The `int_to_ptr` operation converts a tile of integers to a tile of pointers.

The inverse of this operation is `cuda_tile.ptr_to_int`.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.7. cuda_tile.ptr_to_int

*Convert a tile of pointers to a tile of integers*

```
cuda_tile.ptr_to_int %source
```

#### Parameters

* **source** (ptr) - The input tile of pointers. 13.1

#### Results

* **result** (tile<i64>) - The output tile of integers. 13.1

#### Description

The `ptr_to_int` operation converts a tile of pointer-type elements to a tile of `i64` elements.

The inverse of this operation is `cuda_tile.int_to_ptr`.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.8. cuda_tile.ptr_to_ptr

*Reinterpret a tile of one pointer type as another*

```
cuda_tile.ptr_to_ptr %source
```

#### Parameters

* **source** (ptr) - Tile with source pointer element type. 13.1

#### Results

* **result** (ptr) - Tile with target pointer element type. 13.1

#### Description

The `ptr_to_ptr` operation casts a tile of pointers from a pointer of one element type to another element. Casts between pointer and non-pointer types are disallowed.

In order to perform those conversions, use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr`. These operations are distinct to enable future compiler reasoning about pointer provenance.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.9. cuda_tile.trunci

*Truncates the width of an integer tile*

```
cuda_tile.trunci %from %overflow
```

#### Parameters

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to truncate. 13.1
* **overflow** (IntegerOverflow) - The overflow behavior of the operation. 13.1

#### Results

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The truncated integer tile. 13.1

#### Description

The `trunci` operation converts a tile of integers of a given element type to one with a strictly smaller width.

The optional overflow attribute specifies whether an overflow can occur when interpreting the operand as a signed and/or unsigned integer. In case of "no signed wrap", all truncated bits must have the same value as the most significant bit of the truncated result. In case of "no unsigned wrap", the truncated bits must be zero.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

* `none` - The compiler makes no assumptions regarding overflow behavior.
* `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
* `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
* `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.


## 8.4. Conversions

There are no implicit type conversions in **Tile IR** thus we expose a set of explicit conversion operations for interconverting between types which have compatible representations or rules for conversion.

`cuda_tile.bitcast` preserves the contents of the input but allows for changing of element types, `cuda_tile.exti` and `cuda_tile.trunci` change the width of integer tiles, `cuda_tile.ftoi` and `cuda_tile.itof` convert floating-point tiles to integer tiles and vice versa, and `cuda_tile.ftof` converts between different floating-point types.

For more details on conversions and their rules see the individual operation's documentation.

### 8.4.1. cuda_tile.bitcast

*Bitcast a tile from one element type to another*

```
cuda_tile.bitcast %source
```

**Parameters**

* **source** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The source tile to cast. 13.1

**Results**

* **result** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The casted tile. 13.1

**Description**

The `bitcast` operation casts the input tile from one element type to another without modifying the underlying bits.

Only non-pointer types of the same bit width are allowed (e.g., `i32` to `f32`). Pointer types must use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr` instead.

**Constraints**

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.2. cuda_tile.exti

*Extend the width of an integer tile*

```
cuda_tile.exti %from %signedness
```

**Parameters**

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to extend. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1

**Results**

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The extended integer tile. 13.1

**Description**

The `exti` operation converts a tile of integers of a given width to a strictly larger width. Zero-extension is used for `unsigned` integers and sign-extension is used for `signed` integers.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

**Constraints**

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.3. cuda_tile.ftof

*Convert between floating-point types*

```
cuda_tile.ftof %from %rounding_mode
```

**Parameters**

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

**Results**

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The result floating-point tile. 13.1

**Description**

The `ftof` operation converts a tile of a given floating-point element type into one of a different floating-point element type (for example, from `f32` to `f64`).

The source type and the result type must be different.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

**Constraints**

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.4. cuda_tile.ftoi

*Convert a tile from floating-point values to integer values*

```
cuda_tile.ftoi %from %signedness %rounding_mode
```

**Parameters**

* **from** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input floating-point tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

**Results**

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The result integer tile. 13.1

**Description**

The `ftoi` operation converts a floating-point tile into an integer tile.

In contrast to a `bitcast` which is bits preserving, this preserves the numerical value of the tile, rounded towards zero to the nearest integer of the provided type.

> **Warning**
> 
> If the input floating-point value, after being rounded, is outside the (signed or unsigned) range of the target integer type, the closest representable value is used instead. `NaN` values are converted to 0. Input `Inf` values are undefined behavior.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

**Constraints**

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.5. cuda_tile.itof

*Convert integer to floating-point*

```
cuda_tile.itof %from %signedness %rounding_mode
```

**Parameters**

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile. 13.1
* **signedness** (Signedness) - Interpret integer(s) as `signed` or `unsigned` 13.1
* **rounding_mode** (RoundingMode) - The rounding mode for the operation. 13.1

**Results**

* **to** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The converted floating-point tile. 13.1

**Description**

The `itof` operation converts an integer tile into a float tile. In contrast to a bitcast, this preserves the numerical value of the tile, rounded to the nearest floating-point number of the provided type.

> **Warning**
> 
> If the input integer value, after being rounded, is outside the range of the target floating-point type, it is converted to `Inf` for types that support that value, and `NaN` otherwise.

The `signedness` attribute specifies the signedness of operand(s).

* `unsigned` - Treat the operands as unsigned integers.
* `signed` - Treat the operands as signed integers.

The `rounding` attribute specifies the rounding mode to use for the operation.

* `nearest_even` - Round to nearest (ties to even).
* `zero` - Round towards zero (truncate).
* `negative_inf` - Round towards negative infinity.
* `positive_inf` - Round towards positive infinity.
* `approx` - Approximate rounding mode.
* `full` - Full precision rounding mode.
* `nearest_int_to_zero` - Round towards zero to the nearest integer.

**Constraints**

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.6. cuda_tile.int_to_ptr

*Convert a tile of integers to a tile of pointers*

```
cuda_tile.int_to_ptr %source
```

**Parameters**

* **source** (tile<i64>) - The input tile of integers. 13.1

**Results**

* **result** (ptr) - The output tile of pointers. 13.1

**Description**

The `int_to_ptr` operation converts a tile of integers to a tile of pointers.

The inverse of this operation is `cuda_tile.ptr_to_int`.

**Constraints**

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.7. cuda_tile.ptr_to_int

*Convert a tile of pointers to a tile of integers*

```
cuda_tile.ptr_to_int %source
```

**Parameters**

* **source** (ptr) - The input tile of pointers. 13.1

**Results**

* **result** (tile<i64>) - The output tile of integers. 13.1

**Description**

The `ptr_to_int` operation converts a tile of pointer-type elements to a tile of `i64` elements.

The inverse of this operation is `cuda_tile.int_to_ptr`.

**Constraints**

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.8. cuda_tile.ptr_to_ptr

*Reinterpret a tile of one pointer type as another*

```
cuda_tile.ptr_to_ptr %source
```

**Parameters**

* **source** (ptr) - Tile with source pointer element type. 13.1

**Results**

* **result** (ptr) - Tile with target pointer element type. 13.1

**Description**

The `ptr_to_ptr` operation casts a tile of pointers from a pointer of one element type to another element. Casts between pointer and non-pointer types are disallowed.

In order to perform those conversions, use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr`. These operations are distinct to enable future compiler reasoning about pointer provenance.

**Constraints**

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.4.9. cuda_tile.trunci

*Truncates the width of an integer tile*

```
cuda_tile.trunci %from %overflow
```

**Parameters**

* **from** (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to truncate. 13.1
* **overflow** (IntegerOverflow) - The overflow behavior of the operation. 13.1

**Results**

* **to** (tile<i1 | i8 | i16 | i32 | i64>) - The truncated integer tile. 13.1

**Description**

The `trunci` operation converts a tile of integers of a given element type to one with a strictly smaller width.

The optional overflow attribute specifies whether an overflow can occur when interpreting the operand as a signed and/or unsigned integer. In case of "no signed wrap", all truncated bits must have the same value as the most significant bit of the truncated result. In case of "no unsigned wrap", the truncated bits must be zero.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

* `none` - The compiler makes no assumptions regarding overflow behavior.
* `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
* `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
* `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

**Constraints**

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.


## 8.5. Control Flow

Control flow operations allow for the manipulation of execution within the module.

### 8.5.1. cuda_tile.br

*Unconditional branch*

```
cuda_tile.br ^target
```

#### Parameters

* **target** (Block) - The target block to branch to.

#### Description

The `br` operation represents an unconditional branch to a target block.

The target block must be a successor of the current block.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation is pure and does not perform any memory side effects.

### 8.5.2. cuda_tile.cond_br

*Conditional branch*

```
cuda_tile.cond_br %condition ^true_block ^false_block
```

#### Parameters

* **condition** (tile<i1>) - The condition to evaluate. 13.1
* **true_block** (Block) - The block to branch to if the condition is true.
* **false_block** (Block) - The block to branch to if the condition is false.

#### Description

The `cond_br` operation represents a conditional branch based on a boolean condition.

If the condition is true, execution continues at the true block; otherwise, it continues at the false block.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation is pure and does not perform any memory side effects.

### 8.5.3. cuda_tile.return

*Return from a function*

```
cuda_tile.return %results
```

#### Parameters

* **results** (Variadic<Any>) - The values to return.

#### Description

The `return` operation terminates the execution of a function and returns the specified values.

The types of the returned values must match the function's return type.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation is pure and does not perform any memory side effects.

### 8.5.4. cuda_tile.yield

*Yield values from a block*

```
cuda_tile.yield %values
```

#### Parameters

* **values** (Variadic<Any>) - The values to yield.

#### Description

The `yield` operation yields values from a block to its parent operation.

This is typically used in structured control flow operations like `if` and `for`.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation is pure and does not perform any memory side effects.


## 8.6. Memory

Memory operations allow for the manipulation of memory within the module.

### 8.6.1. cuda_tile.load

*Load a tile from memory*

```
cuda_tile.load %ptr %alignment %is_volatile %cache_mode
```

#### Parameters

* **ptr** (ptr) - The pointer to load from. 13.1
* **alignment** (i64) - The alignment of the memory access. 13.1
* **is_volatile** (bool) - Whether the load is volatile. 13.1
* **cache_mode** (CacheMode) - The cache mode for the load. 13.1

#### Results

* **result** (tile<Any>) - The loaded tile. 13.1

#### Description

The `load` operation loads a tile from the specified memory address.

The `cache_mode` attribute specifies how the load should interact with the cache:

* `always` - Always cache the loaded data.
* `never` - Never cache the loaded data.
* `global` - Cache at the global level.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation reads from memory.

### 8.6.2. cuda_tile.store

*Store a tile to memory*

```
cuda_tile.store %ptr %value %alignment %is_volatile %cache_mode
```

#### Parameters

* **ptr** (ptr) - The pointer to store to. 13.1
* **value** (tile<Any>) - The tile to store. 13.1
* **alignment** (i64) - The alignment of the memory access. 13.1
* **is_volatile** (bool) - Whether the store is volatile. 13.1
* **cache_mode** (CacheMode) - The cache mode for the store. 13.1

#### Description

The `store` operation stores a tile to the specified memory address.

The `cache_mode` attribute specifies how the store should interact with the cache:

* `always` - Always write through to memory.
* `never` - Write only to cache.
* `global` - Write to global memory.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation writes to memory.

### 8.6.3. cuda_tile.alloca

*Allocate stack memory*

```
cuda_tile.alloca %size %alignment
```

#### Parameters

* **size** (i64) - The number of bytes to allocate. 13.1
* **alignment** (i64) - The alignment of the allocation. 13.1

#### Results

* **result** (ptr) - The pointer to the allocated memory. 13.1

#### Description

The `alloca` operation allocates memory on the stack.

The allocated memory is automatically freed when the function returns.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation may have side effects (allocation).


## 8.7. Floating Point

Floating point operations allow for the manipulation of floating-point values within the module.

### 8.7.1. cuda_tile.addf

*Add two floating-point tiles*

```
cuda_tile.addf %lhs %rhs
```

#### Parameters

* **lhs** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The left-hand side operand. 13.1
* **rhs** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The right-hand side operand. 13.1

#### Results

* **result** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The sum of the operands. 13.1

#### Description

The `addf` operation performs element-wise addition of two floating-point tiles.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.7.2. cuda_tile.subf

*Subtract two floating-point tiles*

```
cuda_tile.subf %lhs %rhs
```

#### Parameters

* **lhs** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The left-hand side operand. 13.1
* **rhs** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The right-hand side operand. 13.1

#### Results

* **result** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The difference of the operands. 13.1

#### Description

The `subf` operation performs element-wise subtraction of two floating-point tiles.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.7.3. cuda_tile.mulf

*Multiply two floating-point tiles*

```
cuda_tile.mulf %lhs %rhs
```

#### Parameters

* **lhs** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The left-hand side operand. 13.1
* **rhs** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The right-hand side operand. 13.1

#### Results

* **result** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The product of the operands. 13.1

#### Description

The `mulf` operation performs element-wise multiplication of two floating-point tiles.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.7.4. cuda_tile.divf

*Divide two floating-point tiles*

```
cuda_tile.divf %lhs %rhs
```

#### Parameters

* **lhs** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The left-hand side operand. 13.1
* **rhs** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The right-hand side operand. 13.1

#### Results

* **result** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The quotient of the operands. 13.1

#### Description

The `divf` operation performs element-wise division of two floating-point tiles.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.7.5. cuda_tile.fmaf

*Fused multiply-add*

```
cuda_tile.fmaf %a %b %c
```

#### Parameters

* **a** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The first operand. 13.1
* **b** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The second operand. 13.1
* **c** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The third operand. 13.1

#### Results

* **result** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The result of `a * b + c`. 13.1

#### Description

The `fmaf` operation performs a fused multiply-add: `a * b + c` with a single rounding.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.7.6. cuda_tile.absf

*Absolute value of floating-point tile*

```
cuda_tile.absf %operand
```

#### Parameters

* **operand** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input operand. 13.1

#### Results

* **result** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The absolute value. 13.1

#### Description

The `absf` operation computes the absolute value of each element in the input tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.7.7. cuda_tile.negf

*Negate a floating-point tile*

```
cuda_tile.negf %operand
```

#### Parameters

* **operand** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input operand. 13.1

#### Results

* **result** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The negated value. 13.1

#### Description

The `negf` operation negates each element in the input tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.7.8. cuda_tile.sqrtf

*Square root of floating-point tile*

```
cuda_tile.sqrtf %operand
```

#### Parameters

* **operand** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input operand. 13.1

#### Results

* **result** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The square root. 13.1

#### Description

The `sqrtf` operation computes the square root of each element in the input tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.7.9. cuda_tile.expf

*Exponential of floating-point tile*

```
cuda_tile.expf %operand
```

#### Parameters

* **operand** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input operand. 13.1

#### Results

* **result** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The exponential. 13.1

#### Description

The `expf` operation computes `e^x` for each element in the input tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.7.10. cuda_tile.logf

*Natural logarithm of floating-point tile*

```
cuda_tile.logf %operand
```

#### Parameters

* **operand** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input operand. 13.1

#### Results

* **result** (tile<f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The natural logarithm. 13.1

#### Description

The `logf` operation computes the natural logarithm of each element in the input tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.


## 8.8. Integer

Integer operations allow for the manipulation of integer values within the module.

### 8.8.1. cuda_tile.addi

*Add two integer tiles*

```
cuda_tile.addi %lhs %rhs %overflow
```

#### Parameters

* **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left-hand side operand. 13.1
* **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right-hand side operand. 13.1
* **overflow** (IntegerOverflow) - The overflow behavior. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The sum of the operands. 13.1

#### Description

The `addi` operation performs element-wise addition of two integer tiles.

The overflow attribute specifies the overflow behavior:

* `none` - The compiler makes no assumptions regarding overflow behavior.
* `no_signed_wrap` - The compiler assumes that overflow will not occur for signed integers.
* `no_unsigned_wrap` - The compiler assumes that overflow will not occur for unsigned integers.
* `no_wrap` - The compiler assumes that overflow will not occur for signed or unsigned integers.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.8.2. cuda_tile.subi

*Subtract two integer tiles*

```
cuda_tile.subi %lhs %rhs %overflow
```

#### Parameters

* **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left-hand side operand. 13.1
* **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right-hand side operand. 13.1
* **overflow** (IntegerOverflow) - The overflow behavior. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The difference of the operands. 13.1

#### Description

The `subi` operation performs element-wise subtraction of two integer tiles.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.8.3. cuda_tile.muli

*Multiply two integer tiles*

```
cuda_tile.muli %lhs %rhs %overflow
```

#### Parameters

* **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left-hand side operand. 13.1
* **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right-hand side operand. 13.1
* **overflow** (IntegerOverflow) - The overflow behavior. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The product of the operands. 13.1

#### Description

The `muli` operation performs element-wise multiplication of two integer tiles.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.8.4. cuda_tile.divi

*Divide two integer tiles*

```
cuda_tile.divi %lhs %rhs %signedness
```

#### Parameters

* **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left-hand side operand. 13.1
* **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right-hand side operand. 13.1
* **signedness** (Signedness) - Whether to perform signed or unsigned division. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The quotient of the operands. 13.1

#### Description

The `divi` operation performs element-wise division of two integer tiles.

The signedness attribute specifies whether to treat the operands as signed or unsigned integers.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.8.5. cuda_tile.remi

*Remainder of integer division*

```
cuda_tile.remi %lhs %rhs %signedness
```

#### Parameters

* **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left-hand side operand. 13.1
* **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right-hand side operand. 13.1
* **signedness** (Signedness) - Whether to perform signed or unsigned remainder. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The remainder of the division. 13.1

#### Description

The `remi` operation computes the remainder of element-wise integer division.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.8.6. cuda_tile.absi

*Absolute value of integer tile*

```
cuda_tile.absi %operand
```

#### Parameters

* **operand** (tile<i1 | i8 | i16 | i32 | i64>) - The input operand. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The absolute value. 13.1

#### Description

The `absi` operation computes the absolute value of each element in the input tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.8.7. cuda_tile.negi

*Negate an integer tile*

```
cuda_tile.negi %operand
```

#### Parameters

* **operand** (tile<i1 | i8 | i16 | i32 | i64>) - The input operand. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The negated value. 13.1

#### Description

The `negi` operation negates each element in the input tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.


## 8.9. Bitwise

Bitwise operations allow for the manipulation of bits within integer values.

### 8.9.1. cuda_tile.and

*Bitwise AND of two integer tiles*

```
cuda_tile.and %lhs %rhs
```

#### Parameters

* **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left-hand side operand. 13.1
* **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right-hand side operand. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The bitwise AND of the operands. 13.1

#### Description

The `and` operation performs element-wise bitwise AND of two integer tiles.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.9.2. cuda_tile.or

*Bitwise OR of two integer tiles*

```
cuda_tile.or %lhs %rhs
```

#### Parameters

* **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left-hand side operand. 13.1
* **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right-hand side operand. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The bitwise OR of the operands. 13.1

#### Description

The `or` operation performs element-wise bitwise OR of two integer tiles.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.9.3. cuda_tile.xor

*Bitwise XOR of two integer tiles*

```
cuda_tile.xor %lhs %rhs
```

#### Parameters

* **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left-hand side operand. 13.1
* **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right-hand side operand. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The bitwise XOR of the operands. 13.1

#### Description

The `xor` operation performs element-wise bitwise XOR of two integer tiles.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.9.4. cuda_tile.not

*Bitwise NOT of an integer tile*

```
cuda_tile.not %operand
```

#### Parameters

* **operand** (tile<i1 | i8 | i16 | i32 | i64>) - The input operand. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The bitwise NOT of the operand. 13.1

#### Description

The `not` operation performs element-wise bitwise NOT of an integer tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.9.5. cuda_tile.shl

*Shift left*

```
cuda_tile.shl %value %amount
```

#### Parameters

* **value** (tile<i1 | i8 | i16 | i32 | i64>) - The value to shift. 13.1
* **amount** (tile<i1 | i8 | i16 | i32 | i64>) - The amount to shift by. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The shifted value. 13.1

#### Description

The `shl` operation performs element-wise left shift of an integer tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.9.6. cuda_tile.shr

*Shift right*

```
cuda_tile.shr %value %amount %signedness
```

#### Parameters

* **value** (tile<i1 | i8 | i16 | i32 | i64>) - The value to shift. 13.1
* **amount** (tile<i1 | i8 | i16 | i32 | i64>) - The amount to shift by. 13.1
* **signedness** (Signedness) - Whether to perform arithmetic (signed) or logical (unsigned) shift. 13.1

#### Results

* **result** (tile<i1 | i8 | i16 | i32 | i64>) - The shifted value. 13.1

#### Description

The `shr` operation performs element-wise right shift of an integer tile.

For signed shift, the sign bit is propagated (arithmetic shift).
For unsigned shift, zeros are shifted in (logical shift).

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.


## 8.10. Atomics

Atomic operations allow for the manipulation of memory with atomic semantics.

### 8.10.1. cuda_tile.atomic_load

*Atomically load from memory*

```
cuda_tile.atomic_load %ptr %ordering
```

#### Parameters

* **ptr** (ptr) - The pointer to load from. 13.1
* **ordering** (MemoryOrdering) - The memory ordering constraint. 13.1

#### Results

* **result** (tile<Any>) - The loaded value. 13.1

#### Description

The `atomic_load` operation atomically loads a value from memory.

The memory ordering specifies the synchronization semantics:

* `relaxed` - No synchronization guarantees.
* `acquire` - Synchronizes with release stores.
* `seq_cst` - Sequentially consistent ordering.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation reads from memory.

### 8.10.2. cuda_tile.atomic_store

*Atomically store to memory*

```
cuda_tile.atomic_store %ptr %value %ordering
```

#### Parameters

* **ptr** (ptr) - The pointer to store to. 13.1
* **value** (tile<Any>) - The value to store. 13.1
* **ordering** (MemoryOrdering) - The memory ordering constraint. 13.1

#### Description

The `atomic_store` operation atomically stores a value to memory.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation writes to memory.

### 8.10.3. cuda_tile.atomic_add

*Atomically add to memory*

```
cuda_tile.atomic_add %ptr %value %ordering
```

#### Parameters

* **ptr** (ptr) - The pointer to the memory location. 13.1
* **value** (tile<Any>) - The value to add. 13.1
* **ordering** (MemoryOrdering) - The memory ordering constraint. 13.1

#### Results

* **result** (tile<Any>) - The old value at the memory location. 13.1

#### Description

The `atomic_add` operation atomically adds a value to a memory location and returns the old value.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation reads and writes memory.

### 8.10.4. cuda_tile.atomic_sub

*Atomically subtract from memory*

```
cuda_tile.atomic_sub %ptr %value %ordering
```

#### Parameters

* **ptr** (ptr) - The pointer to the memory location. 13.1
* **value** (tile<Any>) - The value to subtract. 13.1
* **ordering** (MemoryOrdering) - The memory ordering constraint. 13.1

#### Results

* **result** (tile<Any>) - The old value at the memory location. 13.1

#### Description

The `atomic_sub` operation atomically subtracts a value from a memory location and returns the old value.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation reads and writes memory.

### 8.10.5. cuda_tile.atomic_and

*Atomically AND with memory*

```
cuda_tile.atomic_and %ptr %value %ordering
```

#### Parameters

* **ptr** (ptr) - The pointer to the memory location. 13.1
* **value** (tile<Any>) - The value to AND. 13.1
* **ordering** (MemoryOrdering) - The memory ordering constraint. 13.1

#### Results

* **result** (tile<Any>) - The old value at the memory location. 13.1

#### Description

The `atomic_and` operation atomically ANDs a value with a memory location and returns the old value.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation reads and writes memory.

### 8.10.6. cuda_tile.atomic_or

*Atomically OR with memory*

```
cuda_tile.atomic_or %ptr %value %ordering
```

#### Parameters

* **ptr** (ptr) - The pointer to the memory location. 13.1
* **value** (tile<Any>) - The value to OR. 13.1
* **ordering** (MemoryOrdering) - The memory ordering constraint. 13.1

#### Results

* **result** (tile<Any>) - The old value at the memory location. 13.1

#### Description

The `atomic_or` operation atomically ORs a value with a memory location and returns the old value.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation reads and writes memory.

### 8.10.7. cuda_tile.atomic_xor

*Atomically XOR with memory*

```
cuda_tile.atomic_xor %ptr %value %ordering
```

#### Parameters

* **ptr** (ptr) - The pointer to the memory location. 13.1
* **value** (tile<Any>) - The value to XOR. 13.1
* **ordering** (MemoryOrdering) - The memory ordering constraint. 13.1

#### Results

* **result** (tile<Any>) - The old value at the memory location. 13.1

#### Description

The `atomic_xor` operation atomically XORs a value with a memory location and returns the old value.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation reads and writes memory.

### 8.10.8. cuda_tile.atomic_max

*Atomically compute maximum*

```
cuda_tile.atomic_max %ptr %value %ordering %signedness
```

#### Parameters

* **ptr** (ptr) - The pointer to the memory location. 13.1
* **value** (tile<Any>) - The value to compare. 13.1
* **ordering** (MemoryOrdering) - The memory ordering constraint. 13.1
* **signedness** (Signedness) - Whether to perform signed or unsigned comparison. 13.1

#### Results

* **result** (tile<Any>) - The old value at the memory location. 13.1

#### Description

The `atomic_max` operation atomically computes the maximum of a value and a memory location and returns the old value.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation reads and writes memory.

### 8.10.9. cuda_tile.atomic_min

*Atomically compute minimum*

```
cuda_tile.atomic_min %ptr %value %ordering %signedness
```

#### Parameters

* **ptr** (ptr) - The pointer to the memory location. 13.1
* **value** (tile<Any>) - The value to compare. 13.1
* **ordering** (MemoryOrdering) - The memory ordering constraint. 13.1
* **signedness** (Signedness) - Whether to perform signed or unsigned comparison. 13.1

#### Results

* **result** (tile<Any>) - The old value at the memory location. 13.1

#### Description

The `atomic_min` operation atomically computes the minimum of a value and a memory location and returns the old value.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation reads and writes memory.

### 8.10.10. cuda_tile.atomic_exchange

*Atomically exchange values*

```
cuda_tile.atomic_exchange %ptr %value %ordering
```

#### Parameters

* **ptr** (ptr) - The pointer to the memory location. 13.1
* **value** (tile<Any>) - The new value. 13.1
* **ordering** (MemoryOrdering) - The memory ordering constraint. 13.1

#### Results

* **result** (tile<Any>) - The old value at the memory location. 13.1

#### Description

The `atomic_exchange` operation atomically exchanges a value in memory and returns the old value.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation reads and writes memory.

### 8.10.11. cuda_tile.atomic_compare_exchange

*Atomically compare and exchange*

```
cuda_tile.atomic_compare_exchange %ptr %expected %desired %success_ordering %failure_ordering
```

#### Parameters

* **ptr** (ptr) - The pointer to the memory location. 13.1
* **expected** (tile<Any>) - The expected value. 13.1
* **desired** (tile<Any>) - The desired new value. 13.1
* **success_ordering** (MemoryOrdering) - The memory ordering on success. 13.1
* **failure_ordering** (MemoryOrdering) - The memory ordering on failure. 13.1

#### Results

* **result** (tile<Any>) - The value at the memory location (expected if successful). 13.1
* **success** (tile<i1>) - Whether the exchange was successful. 13.1

#### Description

The `atomic_compare_exchange` operation atomically compares a value in memory with an expected value and exchanges it if they match.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation reads and writes memory.


## 8.11. Views

View operations allow for the manipulation of tensor views within the module.

### 8.11.1. cuda_tile.view

*Create a view of a tile*

```
cuda_tile.view %source %offsets %strides %sizes
```

#### Parameters

* **source** (tile<Any>) - The source tile. 13.1
* **offsets** (Array<i64>) - The offsets for each dimension. 13.1
* **strides** (Array<i64>) - The strides for each dimension. 13.1
* **sizes** (Array<i64>) - The sizes for each dimension. 13.1

#### Results

* **result** (view_type) - A view of the source tile. 13.1

#### Description

The `view` operation creates a view of a tile with specified offsets, strides, and sizes.

This operation does not copy data; it only creates a new view into the existing data.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.11.2. cuda_tile.reshape

*Reshape a tile*

```
cuda_tile.reshape %source %new_shape
```

#### Parameters

* **source** (tile<Any>) - The source tile. 13.1
* **new_shape** (Array<i64>) - The new shape. 13.1

#### Results

* **result** (tile<Any>) - The reshaped tile. 13.1

#### Description

The `reshape` operation reshapes a tile to a new shape without changing the underlying data.

The total number of elements must remain the same.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.11.3. cuda_tile.slice

*Slice a tile*

```
cuda_tile.slice %source %starts %ends %steps
```

#### Parameters

* **source** (tile<Any>) - The source tile. 13.1
* **starts** (Array<i64>) - The start indices for each dimension. 13.1
* **ends** (Array<i64>) - The end indices for each dimension. 13.1
* **steps** (Array<i64>) - The step sizes for each dimension. 13.1

#### Results

* **result** (tile<Any>) - The sliced tile. 13.1

#### Description

The `slice` operation extracts a slice from a tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.11.4. cuda_tile.expand_dims

*Expand dimensions of a tile*

```
cuda_tile.expand_dims %source %axes
```

#### Parameters

* **source** (tile<Any>) - The source tile. 13.1
* **axes** (Array<i64>) - The axes to expand. 13.1

#### Results

* **result** (tile<Any>) - The tile with expanded dimensions. 13.1

#### Description

The `expand_dims` operation inserts new dimensions of size 1 into the tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.11.5. cuda_tile.squeeze

*Remove singleton dimensions from a tile*

```
cuda_tile.squeeze %source %axes
```

#### Parameters

* **source** (tile<Any>) - The source tile. 13.1
* **axes** (Array<i64>) - The axes to squeeze. 13.1

#### Results

* **result** (tile<Any>) - The tile with squeezed dimensions. 13.1

#### Description

The `squeeze` operation removes dimensions of size 1 from the tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.11.6. cuda_tile.transpose

*Transpose a tile*

```
cuda_tile.transpose %source %permutation
```

#### Parameters

* **source** (tile<Any>) - The source tile. 13.1
* **permutation** (Array<i64>) - The permutation of dimensions. 13.1

#### Results

* **result** (tile<Any>) - The transposed tile. 13.1

#### Description

The `transpose` operation permutes the dimensions of a tile.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.11.7. cuda_tile.broadcast

*Broadcast a tile to a new shape*

```
cuda_tile.broadcast %source %target_shape
```

#### Parameters

* **source** (tile<Any>) - The source tile. 13.1
* **target_shape** (Array<i64>) - The target shape. 13.1

#### Results

* **result** (tile<Any>) - The broadcasted tile. 13.1

#### Description

The `broadcast` operation broadcasts a tile to a new shape.

Dimensions of size 1 in the source are replicated to match the target shape.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.11.8. cuda_tile.concatenate

*Concatenate tiles*

```
cuda_tile.concatenate %tiles %axis
```

#### Parameters

* **tiles** (Variadic<tile<Any>>) - The tiles to concatenate. 13.1
* **axis** (i64) - The axis to concatenate along. 13.1

#### Results

* **result** (tile<Any>) - The concatenated tile. 13.1

#### Description

The `concatenate` operation concatenates multiple tiles along a specified axis.

All tiles must have the same shape except for the concatenation axis.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

### 8.11.9. cuda_tile.split

*Split a tile*

```
cuda_tile.split %source %axis %num_splits
```

#### Parameters

* **source** (tile<Any>) - The source tile. 13.1
* **axis** (i64) - The axis to split along. 13.1
* **num_splits** (i64) - The number of splits. 13.1

#### Results

* **results** (Variadic<tile<Any>>) - The split tiles. 13.1

#### Description

The `split` operation splits a tile into multiple tiles along a specified axis.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.


## 8.12. Miscellaneous

Miscellaneous operations that do not fit into the other categories.

### 8.12.1. cuda_tile.assert

*Assert a condition*

```
cuda_tile.assert %condition %message
```

#### Parameters

* **condition** (tile<i1>) - The condition to assert. 13.1
* **message** (String) - The message to display on failure. 13.1

#### Description

The `assert` operation checks that a condition is true and fails if it is not.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation may have side effects (assertion failure).

### 8.12.2. cuda_tile.barrier

*Synchronization barrier*

```
cuda_tile.barrier
```

#### Description

The `barrier` operation synchronizes all threads in a tile block.

All threads must reach the barrier before any can continue.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation may have side effects (synchronization).

### 8.12.3. cuda_tile.unreachable

*Mark unreachable code*

```
cuda_tile.unreachable
```

#### Description

The `unreachable` operation marks a point in the code that should never be reached.

This is typically used after a return or branch that exits the function.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation is pure and does not perform any memory side effects.

### 8.12.4. cuda_tile.print

*Print values for debugging*

```
cuda_tile.print %values
```

#### Parameters

* **values** (Variadic<Any>) - The values to print. 13.1

#### Description

The `print` operation prints values for debugging purposes.

#### Constraints

* The operation is not speculatable.
* The operation may NOT be speculatively executed.
* The operation has side effects (I/O).

### 8.12.5. cuda_tile.comment

*Add a comment*

```
cuda_tile.comment %text
```

#### Parameters

* **text** (String) - The comment text. 13.1

#### Description

The `comment` operation adds a comment to the IR for documentation purposes.

This operation has no effect on execution.

#### Constraints

* The operation is conditionally speculatable based on the specific operands and attributes.
* The operation may be speculatively executed without side effects.
* The operation is pure and does not perform any memory side effects.

---

## Summary

This document provides a comprehensive reference for CUDA Tile IR operations, including:

1. **Meta Types** - Core type system concepts
2. **Operation Design Considerations** - Guidelines for operation design
3. **Core Operations** - Fundamental operations like broadcast, reduce, reshape, etc.
4. **Conversions** - Type conversion operations
5. **Control Flow** - Branching and control flow operations
6. **Memory** - Memory access operations
7. **Floating Point** - Floating-point arithmetic operations
8. **Integer** - Integer arithmetic operations
9. **Bitwise** - Bit manipulation operations
10. **Atomics** - Atomic memory operations
11. **Views** - Tensor view operations
12. **Miscellaneous** - Additional utility operations

For more information, refer to the [NVIDIA CUDA Tile IR Documentation](https://docs.nvidia.com/cuda/tile-ir/).
