
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
