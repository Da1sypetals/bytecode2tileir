## 8.8. Integer

**Tile IR** contains a set of typed arithmetic operations which implement familiar arithmetic operations on tiles of integers, for floating-point operations see [Floating Point](#op-group-floating-point).

All operations are implemented in a manner that is efficient for the target architecture and device family. In most common cases this means utilizing the underlying hardware's native floating-point operations. Due to **Tile IR**'s stability guarantees and higher-level programming model some types on some hardware may be emulated, see [Stability](stability.html#section-stability) for more information about the stability guarantees and information about per device behavior.

### 8.8.1. Integer Arithmetic

Integer types in **Tile IR** are signless, which is importantly not the same as unsigned. We store all integers in a two's complement representation and with required operations supporting a `signed` or `unsigned` flag as needed. This design allows us to not have to differentiate between signed and unsigned integer types at the IR level and keeps sign information local to the operation.

For the `i1` type, unsigned operations see values 0/1, while signed operations see values 0/-1, with all i1 values canonicalized to 0x00 (false) or 0x01 (true) for consistent LSB-only semantics.

### 8.8.2. cuda_tile.absi

*Element-wise integer absolute value*

```
cuda_tile.absi %source
```

#### Parameters

- **source** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The input integer tile. 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The absolute value of the input tile. 13.1

#### Description

The `absi` operation computes the absolute value of the input integer tile.

The input tile is always interpreted as a signed integer.
The output tile is always interpreted as an unsigned integer.

```
absi(x) = |x|
```

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same shape.
- `source` and `result` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.

### 8.8.3. cuda_tile.addi

*Element-wise integer addition*

```
cuda_tile.addi %lhs %rhs %overflow
```

#### Parameters

- **lhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The left hand side operand. 13.1
- **rhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The right hand side operand. 13.1
- **overflow** ([IntegerOverflow](#op-attribute-cuda-tile-addi-integeroverflow-attr)) - The overflow behavior of the operation. 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The sum of the input tiles. 13.1

#### Description

The `addi` operation computes the element-wise addition of two tiles with integer element types.

```
addi(x, y)i = xi + yi
```

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

- `none` - The compiler makes no assumptions regarding overflow behavior.
- `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
- `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
- `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.

### 8.8.4. cuda_tile.cmpi

*Element-wise integer comparison*

```
cuda_tile.cmpi %comparison_predicate %lhs %rhs %signedness
```

#### Parameters

- **comparison_predicate** ([ComparisonPredicate](#op-attribute-cuda-tile-cmpi-comparisonpredicate-attr)) - The comparison predicate. 13.1
- **lhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The left hand side operand. 13.1
- **rhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The right hand side operand. 13.1
- **signedness** ([Signedness](#op-attribute-cuda-tile-cmpi-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### Results

- **result** (`tile`<`i1`>) - The result of the comparison. 13.1

#### Description

The `cmpi` operation is a generic comparison for integer-like types. The operands must have the same shape and type, and this type must be an integer type. The result type has i1 element type and the same shape as the operands.

The result is `1` if the comparison is true and `0` otherwise. The comparison is performed element-wise and the element of the result indicates whether the comparison is true for the operand elements with the same indices as those of the result.

The `comparison_predicate` attribute specifies the kind of comparison to be performed.

- `equal` - Equal comparison.
- `not_equal` - Not equal comparison.
- `less_than` - Less than comparison.
- `less_than_or_equal` - Less than or equal comparison.
- `greater_than` - Greater than comparison.
- `greater_than_or_equal` - Greater than or equal comparison.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs` and `rhs` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- Result type has i1 element type and same shape as operands
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
%lhs0 = constant dense<0> : tile<i16>
%rhs0 = constant dense<0> : tile<i16>
// Scalar "signed less than" comparison.
%x0 = cmpi less_than %lhs0, %rhs0, signed : tile<i16>
%lhs1 = constant dense<0> : tile<2x2xi64>
%rhs1 = constant dense<0> : tile<2x2xi64>
// Tile equality comparison.
// There is no difference between "signed" and "unsigned" when performing equality and inequality comparison.
%x1 = cmpi equal %lhs1, %rhs1, signed : tile<2x2xi64>
```

See [cuda_tile.cmpi_0](appendix.html#example-cuda-tile-cmpi-0) for the full example listing.

### 8.8.5. cuda_tile.divi

*Element-wise integer division*

```
cuda_tile.divi %lhs %rhs %signedness %rounding
```

#### Parameters

- **lhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The left hand side operand. 13.1
- **rhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The right hand side operand. 13.1
- **signedness** ([Signedness](#op-attribute-cuda-tile-divi-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1
- **rounding** ([RoundingMode](#op-attribute-cuda-tile-divi-roundingmode-attr)) - Set the rounding direction (implements floordiv/ceildiv). 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The result of the division. 13.1

#### Description

The `divi` operation computes the element-wise division of two tile values with integer element type.

The default rounding is towards zero. The rounding mode can be set to `positive_inf` ("ceil div"), or `negative_inf` ("floor div"), other values are illegal.

The use of the rounding flag `negative_inf` with `unsigned` is not a valid combination.

If the `unsigned` flag is provided, the operands are treated as unsigned integers, otherwise they are treated as signed integers.

The behavior is undefined if the right hand side is zero. A signed division overflow (minimum value divided by -1) is undefined behavior.

```
div(lhs, rhs)i = lhsi / rhsi
```

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

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

- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.

### 8.8.6. cuda_tile.maxi

*Element-wise integer maximum*

```
cuda_tile.maxi %lhs %rhs %signedness
```

#### Parameters

- **lhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The left hand side operand. 13.1
- **rhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The right hand side operand. 13.1
- **signedness** ([Signedness](#op-attribute-cuda-tile-maxi-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The result of the maxi operation. 13.1

#### Description

The `maxi` operation computes the element-wise maximum between two input integer tiles.

```
maxi(x, y)i = { xi if xi ≥ yi
                yi if xi < yi
```

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
// Create tensor view from a pointer to global memory
%0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
%1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
// Convert tensor views to partition views and load tiles from them.
%p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
%p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
%c0 = constant dense<0> : tile<i32>
%2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
%3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
// Signless i32 treated as unsigned
%4 = maxi %2, %3 unsigned : tile<2x4xi32>
// Signless i32 treated as signed
%5 = maxi %2, %3 signed : tile<2x4xi32>
```

See [cuda_tile.maxi_0](appendix.html#example-cuda-tile-maxi-0) for the full example listing.

### 8.8.7. cuda_tile.mini

*Element-wise integer minimum*

```
cuda_tile.mini %lhs %rhs %signedness
```

#### Parameters

- **lhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The left hand side operand. 13.1
- **rhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The right hand side operand. 13.1
- **signedness** ([Signedness](#op-attribute-cuda-tile-mini-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The minimum of the input tiles. 13.1

#### Description

The `mini` operation computes the element-wise minimum between the two input tiles with integer element types.

```
mini(x, y)i = { xi if xi ≤ yi
                yi if xi > yi
```

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
// Create tensor view from a pointer to global memory
%0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
%1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
// Convert tensor views to partition views and load tiles from partition views.
%p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
%p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
%c0 = constant dense<0> : tile<i32>
%2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
%3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
// Signless i32 treated as unsigned
%4 = mini %2, %3 unsigned : tile<2x4xi32>
// Signless i32 treated as signed
%5 = mini %2, %3 signed : tile<2x4xi32>
```

See [cuda_tile.mini_0](appendix.html#example-cuda-tile-mini-0) for the full example listing.

### 8.8.8. cuda_tile.mmai

*Integer matrix-multiply-accumulate*

```
cuda_tile.mmai %lhs %rhs %acc %signedness_lhs %signedness_rhs
```

#### Parameters

- **lhs** (`tile`<`i8`>) - The left hand side matrix operand. 13.1
- **rhs** (`tile`<`i8`>) - The right hand side matrix operand. 13.1
- **acc** (`tile`<`i32`>) - The accumulator matrix operand. 13.1
- **signedness_lhs** ([Signedness](#op-attribute-cuda-tile-mmai-signedness-attr)) - The signedness of the `lhs` operand. 13.1
- **signedness_rhs** ([Signedness](#op-attribute-cuda-tile-mmai-signedness-attr)) - The signedness of the `rhs` operand. 13.1

#### Results

- **result** (`tile`<`i32`>) - The result matrix after multiplication and accumulation. 13.1

#### Description

The `mmai` operation implements an MMA (matrix-multiply-accumulate) operation for integer tiles. It performs matrix multiplication on the integer tiles `lhs` and `rhs`, then adds the tile `acc` to the result. `lhs`, `rhs`, and `acc` must be 2D tiles or 3D tiles. The latter case indicates a batched matrix multiplication.

Input tiles `lhs` and `rhs` must be of integer type `i8`. The signedness of `lhs` and `rhs` are specified separately by the `signedness_lhs` and `signedness_rhs` attributes, respectively. The accumulator tile `acc` must be of type `i32` and is always interpreted as signed. The output tile `result` is of type `i32` and is always interpreted as signed.

Shapes must be a valid matrix multiplication configuration. Unbatched (2D) MMA expects the operands `lhs`, `rhs`, and `acc` to have shapes `M x K`, `K x N`, and `M x N` (respectively). Batched (3D) MMA expects the operands to have shapes `B x M x K`, `B x K x N`, and `B x M x N` (respectively).

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `acc` and `result` must have the same shape and element type (`tile`<`i32`>).
- `lhs` and `rhs` must have the same element type (`tile`<`i8`>).
- `lhs`, `rhs` and `acc` must have the same rank.
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
%lhs0 = cuda_tile.constant <i8: 0> : tile<4x8xi8>
%rhs0 = cuda_tile.constant <i8: 0> : tile<8x2xi8>
%acc0 = cuda_tile.constant <i32: 0> : tile<4x2xi32>
%0 = mmai %lhs0, %rhs0, %acc0 signed signed
    : tile<4x8xi8>, tile<8x2xi8>,
      tile<4x2xi32>
%lhs1 = cuda_tile.constant <i8: 0> : tile<2x4x8xi8>
%rhs1 = cuda_tile.constant <i8: 0> : tile<2x8x2xi8>
%acc1 = cuda_tile.constant <i32: 0> : tile<2x4x2xi32>
%1 = mmai %lhs1, %rhs1, %acc1 unsigned unsigned
    : tile<2x4x8xi8>, tile<2x8x2xi8>,
      tile<2x4x2xi32>
```

See [cuda_tile.mmai_0](appendix.html#example-cuda-tile-mmai-0) for the full example listing.

### 8.8.9. cuda_tile.muli

*Element-wise integer multiplication*

```
cuda_tile.muli %lhs %rhs %overflow
```

#### Parameters

- **lhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The left hand side input integer tile. 13.1
- **rhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The right hand side input integer tile. 13.1
- **overflow** ([IntegerOverflow](#op-attribute-cuda-tile-muli-integeroverflow-attr)) - The overflow behavior of the operation. 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The product of the input tiles. 13.1

#### Description

The `muli` operation computes the element-wise product between the two input tiles with integer element types.

```
muli(x, y)i = xi × yi
```

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

- `none` - The compiler makes no assumptions regarding overflow behavior.
- `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
- `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
- `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.

### 8.8.10. cuda_tile.mulhii

*Element-wise high bits of integer multiplication*

```
cuda_tile.mulhii %x %y
```

#### Parameters

- **x** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The left hand side operand. 13.1
- **y** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The right hand side operand. 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The result of the mulhii operation. 13.1

#### Description

The `mulhii` operation produces the most significant N bits of the 2N-bit product of two N-bit integer tiles. For `i64`, this is the most significant 64 bits of the full 128-bit product; for `i8`, it is the most significant 8 bits of the full 16-bit product; etc.

This is in contrast to `muli`, which produces the lower N bits of the 2N-bit product.

The `mulhii` operation is only defined for unsigned integers.

```
mulhii(xi, yi) = xi × yi >> bitwidth(type(xi))
```

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `x`, `y` and `result` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
// 2^31 * 2 = 2^32, or 0x100000000.
// The most significant 32 bits of the product are 0x00000001.
// The lower 32 bits of the product are 0x00000000.
%a = constant dense<2147483648> : tile<i32>  // %a = 2^31
%b = constant dense<2> : tile<i32>           // %b = 2
%res_hi = mulhii %a, %b : tile<i32>          // %res_hi = 1
%res_lo = muli %a, %b : tile<i32>            // %res_lo = 0
```

See [cuda_tile.mulhii_0](appendix.html#example-cuda-tile-mulhii-0) for the full example listing.

### 8.8.11. cuda_tile.negi

*Element-wise integer negation*

```
cuda_tile.negi %source
```

#### Parameters

- **source** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The input integer tile. 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The negated integer tile. 13.1

#### Description

The `negi` operation computes the element-wise negation of the input integer tile. The input and output tiles are always interpreted as signed integers.

```
negi(xi) = −xi
```

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `source` and `result` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
%source = constant dense<[0, 1, 2, 3]> : tile<4xi16>
%result = negi %source : tile<4xi16>
// %result = [0, -1, -2, -3]
```

See [cuda_tile.negi_0](appendix.html#example-cuda-tile-negi-0) for the full example listing.

### 8.8.12. cuda_tile.remi

*Element-wise integer remainder*

```
cuda_tile.remi %lhs %rhs %signedness
```

#### Parameters

- **lhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The left hand side operand. 13.1
- **rhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The right hand side operand. 13.1
- **signedness** ([Signedness](#op-attribute-cuda-tile-remi-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The remainder after division. 13.1

#### Description

The `remi` operation computes the element-wise remainder of the input tiles with integer element types using truncated division (rounding towards zero). Division by zero is undefined behavior.

```
remi(x, y)i = xi − trunc(xi / yi) × yi
```

If the operation is signed, the sign of the result matches the sign of the dividend (`lhs`). For example:

- `remi(7, 3) = 1`
- `remi(7, -3) = 1`
- `remi(-7, 3) = -1`
- `remi(-7, -3) = -1`

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `result`, `lhs` and `rhs` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.

### 8.8.13. cuda_tile.shli

*Element-wise shift-left*

```
cuda_tile.shli %lhs %rhs %overflow
```

#### Parameters

- **lhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The left hand side operand. 13.1
- **rhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The right hand side operand (shift amount). 13.1
- **overflow** ([IntegerOverflow](#op-attribute-cuda-tile-shli-integeroverflow-attr)) - The overflow behavior of the operation. 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The result of the left shift operation. 13.1

#### Description

The `shli` operation computes the element-wise left shift of the `lhs` integer operand by the `rhs` operand. The lower-order bits on the right are filled with zeros.

The `rhs` operand is interpreted as an unsigned integer.

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

- `none` - The compiler makes no assumptions regarding overflow behavior.
- `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
- `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
- `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.

### 8.8.14. cuda_tile.shri

*Element-wise shift-right*

```
cuda_tile.shri %lhs %rhs %signedness
```

#### Parameters

- **lhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The left hand side operand. 13.1
- **rhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The right hand side operand (shift amount). 13.1
- **signedness** ([Signedness](#op-attribute-cuda-tile-shri-signedness-attr)) - Interpret integer(s) as `signed` or `unsigned` 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The result of the right shift operation. 13.1

#### Description

The `shri` operation computes the element-wise right shift of the `lhs` integer operand by the value of the `rhs` operand for tiles with integer element types.

When `unsigned`, higher-order bits are zero-filled; when `signed`, the higher-order bits are filled with the sign bit.

The `rhs` operand is always interpreted as an unsigned integer.

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

The `signedness` attribute specifies the signedness of operand(s).

- `unsigned` - Treat the operands as unsigned integers.
- `signed` - Treat the operands as signed integers.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.

### 8.8.15. cuda_tile.subi

*Element-wise integer subtraction*

```
cuda_tile.subi %lhs %rhs %overflow
```

#### Parameters

- **lhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The left hand side operand. 13.1
- **rhs** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The right hand side operand. 13.1
- **overflow** ([IntegerOverflow](#op-attribute-cuda-tile-subi-integeroverflow-attr)) - The overflow behavior of the operation. 13.1

#### Results

- **result** (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>) - The result of the subtraction. 13.1

#### Description

The `subi` operation computes the element-wise subtraction of two input integer tiles.

```
subi(x, y)i = xi − yi
```

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See [Integer](#op-group-integer) for more details.

The `overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution.

- `none` - The compiler makes no assumptions regarding overflow behavior.
- `no_signed_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.
- `no_unsigned_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.
- `no_wrap` - The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (`tile`<`i1` | `i8` | `i16` | `i32` | `i64`>).
- The operation's result type may be inferred from its operands and attributes.