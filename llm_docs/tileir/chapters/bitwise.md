## 8.9. Bitwise

### 8.9.1. cuda_tile.andi

*Element-wise bitwise logical AND*

```
cuda_tile.andi %lhs %rhs
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand. 13.1
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand. 13.1

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The bitwise AND of the input tiles. 13.1

#### Description

The `andi` operation produces a value that is the result of an element-wise, bitwise "and" of two tiles with integer element type.

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See Integer for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (tile<i1 | i8 | i16 | i32 | i64>).
- The operation's result type may be inferred from its operands and attributes.

### 8.9.2. cuda_tile.ori

*Element-wise bitwise OR*

```
cuda_tile.ori %lhs %rhs
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand. 13.1
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand. 13.1

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The bitwise OR of the input tiles. 13.1

#### Description

The `ori` operation computes the element-wise bitwise OR of two tiles with integer element types.

$$ \text{ori}(x, y)_i = x_i | y_i $$

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See Integer for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (tile<i1 | i8 | i16 | i32 | i64>).
- The operation's result type may be inferred from its operands and attributes.

### 8.9.3. cuda_tile.xori

*Element-wise bitwise XOR*

```
cuda_tile.xori %lhs %rhs
```

#### Parameters

- **lhs** (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand. 13.1
- **rhs** (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand. 13.1

#### Results

- **result** (tile<i1 | i8 | i16 | i32 | i64>) - The bitwise XOR of the input tiles. 13.1

#### Description

The `xori` operation computes the element-wise bitwise exclusive or (XOR) of two tile values with integer element types.

$$ \text{xori}(x, y)_i = x_i \oplus y_i $$

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow. See Integer for more details.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- `lhs`, `rhs` and `result` must have the same shape and element type (tile<i1 | i8 | i16 | i32 | i64>).
- The operation's result type may be inferred from its operands and attributes.

#### Examples

```mlir
%lhs = constant dense<[0, 1, 2, 3]> : tile<4xi32>
%rhs = constant dense<[4, 5, 6, 7]> : tile<4xi32>
// This computes the bitwise XOR of each element in `%lhs` and `%rhs`, which
// are tiles of shape `4xi32`, and returns the result as `%result`.
%result = xori %lhs, %rhs : tile<4xi32>
```

See cuda_tile.xori_0 for the full example listing.