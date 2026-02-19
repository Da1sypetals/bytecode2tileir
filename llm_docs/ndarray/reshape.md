# Reshape Methods in ndarray

## `into_shape_with_order`

Transforms the array into a new shape, consuming the original array. It requires the source array to be **contiguous** and only succeeds if the memory layout is compatible with the specified order.

```rust
use ndarray::{aview1, aview2};
use ndarray::Order;

// Row major (C order) - default
assert!(
    aview1(&[1., 2., 3., 4.]).into_shape_with_order((2, 2)).unwrap()
    == aview2(&[[1., 2.],
                [3., 4.]])
);

// Column major (F order)
assert!(
    aview1(&[1., 2., 3., 4.]).into_shape_with_order(((2, 2), Order::ColumnMajor)).unwrap()
    == aview2(&[[1., 3.],
                [2., 4.]])
);
```

**Key points:**
- Consumes `self`
- Requires contiguous memory layout
- Returns `Result<Array, ShapeError>`
- Use when you own the array and want guaranteed zero-copy reshaping
- Errors if shapes don't have same number of elements
- Errors if RowMajor requested but input is not c-contiguous
- Errors if ColumnMajor requested but input is not f-contiguous

## `to_shape`

Transforms the array into a new shape, borrowing the original. Returns a `CowArray` (view if possible, owned copy if needed). More flexible than `into_shape_with_order`.

```rust
use ndarray::array;
use ndarray::Order;

// Row major
assert!(
    array![1., 2., 3., 4., 5., 6.].to_shape(((2, 3), Order::RowMajor)).unwrap()
    == array![[1., 2., 3.],
              [4., 5., 6.]]
);

// Column major
assert!(
    array![1., 2., 3., 4., 5., 6.].to_shape(((2, 3), Order::ColumnMajor)).unwrap()
    == array![[1., 3., 5.],
              [2., 4., 6.]]
);
```

**Key points:**
- Borrows `&self`
- Returns `Result<CowArray, ShapeError>`
- Will copy elements if necessary to achieve the requested ordering
- Works with views and produces views when possible
- More flexible, but may have overhead from copying

## Shape Parameter Format

Both methods accept the shape parameter in these formats:
- `(3, 4)` - Shape with default RowMajor order
- `((3, 4), Order::RowMajor)` - Explicit RowMajor order
- `((3, 4), Order::ColumnMajor)` - Explicit ColumnMajor order
- `((3, 4), Order::C)` or `Order::F` - Shorthand notation (C/F instead of RowMajor/ColumnMajor)

## Comparison

| Feature | `into_shape_with_order` | `to_shape` |
|---------|------------------------|------------|
| Ownership | Consumes array | Borrows array |
| Works with views | Yes | Yes |
| Copying behavior | Never copies (errors if not possible) | Copies if needed |
| Return type | `Result<Array, ShapeError>` | `Result<CowArray, ShapeError>` |
| Use case | When you own the array and want guaranteed zero-copy | When you need flexibility or don't own the array |

## When to use which

- **Use `into_shape_with_order`** when:
  - You own the array
  - You want guaranteed zero-copy operation
  - You know the array is contiguous
  - Performance is critical

- **Use `to_shape`** when:
  - You need to borrow the array
  - You're working with views
  - You're unsure about memory layout
  - You want automatic copying as fallback
  - Flexibility is preferred over strict performance guarantees
