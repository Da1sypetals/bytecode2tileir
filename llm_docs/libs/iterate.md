# Iterating Through ndarray Arrays

This document covers three key iteration methods in ndarray for different use cases.

## `indexed_iter()` - Iterate with Indices

Iterate over elements while also getting their indices. Useful when you need to know *where* each element is in the array.

```rust
use ndarray::Array2;

let a = Array2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();

// Returns (index, element) pairs
for (idx, elem) in a.indexed_iter() {
    println!("{:?}: {}", idx, elem);
}

// Output:
// (0, 0): 1
// (0, 1): 2
// (0, 2): 3
// (1, 0): 4
// (1, 1): 5
// (1, 2): 6
```

The index follows row-major order (rightmost index varies fastest).

**Mutable version:** `.indexed_iter_mut()` for modifying elements in place with index awareness.

---

## `outer_iter()` - Iterate Over First Axis

Iterate over subviews along the **first axis** (axis 0). This reduces dimensionality by one.

For a 2D array, this gives you **rows**. For a 3D array, this gives you 2D slices.

```rust
use ndarray::Array2;

let a = Array2::from_shape_vec((3, 4), vec![
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12
]).unwrap();

// Each subview is 1D (a row in this case)
for row in a.outer_iter() {
    println!("{:?}", row);
}

// Output:
// [1, 2, 3, 4]
// [5, 6, 7, 8]
// [9, 10, 11, 12]
```

**When to use:** When you want to process each "slice" along the first axis independently.

**Mutable version:** `.outer_iter_mut()` for mutable access to subviews.

---

## `axis_iter()` - Iterate Over Any Axis

Like `outer_iter()`, but lets you specify **which axis** to iterate over. This is the more general form.

```rust
use ndarray::{Array2, Axis};

let a = Array2::from_shape_vec((3, 4), vec![
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12
]).unwrap();

// Iterate over rows (axis 0) - same as outer_iter()
for row in a.axis_iter(Axis(0)) {
    println!("row: {:?}", row);
}

// Iterate over columns (axis 1)
for col in a.axis_iter(Axis(1)) {
    println!("col: {:?}", col);
}

// Output for columns:
// col: [1, 5, 9]
// col: [2, 6, 10]
// col: [3, 7, 11]
// col: [4, 8, 12]
```

**When to use:** When you need to iterate over an axis other than the first one, or when you want explicit axis specification for clarity.

**Mutable version:** `.axis_iter_mut()` for mutable access to subviews along any axis.

---

## Quick Comparison

| Method | What it yields | Dimensionality | Use case |
|--------|----------------|----------------|----------|
| `indexed_iter()` | `(index, &element)` | Same as array | Need position + value |
| `outer_iter()` | Subviews along axis 0 | n-1 D | First-axis traversal |
| `axis_iter(Axis(n))` | Subviews along any axis | n-1 D | General axis traversal |

**Note:** Both `outer_iter()` and `axis_iter()` are **producers** in ndarray's terminology, meaning they can be used with `Zip` for parallel or multi-array operations.
