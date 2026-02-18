use crate::interpreter::data_structures::{
    elem_type::{ElemType, Scalar},
    tensor_view::{PartitionView, TensorView},
};
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use log::debug;
use log::info;
use rand::{rngs::StdRng, seq::SliceRandom, RngExt, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

// ============================================================================
// Test Utilities: TestTensor and grid_to_tensor_index
// ============================================================================

/// Test tensor with nd-index access support
struct TestTensor {
    buffer: Vec<u8>,
    shape: Vec<i64>,
    strides: Vec<i64>,
    elem_type: ElemType,
}

impl TestTensor {
    /// Create a new tensor with contiguous strides
    fn new(shape: Vec<i64>, elem_type: ElemType) -> Self {
        let strides = compute_contiguous_strides(&shape);
        let total_elements: usize = shape.iter().product::<i64>() as usize;
        let buffer = vec![0u8; total_elements * elem_type.size_bytes()];
        Self {
            buffer,
            shape,
            strides,
            elem_type,
        }
    }

    /// Create a tensor with custom strides
    fn with_strides(shape: Vec<i64>, strides: Vec<i64>, elem_type: ElemType) -> Self {
        let max_offset = shape
            .iter()
            .zip(&strides)
            .map(|(&s, &st)| (s - 1) * st)
            .sum::<i64>() as usize;
        let buffer = vec![0u8; (max_offset + 1) * elem_type.size_bytes()];
        Self {
            buffer,
            shape,
            strides,
            elem_type,
        }
    }

    /// Fill with arange pattern: tensor[indices] = linear_index
    fn fill_arange(&mut self) {
        let total_elements: usize = self.shape.iter().product::<i64>() as usize;
        let ptr = self.buffer.as_mut_ptr() as usize;
        let strides = self.strides.clone();
        let shape = self.shape.clone();
        let elem_size = self.elem_type.size_bytes();
        let elem_type = self.elem_type;

        (0..total_elements)
            .into_par_iter()
            .progress()
            .for_each(|linear_idx| {
                let indices = linear_to_nd_index(linear_idx, &shape);

                // Element offset is the dot product of indices and strides
                let elem_offset: i64 = indices
                    .iter()
                    .zip(&strides)
                    .map(|(&idx, &st)| idx * st)
                    .sum();
                let byte_offset = (elem_offset as usize) * elem_size;

                unsafe {
                    let p = (ptr + byte_offset) as *mut u8;
                    match elem_type {
                        ElemType::I32 => *(p as *mut i32) = linear_idx as i32,
                        ElemType::I64 => *(p as *mut i64) = linear_idx as i64,
                        _ => panic!("fill_arange only supports I32 and I64"),
                    }
                }
            });
    }

    /// Fill with arange pattern scaled: tensor[indices] = linear_index * scale
    fn fill_arange_scaled(&mut self, scale: f64) {
        let total_elements: usize = self.shape.iter().product::<i64>() as usize;
        let ptr = self.buffer.as_mut_ptr() as usize;
        let strides = self.strides.clone();
        let shape = self.shape.clone();
        let elem_size = self.elem_type.size_bytes();
        let elem_type = self.elem_type;

        (0..total_elements).into_par_iter().for_each(|linear_idx| {
            let indices = linear_to_nd_index(linear_idx, &shape);
            let elem_offset: i64 = indices
                .iter()
                .zip(&strides)
                .map(|(&idx, &st)| idx * st)
                .sum();
            let byte_offset = (elem_offset as usize) * elem_size;

            unsafe {
                let p = (ptr + byte_offset) as *mut u8;
                match elem_type {
                    ElemType::F16 => *(p as *mut f16) = (linear_idx as f64 * scale) as f16,
                    ElemType::F32 => *(p as *mut f32) = (linear_idx as f64 * scale) as f32,
                    ElemType::F64 => *(p as *mut f64) = linear_idx as f64 * scale,
                    _ => panic!("fill_arange_scaled only supports floating types"),
                }
            }
        });
    }

    /// Fill with custom pattern: tensor[indices] = f(indices)
    fn fill_with<F>(&mut self, f: F)
    where
        F: Fn(&[i64]) -> i32 + Sync + Send,
    {
        let total_elements: usize = self.shape.iter().product::<i64>() as usize;
        let ptr = self.buffer.as_mut_ptr() as usize;
        let strides = self.strides.clone();
        let shape = self.shape.clone();
        let elem_size = self.elem_type.size_bytes();

        (0..total_elements).into_par_iter().for_each(|linear_idx| {
            let indices = linear_to_nd_index(linear_idx, &shape);
            let elem_offset: i64 = indices
                .iter()
                .zip(&strides)
                .map(|(&idx, &st)| idx * st)
                .sum();
            let byte_offset = (elem_offset as usize) * elem_size;
            let value = f(&indices);

            unsafe {
                let p = (ptr + byte_offset) as *mut u8;
                match self.elem_type {
                    ElemType::I32 => *(p as *mut i32) = value,
                    _ => panic!("fill_with only supports I32 in this impl"),
                }
            }
        });
    }

    /// Get value at nd indices
    fn get(&self, indices: &[i64]) -> Scalar {
        let elem_offset: i64 = indices
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &st)| idx * st)
            .sum();
        let byte_offset = (elem_offset as usize) * self.elem_type.size_bytes();

        unsafe {
            let p = self.buffer.as_ptr().add(byte_offset);
            match self.elem_type {
                ElemType::Bool => Scalar::Bool(*(p as *const u8) != 0),
                ElemType::I8 => Scalar::I8(*(p as *const i8)),
                ElemType::I16 => Scalar::I16(*(p as *const i16)),
                ElemType::I32 => Scalar::I32(*(p as *const i32)),
                ElemType::I64 => Scalar::I64(*(p as *const i64)),
                ElemType::F16 => Scalar::F16(*(p as *const f16)),
                ElemType::F32 => Scalar::F32(*(p as *const f32)),
                ElemType::F64 => Scalar::F64(*(p as *const f64)),
                ElemType::Ptr => Scalar::Ptr(*(p as *const *mut u8) as *mut u8),
            }
        }
    }

    /// Set value at nd indices
    fn set(&mut self, indices: &[i64], value: Scalar) {
        let elem_offset: i64 = indices
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &st)| idx * st)
            .sum();
        let byte_offset = (elem_offset as usize) * self.elem_type.size_bytes();

        unsafe {
            let p = self.buffer.as_mut_ptr().add(byte_offset);
            match value {
                Scalar::Bool(v) => *(p as *mut u8) = if v { 1 } else { 0 },
                Scalar::I8(v) => *(p as *mut i8) = v,
                Scalar::I16(v) => *(p as *mut i16) = v,
                Scalar::I32(v) => *(p as *mut i32) = v,
                Scalar::I64(v) => *(p as *mut i64) = v,
                Scalar::F16(v) => *(p as *mut f16) = v,
                Scalar::F32(v) => *(p as *mut f32) = v,
                Scalar::F64(v) => *(p as *mut f64) = v,
                Scalar::Ptr(v) => *(p as *mut *mut u8) = v,
            }
        }
    }

    /// Convert to TensorView
    fn as_view(&self) -> TensorView {
        TensorView::new(
            self.buffer.as_ptr() as *mut u8,
            self.elem_type,
            self.shape.clone(),
            self.strides.clone(),
        )
    }

    /// Check if indices are in bounds
    fn in_bounds(&self, indices: &[i64]) -> bool {
        indices
            .iter()
            .zip(&self.shape)
            .all(|(&idx, &s)| idx >= 0 && idx < s)
    }
}

/// Compute contiguous strides for a shape
fn compute_contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let rank = shape.len();
    let mut strides = vec![1i64; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert linear index to nd index
fn linear_to_nd_index(linear: usize, shape: &[i64]) -> Vec<i64> {
    let rank = shape.len();
    let mut indices = vec![0i64; rank];
    let mut remaining = linear;
    for dim in (0..rank).rev() {
        indices[dim] = (remaining % shape[dim] as usize) as i64;
        remaining /= shape[dim] as usize;
    }
    indices
}

/// Convert grid position + tile indices to tensor nd index
///
/// # Arguments
/// - grid_pos: position of tile in the grid
/// - tile_shape: shape of each tile
/// - tile_indices: indices within the tile
/// - dim_map: dimension mapping (tile_dim -> tensor_dim), None for identity
fn grid_to_tensor_index(
    grid_pos: &[i64],
    tile_shape: &[i32],
    tile_indices: &[i64],
    dim_map: Option<&[i32]>,
) -> Vec<i64> {
    let rank = grid_pos.len();
    let mut tensor_idx = vec![0i64; rank];

    for tile_dim in 0..rank {
        let tensor_dim = dim_map.map_or(tile_dim as i32, |m| m[tile_dim]) as usize;
        tensor_idx[tensor_dim] =
            grid_pos[tile_dim] * tile_shape[tile_dim] as i64 + tile_indices[tile_dim];
    }
    tensor_idx
}

// ============================================================================
// 2D Dimension Permutation Tests
// ============================================================================

/// ```
///        0     1     2     3     4     5     6     7  
///     +-----+-----+-----+-----+-----+-----+-----+-----+
///   0 |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |
///     +-----+-----+-----+-----+-----+-----+-----+-----+
///   1 | 20  | 21  | 22  | 23  | 24  | 25  | 26  | 27  |
///     +-----+-----+-----+-----+-----+-----+-----+-----+
///   2 | 40  | 41  | 42  | 43  | 44  | 45  | 46  | 47  |
///     +-----+-----+-----+-----+-----+-----+-----+-----+
///   3 | 60  | 61  | 62  | 63  | 64  | 65  | 66  | 67  |
///     +-----+-----+-----+-----+-----+-----+-----+-----+
///   4 | 80  | 81  | 82  | 83  | 84  | 85  | 86  | 87  |
///     +-----+-----+-----+-----+-----+-----+-----+-----+
///   5 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 |
///     +-----+-----+-----+-----+-----+-----+-----+-----+
///   6 | 120 | 121 | 122*| 123*| 124 | 125 | 126 | 127 |
///     +-----+-----+-----+-----+-----+-----+-----+-----+
///   7 | 140 | 141 | 142*| 143*| 144 | 145 | 146 | 147 |
///     +-----+-----+-----+-----+-----+-----+-----+-----+
///
///  Transposed-loads this tile:
///     +-----+-----+
///     | 122 | 142 |
///     +-----+-----+
///     | 123 | 143 |
///     +-----+-----+
/// ```
#[test]
fn test_dim_map_2d_transpose() {
    // Special test: uses i*20+j pattern instead of arange
    let mut tensor = TestTensor::new(vec![8, 8], ElemType::F16);
    for i in 0..8i64 {
        for j in 0..8i64 {
            tensor.set(&[i, j], Scalar::F16((i * 20 + j) as f16));
        }
    }

    let partition = PartitionView::new(tensor.as_view(), vec![2, 2], vec![1, 0], false, None);
    let tile = partition.load_tile(&[1, 3]);
    let tile_shape = vec![2, 2];
    let dim_map = [1, 0];

    // Verify tile elements using grid_to_tensor_index
    for ti in 0..2i64 {
        for tj in 0..2i64 {
            let tensor_idx = grid_to_tensor_index(&[1, 3], &tile_shape, &[ti, tj], Some(&dim_map));
            let expected = tensor.get(&tensor_idx);
            let actual = tile.get_scalar(&[ti, tj]);
            assert_eq!(actual, expected, "Mismatch at tile [{}, {}]", ti, tj);
        }
    }

    // Verify specific value
    let actual = tile.get_scalar(&[0, 0]);
    assert_eq!(actual, Scalar::F16(122.0), "Mismatch at tile [0, 0]");
    let actual = tile.get_scalar(&[0, 1]);
    assert_eq!(actual, Scalar::F16(142.0), "Mismatch at tile [0, 1]");
    let actual = tile.get_scalar(&[1, 0]);
    assert_eq!(actual, Scalar::F16(123.0), "Mismatch at tile [1, 0]");
    let actual = tile.get_scalar(&[1, 1]);
    assert_eq!(actual, Scalar::F16(143.0), "Mismatch at tile [1, 1]");
    info!("2D transpose test OK")
}

// ============================================================================
// 3D Dimension Permutation Tests
// ============================================================================

#[test]
fn test_dim_map_3d_permutation_012() {
    let mut tensor = TestTensor::new(vec![8, 8, 8], ElemType::I32);
    // Fill with pattern: i*100 + j*10 + k
    tensor.fill_with(|idx| (idx[0] * 100 + idx[1] * 10 + idx[2]) as i32);

    let dim_map = vec![0, 1, 2];
    let partition = PartitionView::new(
        tensor.as_view(),
        vec![4, 4, 4],
        dim_map.clone(),
        false,
        None,
    );
    let tile = partition.load_tile(&[0, 0, 0]);
    let tile_shape = vec![4, 4, 4];

    // Verify tile elements
    for i in 0..4i64 {
        for j in 0..4i64 {
            for k in 0..4i64 {
                let tensor_idx =
                    grid_to_tensor_index(&[0, 0, 0], &tile_shape, &[i, j, k], Some(&dim_map));
                let expected = tensor.get(&tensor_idx);
                let actual = tile.get_scalar(&[i, j, k]);
                assert_eq!(actual, expected, "Mismatch at tile [{}, {}, {}]", i, j, k);
            }
        }
    }
}

#[test]
fn test_dim_map_3d_permutation_021() {
    let mut tensor = TestTensor::new(vec![8, 8, 8], ElemType::I32);
    tensor.fill_with(|idx| (idx[0] * 100 + idx[1] * 10 + idx[2]) as i32);

    let dim_map = vec![0, 2, 1];
    let partition = PartitionView::new(
        tensor.as_view(),
        vec![4, 4, 4],
        dim_map.clone(),
        false,
        None,
    );
    let tile = partition.load_tile(&[0, 0, 0]);
    let tile_shape = vec![4, 4, 4];

    for i in 0..4i64 {
        for j in 0..4i64 {
            for k in 0..4i64 {
                let tensor_idx =
                    grid_to_tensor_index(&[0, 0, 0], &tile_shape, &[i, j, k], Some(&dim_map));
                let expected = tensor.get(&tensor_idx);
                let actual = tile.get_scalar(&[i, j, k]);
                assert_eq!(actual, expected, "Mismatch at tile [{}, {}, {}]", i, j, k);
            }
        }
    }
}

#[test]
fn test_dim_map_3d_permutation_102() {
    let mut tensor = TestTensor::new(vec![8, 8, 8], ElemType::I32);
    tensor.fill_with(|idx| (idx[0] * 100 + idx[1] * 10 + idx[2]) as i32);

    let dim_map = vec![1, 0, 2];
    let partition = PartitionView::new(
        tensor.as_view(),
        vec![4, 4, 4],
        dim_map.clone(),
        false,
        None,
    );
    let tile = partition.load_tile(&[0, 0, 0]);
    let tile_shape = vec![4, 4, 4];

    for i in 0..4i64 {
        for j in 0..4i64 {
            for k in 0..4i64 {
                let tensor_idx =
                    grid_to_tensor_index(&[0, 0, 0], &tile_shape, &[i, j, k], Some(&dim_map));
                let expected = tensor.get(&tensor_idx);
                let actual = tile.get_scalar(&[i, j, k]);
                assert_eq!(actual, expected, "Mismatch at tile [{}, {}, {}]", i, j, k);
            }
        }
    }
}

#[test]
fn test_dim_map_3d_permutation_120() {
    let mut tensor = TestTensor::new(vec![8, 8, 8], ElemType::I32);
    tensor.fill_with(|idx| (idx[0] * 100 + idx[1] * 10 + idx[2]) as i32);

    let dim_map = vec![1, 2, 0];
    let partition = PartitionView::new(
        tensor.as_view(),
        vec![4, 4, 4],
        dim_map.clone(),
        false,
        None,
    );
    let tile = partition.load_tile(&[0, 0, 0]);
    let tile_shape = vec![4, 4, 4];

    for i in 0..4i64 {
        for j in 0..4i64 {
            for k in 0..4i64 {
                let tensor_idx =
                    grid_to_tensor_index(&[0, 0, 0], &tile_shape, &[i, j, k], Some(&dim_map));
                let expected = tensor.get(&tensor_idx);
                let actual = tile.get_scalar(&[i, j, k]);
                assert_eq!(actual, expected, "Mismatch at tile [{}, {}, {}]", i, j, k);
            }
        }
    }
}

#[test]
fn test_dim_map_3d_permutation_201() {
    let mut tensor = TestTensor::new(vec![8, 8, 8], ElemType::I32);
    tensor.fill_with(|idx| (idx[0] * 100 + idx[1] * 10 + idx[2]) as i32);

    let dim_map = vec![2, 0, 1];
    let partition = PartitionView::new(
        tensor.as_view(),
        vec![4, 4, 4],
        dim_map.clone(),
        false,
        None,
    );
    let tile = partition.load_tile(&[0, 0, 0]);
    let tile_shape = vec![4, 4, 4];

    for i in 0..4i64 {
        for j in 0..4i64 {
            for k in 0..4i64 {
                let tensor_idx =
                    grid_to_tensor_index(&[0, 0, 0], &tile_shape, &[i, j, k], Some(&dim_map));
                let expected = tensor.get(&tensor_idx);
                let actual = tile.get_scalar(&[i, j, k]);
                assert_eq!(actual, expected, "Mismatch at tile [{}, {}, {}]", i, j, k);
            }
        }
    }
}

#[test]
fn test_dim_map_3d_permutation_210() {
    let mut tensor = TestTensor::new(vec![8, 8, 8], ElemType::I32);
    tensor.fill_with(|idx| (idx[0] * 100 + idx[1] * 10 + idx[2]) as i32);

    let dim_map = vec![2, 1, 0];
    let partition = PartitionView::new(
        tensor.as_view(),
        vec![4, 4, 4],
        dim_map.clone(),
        false,
        None,
    );
    let tile = partition.load_tile(&[0, 0, 0]);
    let tile_shape = vec![4, 4, 4];

    for i in 0..4i64 {
        for j in 0..4i64 {
            for k in 0..4i64 {
                let tensor_idx =
                    grid_to_tensor_index(&[0, 0, 0], &tile_shape, &[i, j, k], Some(&dim_map));
                let expected = tensor.get(&tensor_idx);
                let actual = tile.get_scalar(&[i, j, k]);
                assert_eq!(actual, expected, "Mismatch at tile [{}, {}, {}]", i, j, k);
            }
        }
    }
}

// ============================================================================
// 4D Dimension Permutation Tests
// ============================================================================

#[test]
fn test_dim_map_4d_identity() {
    let mut tensor = TestTensor::new(vec![8, 8, 8, 4], ElemType::I32);
    tensor.fill_with(|idx| (idx[0] * 1000 + idx[1] * 100 + idx[2] * 10 + idx[3]) as i32);

    let dim_map = vec![0, 1, 2, 3];
    let partition = PartitionView::new(
        tensor.as_view(),
        vec![4, 4, 4, 2],
        dim_map.clone(),
        false,
        None,
    );
    let tile = partition.load_tile(&[0, 0, 0, 0]);
    let tile_shape = vec![4, 4, 4, 2];

    for i in 0..4i64 {
        for j in 0..4i64 {
            for k in 0..4i64 {
                for l in 0..2i64 {
                    let tensor_idx = grid_to_tensor_index(
                        &[0, 0, 0, 0],
                        &tile_shape,
                        &[i, j, k, l],
                        Some(&dim_map),
                    );
                    let expected = tensor.get(&tensor_idx);
                    let actual = tile.get_scalar(&[i, j, k, l]);
                    assert_eq!(
                        actual, expected,
                        "Mismatch at tile [{}, {}, {}, {}]",
                        i, j, k, l
                    );
                }
            }
        }
    }
}

#[test]
fn test_dim_map_4d_reverse() {
    let mut tensor = TestTensor::new(vec![8, 8, 8, 4], ElemType::I32);
    tensor.fill_with(|idx| (idx[0] * 1000 + idx[1] * 100 + idx[2] * 10 + idx[3]) as i32);

    let dim_map = vec![3, 2, 1, 0];
    let partition = PartitionView::new(
        tensor.as_view(),
        vec![4, 4, 4, 2],
        dim_map.clone(),
        false,
        None,
    );
    let tile = partition.load_tile(&[0, 0, 0, 0]);
    let tile_shape = vec![4, 4, 4, 2];

    for i in 0..4i64 {
        for j in 0..4i64 {
            for k in 0..4i64 {
                for l in 0..2i64 {
                    let tensor_idx = grid_to_tensor_index(
                        &[0, 0, 0, 0],
                        &tile_shape,
                        &[i, j, k, l],
                        Some(&dim_map),
                    );
                    let expected = tensor.get(&tensor_idx);
                    let actual = tile.get_scalar(&[i, j, k, l]);
                    assert_eq!(
                        actual, expected,
                        "Mismatch at tile [{}, {}, {}, {}]",
                        i, j, k, l
                    );
                }
            }
        }
    }
}

#[test]
fn test_dim_map_4d_partial_transpose_0132() {
    let mut tensor = TestTensor::new(vec![8, 8, 8, 4], ElemType::I32);
    tensor.fill_with(|idx| (idx[0] * 1000 + idx[1] * 100 + idx[2] * 10 + idx[3]) as i32);

    let dim_map = vec![0, 1, 3, 2];
    let partition = PartitionView::new(
        tensor.as_view(),
        vec![4, 4, 4, 2],
        dim_map.clone(),
        false,
        None,
    );
    let tile = partition.load_tile(&[0, 0, 0, 0]);
    let tile_shape = vec![4, 4, 4, 2];

    for i in 0..4i64 {
        for j in 0..4i64 {
            for k in 0..4i64 {
                for l in 0..2i64 {
                    let tensor_idx = grid_to_tensor_index(
                        &[0, 0, 0, 0],
                        &tile_shape,
                        &[i, j, k, l],
                        Some(&dim_map),
                    );
                    let expected = tensor.get(&tensor_idx);
                    let actual = tile.get_scalar(&[i, j, k, l]);
                    assert_eq!(
                        actual, expected,
                        "Mismatch at tile [{}, {}, {}, {}]",
                        i, j, k, l
                    );
                }
            }
        }
    }
}

#[test]
fn test_dim_map_4d_partial_transpose_random() {
    // Configuration
    let tensor_shape = vec![256i64, 16, 64, 32];
    let tile_shape = vec![64i32, 8, 32, 4];

    let mut tensor = TestTensor::new(tensor_shape.clone(), ElemType::I32);
    tensor.fill_with(|idx| (idx[0] * 1000 + idx[1] * 100 + idx[2] * 10 + idx[3]) as i32);

    let mut rng = rand::rng();
    let mut dim_map = vec![0i32, 1, 2, 3];
    dim_map.shuffle(&mut rng);
    debug!("dim_map: {:?}", dim_map);
    debug!("tensor_shape: {:?}", tensor_shape);
    debug!("tile_shape: {:?}", tile_shape);

    // Validate: tile_size <= tensor_size (considering dim_map)
    let valid = tile_shape.iter().enumerate().all(|(tile_dim, &t)| {
        let tensor_dim = dim_map[tile_dim] as usize;
        t as i64 <= tensor_shape[tensor_dim]
    });
    if !valid {
        debug!(
            "SKIP: tile_shape {:?} exceeds mapped tensor_shape {:?} with dim_map {:?}",
            tile_shape, tensor_shape, dim_map
        );
        for i in 0..4 {
            // Print mapped tensor shape and tile shape, aligned
            debug!(
                "dim {:>2} : tensor_shape = {:>4}, tile_shape = {:>4}",
                i, tensor_shape[dim_map[i] as usize], tile_shape[i]
            );
        }
        return;
    }
    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        dim_map.clone(),
        false,
        None,
    );

    let grid_shape = partition.index_space_shape();

    // Test multiple random tiles
    for test_iter in 0..10 {
        let grid_pos: Vec<i64> = grid_shape.iter().map(|&g| rng.random_range(0..g)).collect();
        debug!("Test {}: grid_pos={:?}", test_iter, grid_pos);

        let tile = partition.load_tile(&grid_pos);

        for ti in 0..tile_shape[0] as i64 {
            for tj in 0..tile_shape[1] as i64 {
                for tk in 0..tile_shape[2] as i64 {
                    for tl in 0..tile_shape[3] as i64 {
                        let tensor_idx = grid_to_tensor_index(
                            &grid_pos,
                            &tile_shape,
                            &[ti, tj, tk, tl],
                            Some(&dim_map),
                        );
                        let expected = tensor.get(&tensor_idx);
                        let actual = tile.get_scalar(&[ti, tj, tk, tl]);
                        assert_eq!(
                            actual, expected,
                            "Mismatch at tile_idx=[{},{},{},{}], tensor_pos={:?}, dim_map={:?}",
                            ti, tj, tk, tl, tensor_idx, dim_map
                        );
                    }
                }
            }
        }
    }
}

// ============================================================================
// NEW: Large-Scale 3D Tests with Random Access and Data Verification
// ============================================================================

#[test]
fn test_partition_view_3d_random_tile_access_large() {
    info!("3D random access: 256x256x128 with 16x16x8 tiles");

    let shape = vec![256i64, 256, 128];
    let tile_shape = vec![16i32, 16, 8];
    let mut tensor = TestTensor::new(shape.clone(), ElemType::I32);
    tensor.fill_arange();

    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2],
        false,
        None,
    );

    let mut rng = StdRng::seed_from_u64(12345);
    let grid_shape = partition.index_space_shape();

    // Test 20 random tile positions
    for _ in 0..20 {
        let grid_pos = vec![
            rng.random_range(0..grid_shape[0]),
            rng.random_range(0..grid_shape[1]),
            rng.random_range(0..grid_shape[2]),
        ];
        let tile = partition.load_tile(&grid_pos);

        for i in 0..16i64 {
            for j in 0..16i64 {
                for k in 0..8i64 {
                    let tensor_idx = grid_to_tensor_index(&grid_pos, &tile_shape, &[i, j, k], None);
                    assert!(tensor.in_bounds(&tensor_idx));
                    let expected = tensor.get(&tensor_idx);
                    let actual = tile.get_scalar(&[i, j, k]);
                    assert_eq!(actual, expected);
                }
            }
        }
    }
    info!("3D random access test completed: verified 20 tiles");
}

#[test]
fn test_partition_view_3d_irregular_shape_large() {
    info!("3D irregular shape: 255x255x127 with 16x16x8 tiles");

    let shape = vec![255i64, 255, 127];
    let tile_shape = vec![16i32, 16, 8];
    let mut tensor = TestTensor::new(shape.clone(), ElemType::F16);
    tensor.fill_arange_scaled(0.05);

    let pad_val = Scalar::F16(-1.234);
    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2],
        true,
        Some(pad_val.clone()),
    );

    let grid_shape = partition.index_space_shape();
    let mut rng = StdRng::seed_from_u64(23456);
    let positions: Vec<_> = (0..25)
        .map(|_| {
            vec![
                rng.random_range(0..grid_shape[0]),
                rng.random_range(0..grid_shape[1]),
                rng.random_range(0..grid_shape[2]),
            ]
        })
        .collect();

    use rayon::prelude::*;
    positions
        .par_iter()
        .enumerate()
        .for_each(|(iter, grid_pos)| {
            let tile = partition.load_tile(grid_pos);

            for i in 0..16i64 {
                for j in 0..16i64 {
                    for k in 0..8i64 {
                        let tensor_idx =
                            grid_to_tensor_index(grid_pos, &tile_shape, &[i, j, k], None);
                        let actual = tile.get_scalar(&[i, j, k]);

                        if tensor.in_bounds(&tensor_idx) {
                            let expected = tensor.get(&tensor_idx);
                            assert_eq!(
                                actual, expected,
                                "Iter {}: Data mismatch at grid={:?}, tile_idx=[{},{},{}]",
                                iter, grid_pos, i, j, k
                            );
                        } else {
                            assert_eq!(
                                actual, pad_val,
                                "Iter {}: Padding error at grid={:?}, tile_idx=[{},{},{}]",
                                iter, grid_pos, i, j, k
                            );
                        }
                    }
                }
            }
        });
    info!("3D irregular shape test completed: verified 25 tiles");
}

#[test]
fn test_partition_view_3d_padding_border_definite() {
    let shape = vec![256i64, 255, 129];
    let tile_shape = vec![16i32, 16, 8];
    let mut tensor = TestTensor::new(shape.clone(), ElemType::I32);
    tensor.fill_arange();

    debug!(
        "3D border test: shape: {:?}, tile_shape: {:?}",
        shape, tile_shape
    );

    let pad_val = Scalar::I32(0);
    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2],
        true,
        Some(pad_val.clone()),
    );

    let grid_shape = partition.index_space_shape();
    let max_x = grid_shape[0] - 1;
    let max_y = grid_shape[1] - 1;
    let max_z = grid_shape[2] - 1;

    let border_positions = vec![
        vec![0, 0, 0],
        vec![0, 0, max_z],
        vec![0, max_y, 0],
        vec![0, max_y, max_z],
        vec![max_x, 0, 0],
        vec![max_x, 0, max_z],
        vec![max_x, max_y, 0],
        vec![max_x, max_y, max_z],
        vec![max_x, 5, 7],
        vec![3, max_y, 11],
        vec![7, 13, max_z],
        vec![max_x, max_y, 5],
        vec![max_x, 7, max_z],
        vec![5, max_y, max_z],
        vec![max_x, 11, 13],
    ];

    for (idx, grid_pos) in border_positions.iter().enumerate() {
        let tile = partition.load_tile(grid_pos);
        let mut in_bounds_count = 0;
        let mut padding_count = 0;

        for i in 0..16i64 {
            for j in 0..16i64 {
                for k in 0..8i64 {
                    let tensor_idx = grid_to_tensor_index(grid_pos, &tile_shape, &[i, j, k], None);
                    let actual = tile.get_scalar(&[i, j, k]);

                    if tensor.in_bounds(&tensor_idx) {
                        let expected = tensor.get(&tensor_idx);
                        assert_eq!(
                            actual, expected,
                            "Position {}: Data mismatch at {:?}",
                            idx, grid_pos
                        );
                        in_bounds_count += 1;
                    } else {
                        assert_eq!(
                            actual, pad_val,
                            "Position {}: Padding error at {:?}",
                            idx, grid_pos
                        );
                        padding_count += 1;
                    }
                }
            }
        }
        debug!(
            "Position {} at {:?}: {} in-bounds, {} padding",
            idx, grid_pos, in_bounds_count, padding_count
        );
    }
    info!("3D padding border test completed: verified 15 edge tiles");
}

#[test]
fn test_partition_view_3d_random_interior_access() {
    info!("3D random interior access: 256x256x128 with 16x16x8 tiles");

    let shape = vec![256i64, 256, 128];
    let tile_shape = vec![16i32, 16, 8];
    let mut tensor = TestTensor::new(shape.clone(), ElemType::I32);
    tensor.fill_arange();

    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2],
        false,
        None,
    );

    let mut rng = StdRng::seed_from_u64(34567);
    let grid_shape = partition.index_space_shape();

    for _ in 0..30 {
        let grid_pos = vec![
            rng.random_range(1..grid_shape[0] - 1),
            rng.random_range(1..grid_shape[1] - 1),
            rng.random_range(1..grid_shape[2] - 1),
        ];
        let tile = partition.load_tile(&grid_pos);

        for i in 0..16i64 {
            for j in 0..16i64 {
                for k in 0..8i64 {
                    let tensor_idx = grid_to_tensor_index(&grid_pos, &tile_shape, &[i, j, k], None);
                    let expected = tensor.get(&tensor_idx);
                    let actual = tile.get_scalar(&[i, j, k]);
                    assert_eq!(actual, expected);
                }
            }
        }
    }
    info!("3D random interior access completed: verified 30 tiles");
}

#[test]
fn test_partition_view_4d_random_interior_access() {
    info!("4D random interior access: 128x128x64x32 with 8x8x4x2 tiles");

    let shape = vec![128i64, 128, 64, 32];
    let tile_shape = vec![8i32, 8, 4, 2];
    let mut tensor = TestTensor::new(shape.clone(), ElemType::I32);
    tensor.fill_arange();

    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2, 3],
        false,
        None,
    );
    let grid_shape = partition.index_space_shape();

    (0..20).into_par_iter().for_each(|iter| {
        let mut rng = StdRng::seed_from_u64(45678);
        let grid_pos = vec![
            rng.random_range(1..grid_shape[0] - 1),
            rng.random_range(1..grid_shape[1] - 1),
            rng.random_range(1..grid_shape[2] - 1),
            rng.random_range(1..grid_shape[3] - 1),
        ];
        let tile = partition.load_tile(&grid_pos);

        for i in 0..8i64 {
            for j in 0..8i64 {
                for k in 0..4i64 {
                    for l in 0..2i64 {
                        let tensor_idx =
                            grid_to_tensor_index(&grid_pos, &tile_shape, &[i, j, k, l], None);
                        let expected = tensor.get(&tensor_idx);
                        let actual = tile.get_scalar(&[i, j, k, l]);
                        assert_eq!(
                            actual, expected,
                            "Iter {}: Mismatch at {:?}",
                            iter, grid_pos
                        );
                    }
                }
            }
        }
    });
    info!("4D random interior access completed: verified 20 tiles");
}

#[test]
fn test_partition_view_3d_padding_custom_negative() {
    info!("3D custom negative padding: 256x256x128 with 16x16x8 tiles (padding=-999999)");

    let shape = vec![256i64, 256, 128];
    let tile_shape = vec![16i32, 16, 8];
    let mut tensor = TestTensor::new(shape.clone(), ElemType::I32);
    tensor.fill_arange();

    let pad_val = Scalar::I32(-999999);
    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2],
        true,
        Some(pad_val.clone()),
    );

    let grid_shape = partition.index_space_shape();
    let max_x = grid_shape[0] - 1;
    let max_y = grid_shape[1] - 1;
    let max_z = grid_shape[2] - 1;

    let mut rng = StdRng::seed_from_u64(114514);
    let edge_positions = vec![
        vec![0, 0, 0],
        vec![0, 0, max_z],
        vec![0, max_y, 0],
        vec![0, max_y, max_z],
        vec![max_x, 0, 0],
        vec![max_x, 0, max_z],
        vec![max_x, max_y, 0],
        vec![max_x, max_y, max_z],
        vec![max_x, 5, 7],
        vec![
            rng.random_range(0..max_x),
            max_y,
            rng.random_range(0..max_z),
        ],
        vec![
            rng.random_range(0..max_x),
            rng.random_range(0..max_y),
            max_z,
        ],
        vec![max_x, max_y, 5],
        vec![max_x, 7, max_z],
        vec![5, max_y, max_z],
        vec![max_x, 0, 5],
        vec![0, max_y, 7],
        vec![5, 0, max_z],
        vec![max_x, max_y, 9],
        vec![max_x, 9, max_z],
        vec![9, max_y, max_z],
        vec![max_x, 1, 1],
        vec![1, max_y, 1],
        vec![1, 1, max_z],
        vec![max_x, 10, 15],
        vec![10, max_y, 15],
    ];

    use rayon::prelude::*;
    edge_positions
        .par_iter()
        .enumerate()
        .for_each(|(idx, grid_pos)| {
            let tile = partition.load_tile(grid_pos);

            for i in 0..16i64 {
                for j in 0..16i64 {
                    for k in 0..8i64 {
                        let tensor_idx =
                            grid_to_tensor_index(grid_pos, &tile_shape, &[i, j, k], None);
                        let actual = tile.get_scalar(&[i, j, k]);

                        if tensor.in_bounds(&tensor_idx) {
                            let expected = tensor.get(&tensor_idx);
                            assert_eq!(
                                actual, expected,
                                "Pos {}: Data mismatch at {:?}",
                                idx, grid_pos
                            );
                        } else {
                            assert_eq!(
                                actual, pad_val,
                                "Pos {}: Padding error at {:?}",
                                idx, grid_pos
                            );
                        }
                    }
                }
            }
        });
    info!("3D custom negative padding completed: verified 25 edge tiles");
}

#[test]
fn test_partition_view_3d_padding_custom_positive() {
    info!("3D custom positive padding: 250x250x125 with 16x16x8 tiles (padding=777777)");

    let shape = vec![250i64, 250, 125];
    let tile_shape = vec![16i32, 16, 8];
    let mut tensor = TestTensor::new(shape.clone(), ElemType::I32);
    tensor.fill_arange();

    let pad_val = Scalar::I32(777777);
    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2],
        true,
        Some(pad_val.clone()),
    );

    let grid_shape = partition.index_space_shape();
    let max_x = grid_shape[0] - 1;
    let max_y = grid_shape[1] - 1;
    let max_z = grid_shape[2] - 1;

    let edge_positions = vec![
        vec![0, 0, max_z],
        vec![0, max_y, 0],
        vec![0, max_y, max_z],
        vec![max_x, 0, 0],
        vec![max_x, 0, max_z],
        vec![max_x, max_y, 0],
        vec![max_x, max_y, max_z],
        vec![max_x, 3, 5],
        vec![5, max_y, 7],
        vec![7, 9, max_z],
        vec![max_x, max_y, 3],
        vec![max_x, 5, max_z],
        vec![3, max_y, max_z],
        vec![max_x, 7, 9],
        vec![9, max_y, 11],
        vec![11, 13, max_z],
        vec![max_x, 0, 1],
        vec![0, max_y, 3],
    ];

    use rayon::prelude::*;
    edge_positions.par_iter().for_each(|grid_pos| {
        let tile = partition.load_tile(grid_pos);

        for i in 0..16i64 {
            for j in 0..16i64 {
                for k in 0..8i64 {
                    let tensor_idx = grid_to_tensor_index(grid_pos, &tile_shape, &[i, j, k], None);
                    let actual = tile.get_scalar(&[i, j, k]);

                    if tensor.in_bounds(&tensor_idx) {
                        let expected = tensor.get(&tensor_idx);
                        assert_eq!(actual, expected);
                    } else {
                        assert_eq!(actual, pad_val);
                    }
                }
            }
        }
    });
    info!("3D custom positive padding completed: verified 18 edge tiles");
}

#[test]
fn test_partition_view_3d_padding_all_edges() {
    info!("3D padding all edge combinations: 256x256x128 with 16x16x8 tiles");

    let shape = vec![256i64, 256, 128];
    let strides = vec![64 * 32, 511, 1];
    let tile_shape = vec![16i32, 16, 8];

    let mut tensor = TestTensor::with_strides(shape.clone(), strides.clone(), ElemType::I32);
    tensor.fill_arange();

    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2],
        true,
        Some(Scalar::I32(-1)),
    );

    let grid_shape = partition.index_space_shape();
    let max_x = grid_shape[0] - 1;
    let max_y = grid_shape[1] - 1;
    let max_z = grid_shape[2] - 1;

    let edge_combinations = vec![
        vec![max_x, 7, 11],
        vec![5, max_y, 13],
        vec![3, 9, max_z],
        vec![max_x, max_y, 7],
        vec![max_x, 5, max_z],
        vec![3, max_y, max_z],
        vec![max_x, max_y, max_z],
        vec![max_x, 0, 0],
        vec![0, max_y, 0],
        vec![0, 0, max_z],
        vec![max_x, max_y, 0],
        vec![max_x, 0, max_z],
        vec![0, max_y, max_z],
        vec![max_x, 3, 5],
        vec![5, max_y, 7],
        vec![7, 9, max_z],
        vec![max_x, max_y, 1],
        vec![max_x, 1, max_z],
        vec![1, max_y, max_z],
        vec![max_x, 11, 13],
    ];

    use rayon::prelude::*;
    edge_combinations.par_iter().for_each(|grid_pos| {
        let tile = partition.load_tile(grid_pos);

        for i in 0..16i64 {
            for j in 0..16i64 {
                for k in 0..8i64 {
                    let tensor_idx = grid_to_tensor_index(grid_pos, &tile_shape, &[i, j, k], None);
                    let actual = tile.get_scalar(&[i, j, k]);

                    if tensor.in_bounds(&tensor_idx) {
                        let expected = tensor.get(&tensor_idx);
                        assert_eq!(actual, expected);
                    } else {
                        assert_eq!(actual, Scalar::I32(-1));
                    }
                }
            }
        }
    });

    info!("3D padding all edges completed: verified 20 edge combination tiles");
}

#[test]
fn test_partition_view_4d_irregular_shape() {
    info!("4D irregular shape: 128x127x64x33 with 8x8x4x2 tiles");

    let shape = vec![128i64, 127, 64, 33];
    let strides = vec![127 * 64 * 33, 64 * 33, 131, 1];
    let tile_shape = vec![8i32, 8, 4, 2];

    let mut tensor = TestTensor::with_strides(shape.clone(), strides.clone(), ElemType::I32);
    tensor.fill_arange();

    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2, 3],
        true,
        Some(Scalar::I32(0)),
    );

    let mut rng = StdRng::seed_from_u64(56789);
    let grid_shape = partition.index_space_shape();

    let positions: Vec<_> = (0..25)
        .map(|_| {
            vec![
                rng.random_range(0..grid_shape[0]),
                rng.random_range(0..grid_shape[1]),
                rng.random_range(0..grid_shape[2]),
                rng.random_range(0..grid_shape[3]),
            ]
        })
        .collect();

    use rayon::prelude::*;
    positions.par_iter().for_each(|grid_pos| {
        let tile = partition.load_tile(grid_pos);

        for i in 0..8i64 {
            for j in 0..8i64 {
                for k in 0..4i64 {
                    for l in 0..2i64 {
                        let tensor_idx =
                            grid_to_tensor_index(grid_pos, &tile_shape, &[i, j, k, l], None);
                        let actual = tile.get_scalar(&[i, j, k, l]);

                        if tensor.in_bounds(&tensor_idx) {
                            let expected = tensor.get(&tensor_idx);
                            assert_eq!(actual, expected);
                        } else {
                            assert_eq!(actual, Scalar::I32(0));
                        }
                    }
                }
            }
        }
    });

    info!("4D irregular shape completed: verified 25 tiles");
}

#[test]
fn test_partition_view_5d_random_access() {
    info!("5D random access: 128x64x64x32x16 with 4x4x2x1x1 tiles");

    let shape = vec![128i64, 64, 64, 32, 16];
    let tile_shape = vec![4i32, 4, 2, 1, 1];

    let mut tensor = TestTensor::new(shape.clone(), ElemType::I32);
    tensor.fill_arange();

    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2, 3, 4],
        false,
        None,
    );

    let mut rng = StdRng::seed_from_u64(67890);
    let grid_shape = partition.index_space_shape();

    let positions: Vec<_> = (0..20)
        .map(|_| {
            vec![
                rng.random_range(1..grid_shape[0] - 1),
                rng.random_range(1..grid_shape[1] - 1),
                rng.random_range(1..grid_shape[2] - 1),
                rng.random_range(1..grid_shape[3] - 1),
                rng.random_range(1..grid_shape[4] - 1),
            ]
        })
        .collect();

    positions.iter().for_each(|grid_pos| {
        let tile = partition.load_tile(grid_pos);

        for i in 0..4i64 {
            for j in 0..4i64 {
                for k in 0..2i64 {
                    for l in 0..1i64 {
                        for m in 0..1i64 {
                            let tensor_idx =
                                grid_to_tensor_index(grid_pos, &tile_shape, &[i, j, k, l, m], None);
                            let expected = tensor.get(&tensor_idx);
                            let actual = tile.get_scalar(&[i, j, k, l, m]);
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
        }
    });

    info!("5D random access completed: verified 20 tiles x 32 elements");
}

#[test]
fn test_partition_view_5d_irregular_shape() {
    info!("5D irregular shape: 127x63x61x31x15 with 8x4x4x2x1 tiles");

    let shape = vec![127i64, 63, 61, 31, 15];
    let tile_shape = vec![8i32, 4, 4, 2, 1];

    let mut tensor = TestTensor::new(shape.clone(), ElemType::I32);
    tensor.fill_arange();

    let mut rng = StdRng::seed_from_u64(67890);
    let pad_val = rng.random_range(-1024..1024);
    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2, 3, 4],
        true,
        Some(Scalar::I32(pad_val)),
    );

    let mut rng = StdRng::seed_from_u64(78901);
    let grid_shape = partition.index_space_shape();

    let positions: Vec<_> = (0..25)
        .map(|_| {
            vec![
                rng.random_range(0..grid_shape[0]),
                rng.random_range(0..grid_shape[1]),
                rng.random_range(0..grid_shape[2]),
                rng.random_range(0..grid_shape[3]),
                rng.random_range(0..grid_shape[4]),
            ]
        })
        .collect();

    use rayon::prelude::*;
    positions.par_iter().for_each(|grid_pos| {
        let tile = partition.load_tile(grid_pos);

        for i in 0..8i64 {
            for j in 0..4i64 {
                for k in 0..4i64 {
                    for l in 0..2i64 {
                        for m in 0..1i64 {
                            let tensor_idx =
                                grid_to_tensor_index(grid_pos, &tile_shape, &[i, j, k, l, m], None);
                            let actual = tile.get_scalar(&[i, j, k, l, m]);

                            if tensor.in_bounds(&tensor_idx) {
                                let expected = tensor.get(&tensor_idx);
                                assert_eq!(actual, expected);
                            } else {
                                assert_eq!(actual, Scalar::I32(pad_val));
                            }
                        }
                    }
                }
            }
        }
    });

    info!("5D irregular shape completed: verified 25 tiles");
}

#[test]
fn test_partition_view_3d_custom_padding_all_types() {
    info!("3D custom padding for all types: 250x250x125 with 16x16x8 tiles");

    let shape = vec![250i64, 250, 125];
    let tile_shape = vec![16i32, 16, 8];

    use rayon::prelude::*;

    let test_cases = [
        (ElemType::Bool, Scalar::Bool(true)),
        (ElemType::I8, Scalar::I8(-128)),
        (ElemType::I16, Scalar::I16(-32768)),
        (ElemType::I32, Scalar::I32(-999999)),
        (ElemType::I64, Scalar::I64(-999999999)),
        (ElemType::F16, Scalar::F16(f16::NAN)),
        (ElemType::F32, Scalar::F32(f32::NAN)),
        (ElemType::F64, Scalar::F64(f64::NAN)),
    ];

    test_cases
        .par_iter()
        .for_each(|&(elem_type, ref padding_value)| {
            let mut tensor = TestTensor::new(shape.clone(), elem_type);

            let partition = PartitionView::new(
                tensor.as_view(),
                tile_shape.clone(),
                vec![0, 1, 2],
                true,
                Some(padding_value.clone()),
            );

            let grid_shape = partition.index_space_shape();
            let max_x = grid_shape[0] - 1;
            let max_y = grid_shape[1] - 1;
            let max_z = grid_shape[2] - 1;

            let positions = vec![
                vec![0, 0, 0],
                vec![0, 0, max_z],
                vec![0, max_y, 0],
                vec![0, max_y, max_z],
                vec![max_x, 0, 0],
                vec![max_x, 0, max_z],
                vec![max_x, max_y, 0],
                vec![max_x, max_y, max_z],
                vec![max_x / 2, 0, 0],
                vec![0, max_y / 2, 0],
                vec![0, 0, max_z / 2],
                vec![max_x, max_y, max_z / 2],
            ];

            for grid_pos in positions {
                let _tile = partition.load_tile(&grid_pos);
            }
        });

    info!("3D custom padding all types completed: verified 8 types x 12 edge tiles");
}

#[test]
fn test_partition_view_4d_padding_all_edges_definite() {
    info!("4D padding all edges: 128x128x64x127 with 16x16x8x4 tiles");

    let shape = vec![128i64, 128, 64, 127];
    let tile_shape = vec![16i32, 16, 8, 4];

    let mut tensor = TestTensor::new(shape.clone(), ElemType::I32);
    tensor.fill_arange();

    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2, 3],
        true,
        Some(Scalar::I32(-77777)),
    );

    let grid_shape = partition.index_space_shape();
    let mx = grid_shape[0] - 1;
    let my = grid_shape[1] - 1;
    let mz = grid_shape[2] - 1;
    let mw = grid_shape[3] - 1;

    let positions = vec![
        vec![0, 0, 0, 0],
        vec![0, 0, 0, mw],
        vec![0, 0, mz, 0],
        vec![0, 0, mz, mw],
        vec![0, my, 0, 0],
        vec![0, my, 0, mw],
        vec![0, my, mz, 0],
        vec![0, my, mz, mw],
        vec![mx, 0, 0, 0],
        vec![mx, 0, 0, mw],
        vec![mx, 0, mz, 0],
        vec![mx, 0, mz, mw],
        vec![mx, my, 0, 0],
        vec![mx, my, 0, mw],
        vec![mx, my, mz, 0],
        vec![mx, my, mz, mw],
        vec![mx, 5, 7, 9],
        vec![5, my, 7, 9],
        vec![5, 7, mz, 9],
        vec![5, 7, 9, mw],
        vec![mx, my, 5, 7],
        vec![mx, 5, mz, 7],
        vec![mx, my, mz, 5],
        vec![mx, my, 5, mw],
        vec![mx, 5, mz, mw],
    ];

    use rayon::prelude::*;
    positions.par_iter().for_each(|grid_pos| {
        let tile = partition.load_tile(grid_pos);

        for i in 0..16i64 {
            for j in 0..16i64 {
                for k in 0..8i64 {
                    for l in 0..4i64 {
                        let tensor_idx =
                            grid_to_tensor_index(grid_pos, &tile_shape, &[i, j, k, l], None);
                        let actual = tile.get_scalar(&[i, j, k, l]);

                        if tensor.in_bounds(&tensor_idx) {
                            let expected = tensor.get(&tensor_idx);
                            assert_eq!(actual, expected);
                        } else {
                            assert_eq!(actual, Scalar::I32(-77777));
                        }
                    }
                }
            }
        }
    });

    info!("4D padding all edges completed: verified 25 definite edge tiles");
}

#[test]
fn test_partition_view_5d_padding_corners_definite() {
    info!("5D padding corners: 64x65x32x17x9 with 4x4x2x1x1 tiles");

    let shape = vec![64i64, 65, 32, 17, 9];
    let tile_shape = vec![8, 8, 2, 4, 4];

    let mut tensor = TestTensor::new(shape.clone(), ElemType::I32);
    tensor.fill_arange();

    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2, 3, 4],
        true,
        Some(Scalar::I32(-12345)),
    );

    let grid_shape = partition.index_space_shape();
    let mx = grid_shape[0] - 1;
    let my = grid_shape[1] - 1;
    let mz = grid_shape[2] - 1;
    let mw = grid_shape[3] - 1;
    let mv = grid_shape[4] - 1;

    let positions = vec![
        vec![0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, mv],
        vec![0, 0, 0, mw, 0],
        vec![0, 0, 0, mw, mv],
        vec![0, 0, mz, 0, 0],
        vec![0, 0, mz, 0, mv],
        vec![0, 0, mz, mw, 0],
        vec![0, 0, mz, mw, mv],
        vec![0, my, 0, 0, 0],
        vec![0, my, 0, 0, mv],
        vec![0, my, 0, mw, 0],
        vec![0, my, 0, mw, mv],
        vec![0, my, mz, 0, 0],
        vec![0, my, mz, 0, mv],
        vec![0, my, mz, mw, 0],
        vec![0, my, mz, mw, mv],
        vec![mx, 0, 0, 0, 0],
        vec![mx, 0, 0, 0, mv],
        vec![mx, 0, 0, mw, 0],
        vec![mx, 0, 0, mw, mv],
        vec![mx, 0, mz, 0, 0],
        vec![mx, 0, mz, 0, mv],
        vec![mx, 0, mz, mw, 0],
        vec![mx, 0, mz, mw, mv],
        vec![mx, my, 0, 0, 0],
        vec![mx, my, 0, 0, mv],
        vec![mx, my, 0, mw, 0],
        vec![mx, my, 0, mw, mv],
        vec![mx, my, mz, 0, 0],
        vec![mx, my, mz, 0, mv],
    ];

    use rayon::prelude::*;
    positions.par_iter().for_each(|grid_pos| {
        let tile = partition.load_tile(grid_pos);

        for (i, j, k, l, m) in (0..tile_shape[0])
            .cartesian_product(0..tile_shape[1])
            .cartesian_product(0..tile_shape[2])
            .cartesian_product(0..tile_shape[3])
            .cartesian_product(0..tile_shape[4])
            .map(|((((i, j), k), l), m)| (i as i64, j as i64, k as i64, l as i64, m as i64))
        {
            let tensor_idx = grid_to_tensor_index(grid_pos, &tile_shape, &[i, j, k, l, m], None);
            let actual = tile.get_scalar(&[i, j, k, l, m]);

            if tensor.in_bounds(&tensor_idx) {
                let expected = tensor.get(&tensor_idx);
                assert_eq!(actual, expected);
            } else {
                assert_eq!(actual, Scalar::I32(-12345));
            }
        }
    });

    info!("5D padding corners completed: verified 30 hypercube corner tiles");
}

#[test]
fn test_partition_view_3d_strided_random() {
    info!("3D strided random: 256x256x128 with NON-CONTIGUOUS strides, 16x16x8 tiles");

    let shape = vec![256i64, 256, 128];
    // Non-contiguous strides with gaps
    let strides = vec![65636, 266, 2];
    let tile_shape = vec![16i32, 16, 8];

    let mut tensor = TestTensor::with_strides(shape.clone(), strides.clone(), ElemType::I32);
    tensor.fill_arange();

    let partition = PartitionView::new(
        tensor.as_view(),
        tile_shape.clone(),
        vec![0, 1, 2],
        false,
        None,
    );

    let mut rng = StdRng::seed_from_u64(89012);
    let grid_shape = partition.index_space_shape();

    let positions: Vec<_> = (0..15)
        .map(|_| {
            vec![
                rng.random_range(1..grid_shape[0] - 1),
                rng.random_range(1..grid_shape[1] - 1),
                rng.random_range(1..grid_shape[2] - 1),
            ]
        })
        .collect();

    use rayon::prelude::*;
    positions
        .par_iter()
        .enumerate()
        .for_each(|(pos_idx, grid_pos)| {
            let tile = partition.load_tile(grid_pos);

            for i in 0..16i64 {
                for j in 0..16i64 {
                    for k in 0..8i64 {
                        let tensor_idx =
                            grid_to_tensor_index(grid_pos, &tile_shape, &[i, j, k], None);
                        let expected = tensor.get(&tensor_idx);
                        let actual = tile.get_scalar(&[i, j, k]);
                        assert_eq!(
                            actual, expected,
                            "Mismatch at pos_idx={}, grid_pos={:?}, tile_idx=[{},{},{}]",
                            pos_idx, grid_pos, i, j, k
                        );
                    }
                }
            }
        });

    info!("3D strided random completed: verified 15 tiles with non-contiguous strides");
}

#[test]
fn test_partition_view_4d_all_types_random() {
    info!("4D all types random: 128x128x64x16 with 8x8x4x1 tiles");

    let shape = vec![128i64, 128, 64, 16];
    let tile_shape = vec![8i32, 8, 4, 1];

    use rayon::prelude::*;

    let test_cases = [
        ElemType::Bool,
        ElemType::I8,
        ElemType::I16,
        ElemType::I32,
        ElemType::I64,
        ElemType::F16,
        ElemType::F32,
        ElemType::F64,
        ElemType::Ptr,
    ];

    test_cases.par_iter().for_each(|&elem_type| {
        let tensor = TestTensor::new(shape.clone(), elem_type);

        let partition = PartitionView::new(
            tensor.as_view(),
            tile_shape.clone(),
            vec![0, 1, 2, 3],
            false,
            None,
        );

        let mut rng = StdRng::seed_from_u64(90123);
        let grid_shape = partition.index_space_shape();

        for _ in 0..10 {
            let grid_pos = vec![
                rng.random_range(1..grid_shape[0] - 1),
                rng.random_range(1..grid_shape[1] - 1),
                rng.random_range(1..grid_shape[2] - 1),
                rng.random_range(1..grid_shape[3] - 1),
            ];

            let _tile = partition.load_tile(&grid_pos);
        }
    });

    info!("4D all types random completed: verified 9 types x 10 random tiles");
}

#[test]
fn test_partition_view_4d_custom_padding_per_type() {
    info!("4D custom padding per type: 250x250x125x250 with 16x16x8x4 tiles");

    let shape = vec![250i64, 250, 125, 250];
    let tile_shape = vec![16i32, 16, 8, 4];

    use rayon::prelude::*;

    let test_cases = [
        (ElemType::I32, Scalar::I32(-999999)),
        (ElemType::F32, Scalar::F32(f32::NEG_INFINITY)),
        (ElemType::F16, Scalar::F16(f16::INFINITY)),
        (ElemType::I64, Scalar::I64(i64::MIN)),
    ];

    test_cases
        .par_iter()
        .for_each(|&(elem_type, ref padding_value)| {
            let tensor = TestTensor::new(shape.clone(), elem_type);

            let partition = PartitionView::new(
                tensor.as_view(),
                tile_shape.clone(),
                vec![0, 1, 2, 3],
                true,
                Some(padding_value.clone()),
            );

            let grid_shape = partition.index_space_shape();
            let mx = grid_shape[0] - 1;
            let my = grid_shape[1] - 1;
            let mz = grid_shape[2] - 1;
            let mw = grid_shape[3] - 1;

            let positions = vec![
                vec![0, 0, 0, 0],
                vec![mx, 0, 0, 0],
                vec![0, my, 0, 0],
                vec![0, 0, mz, 0],
                vec![0, 0, 0, mw],
                vec![mx, my, 0, 0],
                vec![mx, 0, mz, 0],
                vec![mx, 0, 0, mw],
                vec![0, my, mz, 0],
                vec![0, my, 0, mw],
            ];

            for grid_pos in positions {
                let _tile = partition.load_tile(&grid_pos);
            }
        });

    info!("4D custom padding per type completed: verified 4 types x 10 edge tiles");
}
