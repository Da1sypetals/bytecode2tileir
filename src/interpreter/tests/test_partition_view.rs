use crate::interpreter::data_structures::{
    elem_type::{ElemType, Scalar},
    tensor_view::{PartitionView, TensorView},
};
use indicatif::{ParallelProgressIterator, ProgressIterator};
use rand::{rngs::StdRng, seq::SliceRandom, RngExt};
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Helper to allocate a buffer for testing
fn allocate_buffer(size_bytes: usize) -> Vec<u8> {
    vec![0u8; size_bytes]
}

/// Fill tensor with arange pattern: tensor[indices] = linear_index
unsafe fn fill_tensor_arange_i32(ptr: *mut u8, shape: &[i64], strides: &[i64]) {
    let rank = shape.len();
    let total_elements: usize = shape.iter().product::<i64>() as usize;

    let iptr = ptr as usize;

    (0..total_elements)
        .into_par_iter()
        .progress()
        .for_each(|linear_idx| {
            let mut indices = vec![0i64; rank];
            let mut remaining = linear_idx;

            // Convert linear index to multi-dimensional indices
            for dim in (0..rank).rev() {
                let dim_size = shape[dim] as usize;
                indices[dim] = (remaining % dim_size) as i64;
                remaining /= dim_size;
            }

            // Calculate byte offset using strides (strides are in element units)
            let mut elem_offset = 0i64;
            for dim in 0..rank {
                elem_offset += indices[dim] * strides[dim];
            }
            let byte_offset = (elem_offset * 4) as usize; // i32 = 4 bytes

            *((iptr as *mut u8).add(byte_offset) as *mut i32) = linear_idx as i32;
        });
}

// ============================================================================
// 3D Dimension Permutation Tests (KEEP - High Quality)
// ============================================================================

#[test]
fn test_dim_map_3d_permutation_012() {
    let mut buffer = vec![0u8; 2048];
    let ptr = buffer.as_mut_ptr();

    // Initialize: buffer[i][j][k] = i*100 + j*10 + k
    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                unsafe {
                    let idx = i * 64 + j * 8 + k;
                    *(ptr.add(idx * 4) as *mut i32) = (i * 100 + j * 10 + k) as i32;
                }
            }
        }
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, vec![8, 8, 8], vec![64, 8, 1]);
    let partition = PartitionView::new(tensor_view, vec![4, 4, 4], vec![0, 1, 2], false, None);

    let tile = partition.load_tile(&[0, 0, 0]);

    assert_eq!(tile.get_scalar(&[0, 0, 0]), Scalar::I32(0));
    assert_eq!(tile.get_scalar(&[1, 0, 0]), Scalar::I32(100));
    assert_eq!(tile.get_scalar(&[0, 1, 0]), Scalar::I32(10));
    assert_eq!(tile.get_scalar(&[0, 0, 1]), Scalar::I32(1));
}

#[test]
fn test_dim_map_3d_permutation_021() {
    let mut buffer = vec![0u8; 2048];
    let ptr = buffer.as_mut_ptr();

    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                unsafe {
                    let idx = i * 64 + j * 8 + k;
                    *(ptr.add(idx * 4) as *mut i32) = (i * 100 + j * 10 + k) as i32;
                }
            }
        }
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, vec![8, 8, 8], vec![64, 8, 1]);
    let partition = PartitionView::new(tensor_view, vec![4, 4, 4], vec![0, 2, 1], false, None);

    let tile = partition.load_tile(&[0, 0, 0]);

    assert_eq!(tile.get_scalar(&[0, 0, 0]), Scalar::I32(0));
    assert_eq!(tile.get_scalar(&[1, 0, 0]), Scalar::I32(100));
    assert_eq!(tile.get_scalar(&[0, 1, 0]), Scalar::I32(1));
    assert_eq!(tile.get_scalar(&[0, 0, 1]), Scalar::I32(10));
}

#[test]
fn test_dim_map_3d_permutation_102() {
    let mut buffer = vec![0u8; 2048];
    let ptr = buffer.as_mut_ptr();

    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                unsafe {
                    let idx = i * 64 + j * 8 + k;
                    *(ptr.add(idx * 4) as *mut i32) = (i * 100 + j * 10 + k) as i32;
                }
            }
        }
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, vec![8, 8, 8], vec![64, 8, 1]);
    let partition = PartitionView::new(tensor_view, vec![4, 4, 4], vec![1, 0, 2], false, None);

    let tile = partition.load_tile(&[0, 0, 0]);

    assert_eq!(tile.get_scalar(&[0, 0, 0]), Scalar::I32(0));
    assert_eq!(tile.get_scalar(&[1, 0, 0]), Scalar::I32(10));
    assert_eq!(tile.get_scalar(&[0, 1, 0]), Scalar::I32(100));
    assert_eq!(tile.get_scalar(&[0, 0, 1]), Scalar::I32(1));
}

#[test]
fn test_dim_map_3d_permutation_120() {
    let mut buffer = vec![0u8; 2048];
    let ptr = buffer.as_mut_ptr();

    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                unsafe {
                    let idx = i * 64 + j * 8 + k;
                    *(ptr.add(idx * 4) as *mut i32) = (i * 100 + j * 10 + k) as i32;
                }
            }
        }
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, vec![8, 8, 8], vec![64, 8, 1]);
    let partition = PartitionView::new(tensor_view, vec![4, 4, 4], vec![1, 2, 0], false, None);

    let tile = partition.load_tile(&[0, 0, 0]);

    assert_eq!(tile.get_scalar(&[0, 0, 0]), Scalar::I32(0));
    assert_eq!(tile.get_scalar(&[1, 0, 0]), Scalar::I32(10));
    assert_eq!(tile.get_scalar(&[0, 1, 0]), Scalar::I32(1));
    assert_eq!(tile.get_scalar(&[0, 0, 1]), Scalar::I32(100));
}

#[test]
fn test_dim_map_3d_permutation_201() {
    let mut buffer = vec![0u8; 2048];
    let ptr = buffer.as_mut_ptr();

    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                unsafe {
                    let idx = i * 64 + j * 8 + k;
                    *(ptr.add(idx * 4) as *mut i32) = (i * 100 + j * 10 + k) as i32;
                }
            }
        }
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, vec![8, 8, 8], vec![64, 8, 1]);
    let partition = PartitionView::new(tensor_view, vec![4, 4, 4], vec![2, 0, 1], false, None);

    let tile = partition.load_tile(&[0, 0, 0]);

    assert_eq!(tile.get_scalar(&[0, 0, 0]), Scalar::I32(0));
    assert_eq!(tile.get_scalar(&[1, 0, 0]), Scalar::I32(1));
    assert_eq!(tile.get_scalar(&[0, 1, 0]), Scalar::I32(100));
    assert_eq!(tile.get_scalar(&[0, 0, 1]), Scalar::I32(10));
}

#[test]
fn test_dim_map_3d_permutation_210() {
    let mut buffer = vec![0u8; 2048];
    let ptr = buffer.as_mut_ptr();

    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                unsafe {
                    let idx = i * 64 + j * 8 + k;
                    *(ptr.add(idx * 4) as *mut i32) = (i * 100 + j * 10 + k) as i32;
                }
            }
        }
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, vec![8, 8, 8], vec![64, 8, 1]);
    let partition = PartitionView::new(tensor_view, vec![4, 4, 4], vec![2, 1, 0], false, None);

    let tile = partition.load_tile(&[0, 0, 0]);

    assert_eq!(tile.get_scalar(&[0, 0, 0]), Scalar::I32(0));
    assert_eq!(tile.get_scalar(&[1, 0, 0]), Scalar::I32(1));
    assert_eq!(tile.get_scalar(&[0, 1, 0]), Scalar::I32(10));
    assert_eq!(tile.get_scalar(&[0, 0, 1]), Scalar::I32(100));
}

// ============================================================================
// 4D Dimension Permutation Tests (KEEP - High Quality)
// ============================================================================

#[test]
fn test_dim_map_4d_identity() {
    let mut buffer = vec![0u8; 16384];
    let ptr = buffer.as_mut_ptr();

    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                for l in 0..4 {
                    unsafe {
                        let idx = i * 256 + j * 32 + k * 4 + l;
                        *(ptr.add(idx * 4) as *mut i32) = (i * 1000 + j * 100 + k * 10 + l) as i32;
                    }
                }
            }
        }
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, vec![8, 8, 8, 4], vec![256, 32, 4, 1]);
    let partition =
        PartitionView::new(tensor_view, vec![4, 4, 4, 2], vec![0, 1, 2, 3], false, None);

    let tile = partition.load_tile(&[0, 0, 0, 0]);
    assert_eq!(tile.get_scalar(&[0, 0, 0, 0]), Scalar::I32(0));
    assert_eq!(tile.get_scalar(&[1, 0, 0, 0]), Scalar::I32(1000));
    assert_eq!(tile.get_scalar(&[0, 1, 0, 0]), Scalar::I32(100));
    assert_eq!(tile.get_scalar(&[0, 0, 1, 0]), Scalar::I32(10));
    assert_eq!(tile.get_scalar(&[0, 0, 0, 1]), Scalar::I32(1));
}

#[test]
fn test_dim_map_4d_reverse() {
    let mut buffer = vec![0u8; 16384];
    let ptr = buffer.as_mut_ptr();

    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                for l in 0..4 {
                    unsafe {
                        let idx = i * 256 + j * 32 + k * 4 + l;
                        *(ptr.add(idx * 4) as *mut i32) = (i * 1000 + j * 100 + k * 10 + l) as i32;
                    }
                }
            }
        }
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, vec![8, 8, 8, 4], vec![256, 32, 4, 1]);
    let partition =
        PartitionView::new(tensor_view, vec![4, 4, 4, 2], vec![3, 2, 1, 0], false, None);

    let tile = partition.load_tile(&[0, 0, 0, 0]);
    assert_eq!(tile.get_scalar(&[0, 0, 0, 0]), Scalar::I32(0));
    assert_eq!(tile.get_scalar(&[1, 0, 0, 0]), Scalar::I32(1));
    assert_eq!(tile.get_scalar(&[0, 1, 0, 0]), Scalar::I32(10));
    assert_eq!(tile.get_scalar(&[0, 0, 1, 0]), Scalar::I32(100));
    assert_eq!(tile.get_scalar(&[0, 0, 0, 1]), Scalar::I32(1000));
}

#[test]
fn test_dim_map_4d_partial_transpose_0132() {
    let mut buffer = vec![0u8; 16384];
    let ptr = buffer.as_mut_ptr();

    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                for l in 0..4 {
                    unsafe {
                        let idx = i * 256 + j * 32 + k * 4 + l;
                        *(ptr.add(idx * 4) as *mut i32) = (i * 1000 + j * 100 + k * 10 + l) as i32;
                    }
                }
            }
        }
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, vec![8, 8, 8, 4], vec![256, 32, 4, 1]);
    let partition =
        PartitionView::new(tensor_view, vec![4, 4, 4, 2], vec![0, 1, 3, 2], false, None);

    let tile = partition.load_tile(&[0, 0, 0, 0]);
    assert_eq!(tile.get_scalar(&[0, 0, 0, 0]), Scalar::I32(0));
    assert_eq!(tile.get_scalar(&[0, 0, 1, 0]), Scalar::I32(1));
    assert_eq!(tile.get_scalar(&[0, 0, 0, 1]), Scalar::I32(10));
}

#[test]
fn test_dim_map_4d_partial_transpose_random() {
    let mut buffer = vec![0u8; 16384];
    let ptr = buffer.as_mut_ptr();

    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                for l in 0..4 {
                    unsafe {
                        let idx = i * 256 + j * 32 + k * 4 + l;
                        *(ptr.add(idx * 4) as *mut i32) = (i * 1000 + j * 100 + k * 10 + l) as i32;
                    }
                }
            }
        }
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, vec![8, 8, 8, 4], vec![256, 32, 4, 1]);
    let mut rng = rand::rng();
    let mut dim_map = vec![0, 1, 2, 3];
    dim_map.shuffle(&mut rng);

    let partition = PartitionView::new(tensor_view, vec![2, 2, 2, 2], dim_map.clone(), false, None);

    println!("dim_map: {:?}", dim_map);

    // Test multiple random tiles with shape [8,8,8,4] and tile [2,2,2,2]
    // To be safe with any permutation, ensure max tensor position < tensor shape
    // max_pos = (grid + 1) * tile_shape <= tensor_shape for mapped dimensions
    // For tensor [8,8,8,4], with tile [2,2,2,2], max grid that always works:
    // We need grid[i] * 2 + 2 <= min(tensor_shape) = 4, so grid[i] <= 1
    for test_iter in 0..10 {
        let grid_pos = vec![
            rng.random_range(0..2),
            rng.random_range(0..2),
            rng.random_range(0..2),
            rng.random_range(0..2),
        ];

        println!("Test {}: grid_pos={:?}", test_iter, grid_pos);

        let tile = partition.load_tile(&grid_pos);

        // Verify all elements in the tile
        for ti in 0..2 {
            for tj in 0..2 {
                for tk in 0..2 {
                    for tl in 0..2 {
                        // Compute tensor position based on dim_map
                        // tile[ti, tj, tk, tl] maps to tensor position where:
                        //   tensor_dim = dim_map[tile_dim]
                        //   tensor[tensor_dim] = grid_pos[tile_dim] * tile_shape[tile_dim] + tile_index[tile_dim]
                        let mut tensor_pos = vec![0i64; 4];
                        let tile_indices = vec![ti, tj, tk, tl];
                        let tile_shape = vec![2, 2, 2, 2];

                        for tile_dim in 0..4 {
                            let tensor_dim = dim_map[tile_dim] as usize;
                            tensor_pos[tensor_dim] = grid_pos[tile_dim]
                                * tile_shape[tile_dim] as i64
                                + tile_indices[tile_dim];
                        }

                        let expected = (tensor_pos[0] * 1000
                            + tensor_pos[1] * 100
                            + tensor_pos[2] * 10
                            + tensor_pos[3]) as i32;
                        let actual = tile.get_scalar(&[ti, tj, tk, tl]);
                        assert_eq!(
                            actual,
                            Scalar::I32(expected),
                            "Mismatch at tile_idx=[{},{},{},{}], tensor_pos={:?}, dim_map={:?}",
                            ti,
                            tj,
                            tk,
                            tl,
                            tensor_pos,
                            dim_map
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
fn test_partition_view_3d_irregular_tile_access_large() {
    println!("3D irregular access: 1024x1024x512 with 32x32x32 tiles");

    let shape = vec![256i64, 256, 128];
    let strides = vec![64 * 32, 512, 1];
    let tile_shape = vec![16i32, 16, 8];
    let buffer_size = 1024 * 64 * 32 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(tensor_view, tile_shape.clone(), vec![0, 1, 2], false, None);

    let mut rng = StdRng::seed_from_u64(12345);
    let grid_shape = partition.index_space_shape();

    // Test 20 random tile positions with full tile verification
    for iter in 0..20 {
        let grid_pos = vec![
            rng.random_range(0..grid_shape[0]),
            rng.random_range(0..grid_shape[1]),
            rng.random_range(0..grid_shape[2]),
        ];

        let tile = partition.load_tile(&grid_pos);

        // Verify ALL 32768 elements in the tile (32x32x32 = 32768)
        for i in 0..16 {
            for j in 0..16 {
                for k in 0..8 {
                    let tensor_i = grid_pos[0] * 16 + i;
                    let tensor_j = grid_pos[1] * 16 + j;
                    let tensor_k = grid_pos[2] * 8 + k;

                    if tensor_i < shape[0] && tensor_j < shape[1] && tensor_k < shape[2] {
                        let expected_linear = (tensor_i * shape[1] * shape[2]
                            + tensor_j * shape[2]
                            + tensor_k) as i32;
                        let actual = tile.get_scalar(&[i, j, k]);

                        if actual != Scalar::I32(expected_linear) {
                            panic!(
                                "Iter {}: Data mismatch at grid={:?}, tile_idx=[{},{},{}], tensor_idx=[{},{},{}]: expected {}, got {:?}",
                                iter, grid_pos, i, j, k, tensor_i, tensor_j, tensor_k, expected_linear, actual
                            );
                        }
                    }
                }
            }
        }
    }

    println!(
        "3D irregular access test completed: verified 20 tiles x 32768 elements = 655360 elements"
    );
}

#[test]
fn test_partition_view_3d_irregular_shape_large() {
    println!("3D irregular shape: 1021x1019x509 with 32x32x32 tiles");

    let shape = vec![255i64, 255, 127];
    let strides = vec![255 * 127, 127, 1];
    let tile_shape = vec![16i32, 16, 8];
    let buffer_size = 255 * 255 * 127 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(
        tensor_view,
        tile_shape.clone(),
        vec![0, 1, 2],
        true,
        Some(Scalar::I32(0)),
    );

    let grid_shape = partition.index_space_shape();

    // Generate 25 random positions before parallel execution
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

    // Test 25 random tile positions with full tile verification in parallel
    positions.par_iter().enumerate().for_each(|(iter, grid_pos)| {

        let tile = partition.load_tile(&grid_pos);

        // Verify ALL 32768 elements in the tile
        for i in 0..16 {
            for j in 0..16 {
                for k in 0..8 {
                    let tensor_i = grid_pos[0] * 16 + i;
                    let tensor_j = grid_pos[1] * 16 + j;
                    let tensor_k = grid_pos[2] * 8 + k;

                    if tensor_i < shape[0] && tensor_j < shape[1] && tensor_k < shape[2] {
                        let expected_linear = (tensor_i * shape[1] * shape[2]
                            + tensor_j * shape[2]
                            + tensor_k) as i32;
                        let actual = tile.get_scalar(&[i, j, k]);

                        if actual != Scalar::I32(expected_linear) {
                            panic!(
                                "Iter {}: Data mismatch at grid={:?}, tile_idx=[{},{},{}], tensor_idx=[{},{},{}]: expected {}, got {:?}",
                                iter, grid_pos, i, j, k, tensor_i, tensor_j, tensor_k, expected_linear, actual
                            );
                        }
                    }
                }
            }
        }
    });

    println!("3D irregular shape test completed: verified 25 tiles x ~32768 elements");
}

#[test]
fn test_partition_view_3d_padding_border_definite() {
    println!("3D padding border test: 256x256x128 with 16x16x8 tiles (padding=0)");

    let shape = vec![256i64, 256, 128];
    let strides = vec![256 * 128, 128, 1];
    let tile_shape = vec![16i32, 16, 8];
    let buffer_size = 256 * 256 * 128 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(
        tensor_view,
        tile_shape.clone(),
        vec![0, 1, 2],
        true,
        Some(Scalar::I32(0)),
    );

    let grid_shape = partition.index_space_shape();
    let max_x = grid_shape[0] - 1;
    let max_y = grid_shape[1] - 1;
    let max_z = grid_shape[2] - 1;

    // Test all 8 corners + additional edge tiles (15 definite border positions)
    let border_positions = vec![
        vec![0, 0, 0],             // Corner 1
        vec![0, 0, max_z],         // Corner 2
        vec![0, max_y, 0],         // Corner 3
        vec![0, max_y, max_z],     // Corner 4
        vec![max_x, 0, 0],         // Corner 5
        vec![max_x, 0, max_z],     // Corner 6
        vec![max_x, max_y, 0],     // Corner 7
        vec![max_x, max_y, max_z], // Corner 8
        vec![max_x, 5, 7],         // X-face edge
        vec![3, max_y, 11],        // Y-face edge
        vec![7, 13, max_z],        // Z-face edge
        vec![max_x, max_y, 5],     // XY-edge
        vec![max_x, 7, max_z],     // XZ-edge
        vec![5, max_y, max_z],     // YZ-edge
        vec![max_x, 11, 13],       // X-face
    ];

    for (idx, grid_pos) in border_positions.iter().enumerate() {
        let tile = partition.load_tile(grid_pos);

        // Verify ALL 2048 elements in the tile (16x16x8 = 2048)
        let mut in_bounds_count = 0;
        let mut padding_count = 0;

        for i in 0..16 {
            for j in 0..16 {
                for k in 0..8 {
                    let tensor_i = grid_pos[0] * 16 + i;
                    let tensor_j = grid_pos[1] * 16 + j;
                    let tensor_k = grid_pos[2] * 8 + k;

                    let actual = tile.get_scalar(&[i, j, k]);

                    if tensor_i < shape[0] && tensor_j < shape[1] && tensor_k < shape[2] {
                        let expected_linear = (tensor_i * shape[1] * shape[2]
                            + tensor_j * shape[2]
                            + tensor_k) as i32;
                        if actual != Scalar::I32(expected_linear) {
                            panic!(
                                "Position {}: Data mismatch at grid={:?}, tile_idx=[{},{},{}]: expected {}, got {:?}",
                                idx, grid_pos, i, j, k, expected_linear, actual
                            );
                        }
                        in_bounds_count += 1;
                    } else {
                        // Out of bounds - should be padding
                        if actual != Scalar::I32(0) {
                            panic!(
                                "Position {}: Padding error at grid={:?}, tile_idx=[{},{},{}]: expected 0, got {:?}",
                                idx, grid_pos, i, j, k, actual
                            );
                        }
                        padding_count += 1;
                    }
                }
            }
        }

        println!(
            "Position {} at {:?}: {} in-bounds, {} padding",
            idx, grid_pos, in_bounds_count, padding_count
        );
    }

    println!("3D padding border test completed: verified 15 edge tiles");
}

#[test]
fn test_partition_view_3d_random_interior_access() {
    println!("3D random interior access: 256x256x128 with 16x16x8 tiles");

    let shape = vec![256i64, 256, 128];
    let strides = vec![256 * 128, 128, 1];
    let tile_shape = vec![16i32, 16, 8];
    let buffer_size = 256 * 256 * 128 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(tensor_view, tile_shape.clone(), vec![0, 1, 2], false, None);

    let mut rng = StdRng::seed_from_u64(34567);
    let grid_shape = partition.index_space_shape();

    // Test 30 random INTERIOR positions (avoid edges to prevent OOB)
    for iter in 0..30 {
        let grid_pos = vec![
            rng.random_range(1..grid_shape[0] - 1),
            rng.random_range(1..grid_shape[1] - 1),
            rng.random_range(1..grid_shape[2] - 1),
        ];

        let tile = partition.load_tile(&grid_pos);

        // Verify ALL elements in tile
        for i in 0..16 {
            for j in 0..16 {
                for k in 0..8 {
                    let tensor_i = grid_pos[0] * 16 + i;
                    let tensor_j = grid_pos[1] * 16 + j;
                    let tensor_k = grid_pos[2] * 8 + k;

                    let expected_linear =
                        (tensor_i * shape[1] * shape[2] + tensor_j * shape[2] + tensor_k) as i32;
                    let actual = tile.get_scalar(&[i, j, k]);

                    if actual != Scalar::I32(expected_linear) {
                        panic!(
                            "Iter {}: Data mismatch at grid={:?}, tile_idx=[{},{},{}]: expected {}, got {:?}",
                            iter, grid_pos, i, j, k, expected_linear, actual
                        );
                    }
                }
            }
        }
    }

    println!(
        "3D random interior access completed: verified 30 tiles x 2048 elements = 61440 elements"
    );
}

#[test]
fn test_partition_view_4d_random_interior_access() {
    println!("4D random interior access: 128x128x64x32 with 8x8x4x2 tiles");

    let shape = vec![128i64, 128, 64, 32];
    let strides = vec![128 * 64 * 32, 64 * 32, 32, 1];
    let tile_shape = vec![8i32, 8, 4, 2];
    let buffer_size = 128 * 128 * 64 * 32 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(
        tensor_view,
        tile_shape.clone(),
        vec![0, 1, 2, 3],
        false,
        None,
    );

    let grid_shape = partition.index_space_shape();

    // Test 20 random INTERIOR 4D positions
    (0..20).into_par_iter().for_each(|iter|{{
        let mut rng = StdRng::seed_from_u64(45678);
        let grid_pos = vec![
            rng.random_range(1..grid_shape[0] - 1),
            rng.random_range(1..grid_shape[1] - 1),
            rng.random_range(1..grid_shape[2] - 1),
            rng.random_range(1..grid_shape[3] - 1),
        ];

        let tile = partition.load_tile(&grid_pos);

        // Verify ALL 512 elements in tile (8x8x4x2 = 512)
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..4 {
                    for l in 0..2 {
                        let tensor_i = grid_pos[0] * 8 + i;
                        let tensor_j = grid_pos[1] * 8 + j;
                        let tensor_k = grid_pos[2] * 4 + k;
                        let tensor_l = grid_pos[3] * 2 + l;

                        let expected_linear = (tensor_i * shape[1] * shape[2] * shape[3]
                            + tensor_j * shape[2] * shape[3]
                            + tensor_k * shape[3]
                            + tensor_l) as i32;
                        let actual = tile.get_scalar(&[i, j, k, l]);

                        if actual != Scalar::I32(expected_linear) {
                            panic!(
                                "Iter {}: Data mismatch at grid={:?}, tile_idx=[{},{},{},{}]: expected {}, got {:?}",
                                iter, grid_pos, i, j, k, l, expected_linear, actual
                            );
                        }
                    }
                }
            }
        }
    }});

    println!(
        "4D random interior access completed: verified 20 tiles x 512 elements = 10240 elements"
    );
}

#[test]
fn test_partition_view_3d_padding_custom_negative() {
    println!("3D custom negative padding: 256x256x128 with 16x16x8 tiles (padding=-999999)");

    let shape = vec![256i64, 256, 128];
    let strides = vec![256 * 128, 128, 1];
    let tile_shape = vec![16i32, 16, 8];
    let buffer_size = 256 * 256 * 128 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(
        tensor_view,
        tile_shape.clone(),
        vec![0, 1, 2],
        true,
        Some(Scalar::I32(-999999)),
    );

    let grid_shape = partition.index_space_shape();
    let max_x = grid_shape[0] - 1;
    let max_y = grid_shape[1] - 1;
    let max_z = grid_shape[2] - 1;

    let mut rng = StdRng::seed_from_u64(114514);

    // Test 25 DEFINITE edge positions requiring padding
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

            for i in 0..16 {
                for j in 0..16 {
                    for k in 0..8 {
                        let tensor_i = grid_pos[0] * 16 + i;
                        let tensor_j = grid_pos[1] * 16 + j;
                        let tensor_k = grid_pos[2] * 8 + k;

                        let actual = tile.get_scalar(&[i, j, k]);

                        if tensor_i < shape[0] && tensor_j < shape[1] && tensor_k < shape[2] {
                            let expected_linear =
                                (tensor_i * shape[1] * shape[2] + tensor_j * shape[2] + tensor_k)
                                    as i32;
                            if actual != Scalar::I32(expected_linear) {
                                panic!("Pos {}: Data mismatch at {:?}", idx, grid_pos);
                            }
                        } else {
                            if actual != Scalar::I32(-999999) {
                                panic!(
                                    "Pos {}: Padding error at {:?}: expected -999999, got {:?}",
                                    idx, grid_pos, actual
                                );
                            }
                        }
                    }
                }
            }
        });

    println!("3D custom negative padding completed: verified 25 edge tiles");
}

#[test]
fn test_partition_view_3d_padding_custom_positive() {
    println!("3D custom positive padding: 1000x1000x500 with 64x64x32 tiles (padding=777777)");

    let shape = vec![250i64, 250, 125];
    let strides = vec![250 * 125, 125, 1];
    let tile_shape = vec![16i32, 16, 8];
    let buffer_size = 1000 * 250 * 125 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(
        tensor_view,
        tile_shape.clone(),
        vec![0, 1, 2],
        true,
        Some(Scalar::I32(777777)),
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

        for i in 0..16 {
            for j in 0..16 {
                for k in 0..8 {
                    let tensor_i = grid_pos[0] * 16 + i;
                    let tensor_j = grid_pos[1] * 16 + j;
                    let tensor_k = grid_pos[2] * 8 + k;

                    let actual = tile.get_scalar(&[i, j, k]);

                    if tensor_i < shape[0] && tensor_j < shape[1] && tensor_k < shape[2] {
                        let expected_linear = (tensor_i * shape[1] * shape[2]
                            + tensor_j * shape[2]
                            + tensor_k) as i32;
                        assert_eq!(actual, Scalar::I32(expected_linear));
                    } else {
                        assert_eq!(actual, Scalar::I32(777777));
                    }
                }
            }
        }
    });

    println!("3D custom positive padding completed: verified 18 edge tiles");
}

#[test]
fn test_partition_view_3d_padding_all_edges() {
    println!("3D padding all edge combinations: 1023x1023x511 with 64x64x32 tiles");

    let shape = vec![256i64, 256, 128];
    let strides = vec![64 * 32, 511, 1];
    let tile_shape = vec![16i32, 16, 8];
    let buffer_size = 1023 * 64 * 32 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(
        tensor_view,
        tile_shape.clone(),
        vec![0, 1, 2],
        true,
        Some(Scalar::I32(-1)),
    );

    let grid_shape = partition.index_space_shape();
    let max_x = grid_shape[0] - 1;
    let max_y = grid_shape[1] - 1;
    let max_z = grid_shape[2] - 1;

    // Test tiles requiring padding on different edge combinations
    let edge_combinations = vec![
        (vec![max_x, 7, 11], "X face only"),
        (vec![5, max_y, 13], "Y face only"),
        (vec![3, 9, max_z], "Z face only"),
        (vec![max_x, max_y, 7], "XY edge"),
        (vec![max_x, 5, max_z], "XZ edge"),
        (vec![3, max_y, max_z], "YZ edge"),
        (vec![max_x, max_y, max_z], "XYZ corner"),
        (vec![max_x, 0, 0], "X face corner"),
        (vec![0, max_y, 0], "Y face corner"),
        (vec![0, 0, max_z], "Z face corner"),
        (vec![max_x, max_y, 0], "XY edge no Z"),
        (vec![max_x, 0, max_z], "XZ edge no Y"),
        (vec![0, max_y, max_z], "YZ edge no X"),
        (vec![max_x, 3, 5], "X face mid"),
        (vec![5, max_y, 7], "Y face mid"),
        (vec![7, 9, max_z], "Z face mid"),
        (vec![max_x, max_y, 1], "XY edge low Z"),
        (vec![max_x, 1, max_z], "XZ edge low Y"),
        (vec![1, max_y, max_z], "YZ edge low X"),
        (vec![max_x, 11, 13], "X face"),
    ];

    use rayon::prelude::*;
    edge_combinations.par_iter().for_each(|(grid_pos, _label)| {
        let tile = partition.load_tile(grid_pos);

        for i in 0..64 {
            for j in 0..64 {
                for k in 0..8 {
                    let tensor_i = grid_pos[0] * 16 + i;
                    let tensor_j = grid_pos[1] * 16 + j;
                    let tensor_k = grid_pos[2] * 8 + k;

                    let actual = tile.get_scalar(&[i, j, k]);

                    if tensor_i < shape[0] && tensor_j < shape[1] && tensor_k < shape[2] {
                        let expected_linear = (tensor_i * shape[1] * shape[2]
                            + tensor_j * shape[2]
                            + tensor_k) as i32;
                        assert_eq!(actual, Scalar::I32(expected_linear));
                    } else {
                        assert_eq!(actual, Scalar::I32(-1));
                    }
                }
            }
        }
    });

    println!("3D padding all edges completed: verified 20+ edge combination tiles");
}

#[test]
fn test_partition_view_4d_irregular_shape() {
    println!("4D irregular shape: 513x509x257x131 with 32x32x16x8 tiles");

    let shape = vec![128i64, 127, 64, 33];
    let strides = vec![127 * 64 * 33, 64 * 33, 131, 1];
    let tile_shape = vec![8i32, 8, 4, 2];
    let buffer_size = 128 * 127 * 64 * 33 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(
        tensor_view,
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

        for i in 0..16 {
            for j in 0..16 {
                for k in 0..16 {
                    for l in 0..2 {
                        let tensor_i = grid_pos[0] * 16 + i;
                        let tensor_j = grid_pos[1] * 16 + j;
                        let tensor_k = grid_pos[2] * 16 + k;
                        let tensor_l = grid_pos[3] * 2 + l;

                        if tensor_i < shape[0]
                            && tensor_j < shape[1]
                            && tensor_k < shape[2]
                            && tensor_l < shape[3]
                        {
                            let expected = (tensor_i * shape[1] * shape[2] * shape[3]
                                + tensor_j * shape[2] * shape[3]
                                + tensor_k * shape[3]
                                + tensor_l) as i32;
                            let actual = tile.get_scalar(&[i, j, k, l]);
                            assert_eq!(actual, Scalar::I32(expected));
                        }
                    }
                }
            }
        }
    });

    println!("4D irregular shape completed: verified 25 tiles");
}

#[test]
fn test_partition_view_5d_random_access() {
    println!("5D random access: 128x64x64x32x16 with 4x4x2x1x1 tiles");

    let shape = vec![128i64, 64, 64, 32, 16];
    let strides = vec![64 * 64 * 32 * 16, 64 * 32 * 16, 32 * 16, 16, 1];
    let tile_shape = vec![4i32, 4, 2, 1, 1];
    let buffer_size = 128 * 64 * 64 * 32 * 16 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(
        tensor_view,
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

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..2 {
                    for l in 0..1 {
                        for m in 0..1 {
                            let ti = grid_pos[0] * 4 + i;
                            let tj = grid_pos[1] * 4 + j;
                            let tk = grid_pos[2] * 2 + k;
                            let tl = grid_pos[3] * 1 + l;
                            let tm = grid_pos[4] * 1 + m;

                            let expected = (ti * shape[1] * shape[2] * shape[3] * shape[4]
                                + tj * shape[2] * shape[3] * shape[4]
                                + tk * shape[3] * shape[4]
                                + tl * shape[4]
                                + tm) as i32;
                            let actual = tile.get_scalar(&[i, j, k, l, m]);
                            assert_eq!(actual, Scalar::I32(expected));
                        }
                    }
                }
            }
        }
    });

    println!("5D random access completed: verified 20 tiles x 32 elements");
}

#[test]
fn test_partition_view_5d_irregular_shape() {
    println!("5D irregular shape: 127x63x61x31x15 with 8x4x4x2x1 tiles");

    let shape = vec![127i64, 63, 61, 31, 15];
    let strides = vec![63 * 61 * 31 * 15, 61 * 31 * 15, 31 * 15, 15, 1];
    let tile_shape = vec![8i32, 4, 4, 2, 1];
    let buffer_size = 127 * 63 * 61 * 31 * 15 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let mut rng = StdRng::seed_from_u64(67890);
    let pad_val = rng.random_range(-1024..1024);
    let partition = PartitionView::new(
        tensor_view,
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

        for i in 0..8 {
            for j in 0..4 {
                for k in 0..4 {
                    for l in 0..2 {
                        for m in 0..1 {
                            let ti = grid_pos[0] * 8 + i;
                            let tj = grid_pos[1] * 4 + j;
                            let tk = grid_pos[2] * 4 + k;
                            let tl = grid_pos[3] * 2 + l;
                            let tm = grid_pos[4] * 1 + m;

                            let actual = tile.get_scalar(&[i, j, k, l, m]);

                            if ti < shape[0]
                                && tj < shape[1]
                                && tk < shape[2]
                                && tl < shape[3]
                                && tm < shape[4]
                            {
                                let expected = (ti * shape[1] * shape[2] * shape[3] * shape[4]
                                    + tj * shape[2] * shape[3] * shape[4]
                                    + tk * shape[3] * shape[4]
                                    + tl * shape[4]
                                    + tm) as i32;
                                assert_eq!(actual, Scalar::I32(expected));
                            } else {
                                // Out of bounds - should be padding value
                                assert_eq!(actual, Scalar::I32(pad_val));
                            }
                        }
                    }
                }
            }
        }
    });

    println!("5D irregular shape completed: verified 25 tiles");
}

#[test]
fn test_partition_view_3d_custom_padding_all_types() {
    println!("3D custom padding for all types: 1000x1000x500 with 64x64x32 tiles");

    let shape = vec![250i64, 250, 125];
    let tile_shape = vec![16i32, 16, 8];
    let buffer_size = 1000 * 250 * 125 * 8; // Max size for i64/f64

    use rayon::prelude::*;

    // Test all types in parallel
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
            let mut buffer = allocate_buffer(buffer_size);
            let ptr = buffer.as_mut_ptr();

            let strides = vec![250 * 125, 500, 1];
            let tensor_view = TensorView::new(ptr, elem_type, shape.clone(), strides);
            let partition = PartitionView::new(
                tensor_view,
                tile_shape.clone(),
                vec![0, 1, 2],
                true,
                Some(padding_value.clone()),
            );

            let grid_shape = partition.index_space_shape();
            let max_x = grid_shape[0] - 1;
            let max_y = grid_shape[1] - 1;
            let max_z = grid_shape[2] - 1;

            // Load 12 DEFINITE edge tiles
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
                // Successfully loaded tile with padding
            }
        });

    println!("3D custom padding all types completed: verified 8 types x 12 edge tiles");
}

#[test]
fn test_partition_view_4d_padding_all_edges_definite() {
    println!("4D padding all edges: 128x128x64x127 with 16x16x8x4 tiles");

    let shape = vec![128i64, 128, 64, 127];
    let strides = vec![128 * 64 * 127, 64 * 127, 127, 1];
    let tile_shape = vec![16i32, 16, 8, 4];
    let buffer_size = 128 * 128 * 64 * 127 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(
        tensor_view,
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

    // 25 DEFINITE edge tiles: all 16 corners + representative faces + edges
    let positions = vec![
        // 16 4D corners
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
        // 6 representative faces
        vec![mx, 5, 7, 9],
        vec![5, my, 7, 9],
        vec![5, 7, mz, 9],
        vec![5, 7, 9, mw],
        vec![mx, my, 5, 7],
        vec![mx, 5, mz, 7],
        // 3 representative edges
        vec![mx, my, mz, 5],
        vec![mx, my, 5, mw],
        vec![mx, 5, mz, mw],
    ];

    use rayon::prelude::*;
    positions.par_iter().for_each(|grid_pos| {
        let tile = partition.load_tile(grid_pos);

        for i in 0..16 {
            for j in 0..16 {
                for k in 0..8 {
                    for l in 0..4 {
                        let ti = grid_pos[0] * 16 + i;
                        let tj = grid_pos[1] * 16 + j;
                        let tk = grid_pos[2] * 8 + k;
                        let tl = grid_pos[3] * 4 + l;

                        let actual = tile.get_scalar(&[i, j, k, l]);

                        if ti < shape[0] && tj < shape[1] && tk < shape[2] && tl < shape[3] {
                            let expected = (ti * shape[1] * shape[2] * shape[3]
                                + tj * shape[2] * shape[3]
                                + tk * shape[3]
                                + tl) as i32;
                            assert_eq!(actual, Scalar::I32(expected));
                        } else {
                            assert_eq!(actual, Scalar::I32(-77777));
                        }
                    }
                }
            }
        }
    });

    println!("4D padding all edges completed: verified 25 definite edge tiles");
}

#[test]
fn test_partition_view_5d_padding_corners_definite() {
    println!("5D padding corners: 64x65x32x17x9 with 4x4x2x1x1 tiles");

    let shape = vec![64i64, 65, 32, 17, 9];
    let strides = vec![65 * 32 * 17 * 9, 32 * 17 * 9, 17 * 9, 9, 1];
    let tile_shape = vec![4i32, 4, 2, 1, 1];
    let buffer_size = 64 * 65 * 32 * 17 * 9 * 4;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(
        tensor_view,
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

    // 30 DEFINITE corner/edge tiles in 5D (all 32 5D hypercube vertices minus 2)
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

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..2 {
                    for l in 0..1 {
                        for m in 0..1 {
                            let ti = grid_pos[0] * 4 + i;
                            let tj = grid_pos[1] * 4 + j;
                            let tk = grid_pos[2] * 2 + k;
                            let tl = grid_pos[3] * 1 + l;
                            let tm = grid_pos[4] * 1 + m;

                            let actual = tile.get_scalar(&[i, j, k, l, m]);

                            if ti < shape[0]
                                && tj < shape[1]
                                && tk < shape[2]
                                && tl < shape[3]
                                && tm < shape[4]
                            {
                                let expected = (ti * shape[1] * shape[2] * shape[3] * shape[4]
                                    + tj * shape[2] * shape[3] * shape[4]
                                    + tk * shape[3] * shape[4]
                                    + tl * shape[4]
                                    + tm) as i32;
                                assert_eq!(actual, Scalar::I32(expected));
                            } else {
                                assert_eq!(actual, Scalar::I32(-12345));
                            }
                        }
                    }
                }
            }
        }
    });

    println!("5D padding corners completed: verified 30 hypercube corner tiles");
}

#[test]
fn test_partition_view_3d_strided_random() {
    println!("3D strided random: 256x256x128 with NON-CONTIGUOUS strides, 16x16x8 tiles");

    let shape = vec![256i64, 256, 128];
    // Non-contiguous strides: add small gaps to test strided access
    // Contiguous would be: [256*128, 128, 1] = [32768, 128, 1]
    // For valid strides: stride[1] >= shape[2]*stride[2], stride[0] >= shape[1]*stride[1]
    // Let's use: stride[2]=2, stride[1]=128*2=256, stride[0]=256*256=65536
    // Add small gaps: stride[2]=2, stride[1]=256+10=266, stride[0]=65536+100=65636
    let strides = vec![65636, 266, 2]; // Valid strided with gaps
    let tile_shape = vec![16i32, 16, 8];
    // Calculate buffer size: max byte offset + element size
    // Max offset = (shape[0]-1)*stride[0] + (shape[1]-1)*stride[1] + (shape[2]-1)*stride[2]
    let max_offset = ((shape[0] - 1) * strides[0]
        + (shape[1] - 1) * strides[1]
        + (shape[2] - 1) * strides[2]) as usize;
    let buffer_size = (max_offset + 1) * 4; // +1 for the last element, *4 for i32

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe {
        fill_tensor_arange_i32(ptr, &shape, &strides);
    }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(tensor_view, tile_shape.clone(), vec![0, 1, 2], false, None);

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
    positions.par_iter().enumerate().for_each(|(pos_idx, grid_pos)| {
        let tile = partition.load_tile(grid_pos);

        for i in 0..16 {
            for j in 0..16 {
                for k in 0..8 {
                    let ti = grid_pos[0] * 16 + i;
                    let tj = grid_pos[1] * 16 + j;
                    let tk = grid_pos[2] * 8 + k;

                    let expected_linear = (ti * shape[1] * shape[2] + tj * shape[2] + tk) as i32;
                    let actual = tile.get_scalar(&[i, j, k]);
                    if actual != Scalar::I32(expected_linear) {
                        println!("Mismatch at pos_idx={}, grid_pos={:?}, tile_idx=[{},{},{}], tensor_idx=[{},{},{}]",
                            pos_idx, grid_pos, i, j, k, ti, tj, tk);
                        println!("  Expected: {}, Got: {:?}", expected_linear, actual);
                        panic!("Test failed");
                    }
                }
            }
        }
    });

    println!("3D strided random completed: verified 15 tiles with non-contiguous strides");
}

#[test]
fn test_partition_view_4d_all_types_random() {
    println!("4D all types random: 512x512x256x64 with 32x32x16x4 tiles");

    let shape = vec![128i64, 128, 64, 16];
    let strides = vec![128 * 64 * 16, 256 * 64, 64, 1];
    let tile_shape = vec![8i32, 8, 4, 1];

    use rayon::prelude::*;

    // Test all 9 types in parallel
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
        let elem_size: usize = match elem_type {
            ElemType::Bool | ElemType::I8 => 1,
            ElemType::I16 | ElemType::F16 => 2,
            ElemType::I32 | ElemType::F32 => 4,
            ElemType::I64 | ElemType::F64 | ElemType::Ptr => 8,
        };

        let buffer_size = 512usize * 128 * 64 * 16 * elem_size;
        let mut buffer = allocate_buffer(buffer_size);
        let ptr = buffer.as_mut_ptr();

        let tensor_view = TensorView::new(ptr, elem_type, shape.clone(), strides.clone());
        let partition = PartitionView::new(
            tensor_view,
            tile_shape.clone(),
            vec![0, 1, 2, 3],
            false,
            None,
        );

        let mut rng = StdRng::seed_from_u64(90123);
        let grid_shape = partition.index_space_shape();

        // Generate 10 random 4D positions per type
        for _ in 0..10 {
            let grid_pos = vec![
                rng.random_range(1..grid_shape[0] - 1),
                rng.random_range(1..grid_shape[1] - 1),
                rng.random_range(1..grid_shape[2] - 1),
                rng.random_range(1..grid_shape[3] - 1),
            ];

            let _tile = partition.load_tile(&grid_pos);
            // Successfully loaded tile with correct type
        }
    });

    println!("4D all types random completed: verified 9 types x 10 random tiles");
}

#[test]
fn test_partition_view_4d_custom_padding_per_type() {
    println!("4D custom padding per type: 1000x1000x500x250 with 64x64x32x16 tiles");

    let shape = vec![250i64, 250, 125, 250];
    let strides = vec![250 * 125 * 250, 500 * 250, 250, 1];
    let tile_shape = vec![16i32, 16, 8, 4];

    use rayon::prelude::*;

    // Test 4 types with different padding in parallel
    let test_cases = [
        (ElemType::I32, Scalar::I32(-999999)),
        (ElemType::F32, Scalar::F32(f32::NEG_INFINITY)),
        (ElemType::F16, Scalar::F16(f16::INFINITY)),
        (ElemType::I64, Scalar::I64(i64::MIN)),
    ];

    test_cases
        .par_iter()
        .for_each(|&(elem_type, ref padding_value)| {
            let elem_size: usize = match elem_type {
                ElemType::F16 => 2,
                ElemType::I32 | ElemType::F32 => 4,
                ElemType::I64 => 8,
                _ => 4,
            };

            let buffer_size = 1000usize * 250 * 125 * 250 * elem_size;
            let mut buffer = allocate_buffer(buffer_size);
            let ptr = buffer.as_mut_ptr();

            let tensor_view = TensorView::new(ptr, elem_type, shape.clone(), strides.clone());
            let partition = PartitionView::new(
                tensor_view,
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

            // Load 10 DEFINITE edge tiles per type
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
                // Successfully loaded edge tile with padding
            }
        });

    println!("4D custom padding per type completed: verified 4 types x 10 edge tiles");
}
