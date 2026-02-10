#!/usr/bin/env python3
"""Generate high-quality partition view tests with comprehensive data verification."""

def generate_header():
    return '''use crate::interpreter::data_structures::{
    elem_type::{ElemType, Scalar},
    tensor_view::{PartitionView, TensorView},
    tile::Tile,
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Helper to allocate a buffer for testing
fn allocate_buffer(size_bytes: usize) -> Vec<u8> {
    vec![0u8; size_bytes]
}

/// Fill tensor with arange pattern: tensor[indices] = linear_index
unsafe fn fill_tensor_arange_i32(ptr: *mut u8, shape: &[i64], strides: &[i64]) {
    let rank = shape.len();
    let total_elements: usize = shape.iter().product::<i64>() as usize;

    for linear_idx in 0..total_elements {
        let mut indices = vec![0i64; rank];
        let mut remaining = linear_idx;

        // Convert linear index to multi-dimensional indices
        for dim in (0..rank).rev() {
            let dim_size = shape[dim] as usize;
            indices[dim] = (remaining % dim_size) as i64;
            remaining /= dim_size;
        }

        // Calculate byte offset using strides
        let mut byte_offset = 0usize;
        for dim in 0..rank {
            byte_offset += (indices[dim] * strides[dim]) as usize * 4; // i32 = 4 bytes
        }

        *(ptr.add(byte_offset) as *mut i32) = linear_idx as i32;
    }
}

/// Verify tile contains expected arange values
fn verify_tile_arange(
    tile: &Tile,
    tile_grid_pos: &[i64],
    tile_shape: &[i32],
    tensor_shape: &[i64],
    dim_map: &[i32],
) {
    let rank = tile_shape.len();
    let tile_size: usize = tile_shape.iter().map(|&x| x as usize).product();

    for tile_linear in 0..tile_size {
        let mut tile_indices = vec![0i64; rank];
        let mut remaining = tile_linear;

        for dim in (0..rank).rev() {
            let dim_size = tile_shape[dim] as usize;
            tile_indices[dim] = (remaining % dim_size) as i64;
            remaining /= dim_size;
        }

        // Map tile indices to tensor indices using dim_map
        let mut tensor_indices = vec![0i64; rank];
        for tile_dim in 0..rank {
            let tensor_dim = dim_map[tile_dim] as usize;
            tensor_indices[tensor_dim] = tile_grid_pos[tile_dim] * tile_shape[tile_dim] as i64 + tile_indices[tile_dim];
        }

        // Check if within tensor bounds
        let mut in_bounds = true;
        for dim in 0..rank {
            if tensor_indices[dim] >= tensor_shape[dim] {
                in_bounds = false;
                break;
            }
        }

        if in_bounds {
            // Calculate expected linear index in tensor
            let mut expected_linear = 0usize;
            let mut multiplier = 1usize;
            for dim in (0..rank).rev() {
                expected_linear += tensor_indices[dim] as usize * multiplier;
                multiplier *= tensor_shape[dim] as usize;
            }

            let actual = tile.get_scalar(&tile_indices);
            let expected = Scalar::I32(expected_linear as i32);

            if actual != expected {
                panic!(
                    "Data mismatch at tile_pos={:?}, tile_idx={:?}, tensor_idx={:?}: expected {:?}, got {:?}",
                    tile_grid_pos, tile_indices, tensor_indices, expected, actual
                );
            }
        }
    }
}

'''

def generate_3d_permutation_tests():
    """Keep the existing 6 3D permutation tests."""
    return '''// ============================================================================
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
    let partition = PartitionView::new(
        tensor_view,
        vec![4, 4, 4],
        vec![0, 1, 2],
        false,
        None,
    );

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
    let partition = PartitionView::new(
        tensor_view,
        vec![4, 4, 4],
        vec![0, 2, 1],
        false,
        None,
    );

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
    let partition = PartitionView::new(
        tensor_view,
        vec![4, 4, 4],
        vec![1, 0, 2],
        false,
        None,
    );

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
    let partition = PartitionView::new(
        tensor_view,
        vec![4, 4, 4],
        vec![1, 2, 0],
        false,
        None,
    );

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
    let partition = PartitionView::new(
        tensor_view,
        vec![4, 4, 4],
        vec![2, 0, 1],
        false,
        None,
    );

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
    let partition = PartitionView::new(
        tensor_view,
        vec![4, 4, 4],
        vec![2, 1, 0],
        false,
        None,
    );

    let tile = partition.load_tile(&[0, 0, 0]);

    assert_eq!(tile.get_scalar(&[0, 0, 0]), Scalar::I32(0));
    assert_eq!(tile.get_scalar(&[1, 0, 0]), Scalar::I32(1));
    assert_eq!(tile.get_scalar(&[0, 1, 0]), Scalar::I32(10));
    assert_eq!(tile.get_scalar(&[0, 0, 1]), Scalar::I32(100));
}

'''

def generate_4d_permutation_tests():
    """Keep the existing 3 4D permutation tests."""
    return '''// ============================================================================
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
    let partition = PartitionView::new(
        tensor_view,
        vec![4, 4, 4, 2],
        vec![0, 1, 2, 3],
        false,
        None,
    );

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
    let partition = PartitionView::new(
        tensor_view,
        vec![4, 4, 4, 2],
        vec![3, 2, 1, 0],
        false,
        None,
    );

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
    let partition = PartitionView::new(
        tensor_view,
        vec![4, 4, 4, 2],
        vec![0, 1, 3, 2],
        false,
        None,
    );

    let tile = partition.load_tile(&[0, 0, 0, 0]);
    assert_eq!(tile.get_scalar(&[0, 0, 0, 0]), Scalar::I32(0));
    assert_eq!(tile.get_scalar(&[0, 0, 1, 0]), Scalar::I32(1));
    assert_eq!(tile.get_scalar(&[0, 0, 0, 1]), Scalar::I32(10));
}

'''

def generate_new_tests():
    """Generate new comprehensive tests."""
    return '''// ============================================================================
// NEW: Large-Scale 3D Random Access Tests
// ============================================================================

#[test]
fn test_partition_view_3d_irregular_tile_access_large() {
    println!("3D irregular access: 1024x1024x512 with 32x32x32 tiles");

    let shape = vec![1024i64, 1024, 512];
    let strides = vec![1024 * 512, 512, 1];
    let tile_shape = vec![32i32, 32, 32];
    let buffer_size = (1024 * 1024 * 512 * 4) as usize;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe { fill_tensor_arange_i32(ptr, &shape, &strides); }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(tensor_view, tile_shape.clone(), vec![0, 1, 2], false, None);

    let mut rng = StdRng::seed_from_u64(12345);
    let grid_shape = partition.index_space_shape();

    for _ in 0..20 {
        let grid_pos = vec![
            rng.gen_range(0..grid_shape[0]),
            rng.gen_range(0..grid_shape[1]),
            rng.gen_range(0..grid_shape[2]),
        ];

        let tile = partition.load_tile(&grid_pos);
        verify_tile_arange(&tile, &grid_pos, &tile_shape, &shape, &[0, 1, 2]);
    }

    println!("3D irregular access test completed");
}

#[test]
fn test_partition_view_3d_irregular_shape_large() {
    println!("3D irregular shape: 1021x1019x509 with 32x32x32 tiles");

    let shape = vec![1021i64, 1019, 509];
    let strides = vec![1019 * 509, 509, 1];
    let tile_shape = vec![32i32, 32, 32];
    let buffer_size = (1021 * 1019 * 509 * 4) as usize;

    let mut buffer = allocate_buffer(buffer_size);
    let ptr = buffer.as_mut_ptr();

    unsafe { fill_tensor_arange_i32(ptr, &shape, &strides); }

    let tensor_view = TensorView::new(ptr, ElemType::I32, shape.clone(), strides);
    let partition = PartitionView::new(tensor_view, tile_shape.clone(), vec![0, 1, 2], false, None);

    let mut rng = StdRng::seed_from_u64(23456);
    let grid_shape = partition.index_space_shape();

    for _ in 0..25 {
        let grid_pos = vec![
            rng.gen_range(0..grid_shape[0]),
            rng.gen_range(0..grid_shape[1]),
            rng.gen_range(0..grid_shape[2]),
        ];

        let tile = partition.load_tile(&grid_pos);
        verify_tile_arange(&tile, &grid_pos, &tile_shape, &shape, &[0, 1, 2]);
    }

    println!("3D irregular shape test completed");
}

'''

# Generate the complete file
output = generate_header()
output += generate_3d_permutation_tests()
output += generate_4d_permutation_tests()
output += generate_new_tests()

print(output)
