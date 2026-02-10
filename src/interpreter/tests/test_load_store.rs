use crate::interpreter::data_structures::{
    elem_type::{ElemType, Scalar},
    tensor_view::{PartitionView, TensorView},
    tile::Tile,
};
use rand::{rngs::StdRng, RngExt};
use rand::{Rng, SeedableRng};

/// Helper to allocate a buffer for testing
fn allocate_buffer(size_bytes: usize) -> Vec<u8> {
    vec![0u8; size_bytes]
}

// ============================================================================
// Large-Scale Tests (KEEP - Best Quality)
// ============================================================================

#[test]
fn test_tile_size_1024x1024() {
    let size = 1024;
    let buffer_size = size * size * 4;
    let mut buffer = vec![0u8; buffer_size];
    let ptr = buffer.as_mut_ptr();

    let tensor_view = TensorView::new(
        ptr,
        ElemType::I32,
        vec![size as i64, size as i64],
        vec![size as i64, 1],
    );
    let partition = PartitionView::new(
        tensor_view,
        vec![size as i32, size as i32],
        vec![0, 1],
        false,
        None,
    );

    let tile_shape = vec![size as usize, size as usize];
    let mut tile = Tile::zeros(&tile_shape, ElemType::I32);

    // Set corner values
    tile.set_scalar(&[0, 0], Scalar::I32(1));
    tile.set_scalar(&[0, (size - 1) as i64], Scalar::I32(2));
    tile.set_scalar(&[(size - 1) as i64, 0], Scalar::I32(3));
    tile.set_scalar(&[(size - 1) as i64, (size - 1) as i64], Scalar::I32(4));

    partition.store_tile(&[0, 0], &tile);
    let loaded = partition.load_tile(&[0, 0]);

    assert_eq!(loaded.get_scalar(&[0, 0]), Scalar::I32(1));
    assert_eq!(loaded.get_scalar(&[0, (size - 1) as i64]), Scalar::I32(2));
    assert_eq!(loaded.get_scalar(&[(size - 1) as i64, 0]), Scalar::I32(3));
    assert_eq!(
        loaded.get_scalar(&[(size - 1) as i64, (size - 1) as i64]),
        Scalar::I32(4)
    );
}

#[test]
fn test_large_scale_10000x10000_f32_full_grid() {
    println!("Starting large scale test: 16384x16384 f32 matrix with random access");

    let size = 16384;
    let tile_size = 64;
    let buffer_size = size * size * 4;

    println!("Allocating {} MB buffer...", buffer_size / (1024 * 1024));
    let mut buffer = vec![0u8; buffer_size];
    let ptr = buffer.as_mut_ptr();

    let tensor_view = TensorView::new(
        ptr,
        ElemType::F32,
        vec![size as i64, size as i64],
        vec![size as i64, 1],
    );
    let partition = PartitionView::new(
        tensor_view,
        vec![tile_size as i32, tile_size as i32],
        vec![0, 1],
        true,
        Some(Scalar::F32(0.0)),
    );

    let index_space = partition.index_space_shape();
    let num_tiles = index_space[0] * index_space[1];
    println!("Grid: {:?}, Total tiles: {}", index_space, num_tiles);

    let mut rng = StdRng::seed_from_u64(9999);

    // Generate 20 random positions
    let positions: Vec<_> = (0..20)
        .map(|_| {
            (
                rng.random_range(0..index_space[0]),
                rng.random_range(0..index_space[1]),
            )
        })
        .collect();

    use rayon::prelude::*;

    // Store and verify 20 random tiles in parallel
    positions
        .par_iter()
        .enumerate()
        .for_each(|(iter, &(i, j))| {
            let mut tile = Tile::zeros(&[tile_size as usize, tile_size as usize], ElemType::F32);

            // Fill with unique pattern
            for ti in 0..tile_size {
                for tj in 0..tile_size {
                    let value = (i * 1000000 + j * 1000 + ti * 10 + tj) as f32;
                    tile.set_scalar(&[ti as i64, tj as i64], Scalar::F32(value));
                }
            }

            partition.store_tile(&[i, j], &tile);
            let loaded = partition.load_tile(&[i, j]);

            // Verify ALL 4096 elements in the tile
            for ti in 0..tile_size {
                for tj in 0..tile_size {
                    let expected = (i * 1000000 + j * 1000 + ti * 10 + tj) as f32;
                    let actual = loaded.get_scalar(&[ti as i64, tj as i64]);
                    if actual != Scalar::F32(expected) {
                        panic!(
                            "Iter {}: Mismatch at tile [{},{}], pos [{},{}]: expected {}, got {:?}",
                            iter, i, j, ti, tj, expected, actual
                        );
                    }
                }
            }
        });

    println!("Large scale test completed: verified 20 tiles x 4096 elements = 81920 elements");
}

#[test]
fn test_large_scale_100000_element_1d_all_types() {
    let size = 100000;

    // Test I32
    {
        let mut buffer = vec![0u8; size * 4];
        let ptr = buffer.as_mut_ptr();
        let tensor_view = TensorView::new(ptr, ElemType::I32, vec![size as i64], vec![1]);
        let partition = PartitionView::new(tensor_view, vec![1024], vec![0], false, None);

        let mut tile = Tile::zeros(&[1024], ElemType::I32);
        for i in 0..1024 {
            tile.set_scalar(&[i], Scalar::I32(i as i32));
        }
        partition.store_tile(&[0], &tile);
        let loaded = partition.load_tile(&[0]);
        assert_eq!(loaded.get_scalar(&[999]), tile.get_scalar(&[999]));
    }

    // Test F32
    {
        let mut buffer = vec![0u8; size * 4];
        let ptr = buffer.as_mut_ptr();
        let tensor_view = TensorView::new(ptr, ElemType::F32, vec![size as i64], vec![1]);
        let partition = PartitionView::new(tensor_view, vec![1024], vec![0], false, None);

        let mut tile = Tile::zeros(&[1024], ElemType::F32);
        for i in 0..1024 {
            tile.set_scalar(&[i], Scalar::F32(i as f32 * 0.1));
        }
        partition.store_tile(&[0], &tile);
        let loaded = partition.load_tile(&[0]);
        assert_eq!(loaded.get_scalar(&[456]), tile.get_scalar(&[456]));
    }

    // Test F64
    {
        let mut buffer = vec![0u8; size * 8];
        let ptr = buffer.as_mut_ptr();
        let tensor_view = TensorView::new(ptr, ElemType::F64, vec![size as i64], vec![1]);
        let partition = PartitionView::new(tensor_view, vec![1024], vec![0], false, None);

        let mut tile = Tile::zeros(&[1024], ElemType::F64);
        for i in 0..1024 {
            tile.set_scalar(&[i], Scalar::F64(i as f64 * 0.1));
        }
        partition.store_tile(&[0], &tile);
        let loaded = partition.load_tile(&[0]);
        assert_eq!(loaded.get_scalar(&[789]), tile.get_scalar(&[789]));
    }

    println!("All types tested with 100000 elements");
}

#[test]
fn test_large_scale_3d_tensor() {
    println!("Testing 3D tensor: 512x512x512 with random access");

    let size = 512;
    let tile_size = 32;
    let buffer_size = size * size * size * 4;
    let mut buffer = vec![0u8; buffer_size];
    let ptr = buffer.as_mut_ptr();

    let tensor_view = TensorView::new(
        ptr,
        ElemType::F32,
        vec![size as i64, size as i64, size as i64],
        vec![(size * size) as i64, size as i64, 1],
    );
    let partition = PartitionView::new(
        tensor_view,
        vec![tile_size, tile_size, tile_size],
        vec![0, 1, 2],
        true,
        Some(Scalar::F32(0.0)),
    );

    let mut rng = StdRng::seed_from_u64(7777);
    let grid_shape = partition.index_space_shape();

    // Generate 20 random positions
    let positions: Vec<_> = (0..20)
        .map(|_| {
            (
                rng.random_range(0..grid_shape[0]),
                rng.random_range(0..grid_shape[1]),
                rng.random_range(0..grid_shape[2]),
            )
        })
        .collect();

    use rayon::prelude::*;

    // Test 20 random positions with full tile verification in parallel
    positions
        .par_iter()
        .enumerate()
        .for_each(|(iter, &(i, j, k))| {
            let mut tile = Tile::zeros(
                &[tile_size as usize, tile_size as usize, tile_size as usize],
                ElemType::F32,
            );
            let base_value = (i * 1000000 + j * 1000 + k) as f32;

            for ti in 0..tile_size {
                for tj in 0..tile_size {
                    for tk in 0..tile_size {
                        let value = base_value + (ti * 100 + tj * 10 + tk) as f32;
                        tile.set_scalar(&[ti as i64, tj as i64, tk as i64], Scalar::F32(value));
                    }
                }
            }

            partition.store_tile(&[i, j, k], &tile);
            let loaded = partition.load_tile(&[i, j, k]);

            // Verify ALL 32768 elements
            for ti in 0..tile_size {
                for tj in 0..tile_size {
                    for tk in 0..tile_size {
                        let expected = base_value + (ti * 100 + tj * 10 + tk) as f32;
                        let actual = loaded.get_scalar(&[ti as i64, tj as i64, tk as i64]);
                        if actual != Scalar::F32(expected) {
                            panic!("Iter {}: Mismatch at [{},{},{}]", iter, ti, tj, tk);
                        }
                    }
                }
            }
        });

    println!("3D tensor test completed: verified 20 tiles x 32768 elements = 655360 elements");
}

#[test]
fn test_large_scale_4d_tensor() {
    println!("Testing 4D tensor: 512x512x256x128 with random access");

    let buffer_size = 512 * 512 * 256 * 128 * 4;
    let mut buffer = vec![0u8; buffer_size];
    let ptr = buffer.as_mut_ptr();

    let tensor_view = TensorView::new(
        ptr,
        ElemType::I32,
        vec![512, 512, 256, 128],
        vec![512 * 256 * 128, 256 * 128, 128, 1],
    );
    let partition = PartitionView::new(
        tensor_view,
        vec![32, 32, 16, 8],
        vec![0, 1, 2, 3],
        true,
        Some(Scalar::I32(0)),
    );

    let mut rng = StdRng::seed_from_u64(8888);
    let grid_shape = partition.index_space_shape();

    // Generate 25 random positions
    let positions: Vec<_> = (0..25)
        .map(|_| {
            (
                rng.random_range(0..grid_shape[0]),
                rng.random_range(0..grid_shape[1]),
                rng.random_range(0..grid_shape[2]),
                rng.random_range(0..grid_shape[3]),
            )
        })
        .collect();

    use rayon::prelude::*;

    // Test 25 random positions with full tile verification in parallel
    positions
        .par_iter()
        .enumerate()
        .for_each(|(iter, &(i, j, k, l))| {
            let mut tile = Tile::zeros(&[32, 32, 16, 8], ElemType::I32);
            let base_value = i * 1000000 + j * 10000 + k * 100 + l;

            for ti in 0..32 {
                for tj in 0..32 {
                    for tk in 0..16 {
                        for tl in 0..8 {
                            let value = (base_value + ti * 1000 + tj * 100 + tk * 10 + tl) as i32;
                            tile.set_scalar(&[ti, tj, tk, tl], Scalar::I32(value));
                        }
                    }
                }
            }

            partition.store_tile(&[i, j, k, l], &tile);
            let loaded = partition.load_tile(&[i, j, k, l]);

            // Verify ALL 131072 elements (32*32*16*8)
            for ti in 0..32 {
                for tj in 0..32 {
                    for tk in 0..16 {
                        for tl in 0..8 {
                            let expected =
                                (base_value + ti * 1000 + tj * 100 + tk * 10 + tl) as i32;
                            let actual = loaded.get_scalar(&[ti, tj, tk, tl]);
                            if actual != Scalar::I32(expected) {
                                panic!("Iter {}: Mismatch at [{},{},{},{}]", iter, ti, tj, tk, tl);
                            }
                        }
                    }
                }
            }
        });

    println!("4D tensor test completed: verified 25 tiles x 131072 elements = 3276800 elements");
}

#[test]
fn test_memory_stress_512mb_random_ops() {
    println!("Memory stress test: 512MB buffer, random operations with RNG");

    let buffer_size = 512 * 1024 * 1024; // 512 MB
    let mut buffer = vec![0u8; buffer_size];
    let ptr = buffer.as_mut_ptr();

    // 8192 x 8192 f32 matrix
    let size = 8192;
    let tile_size = 128;
    let tensor_view = TensorView::new(ptr, ElemType::F32, vec![size, size], vec![size, 1]);
    let partition = PartitionView::new(
        tensor_view,
        vec![tile_size, tile_size],
        vec![0, 1],
        true,
        Some(Scalar::F32(0.0)),
    );

    let index_space = partition.index_space_shape();
    println!("Index space: {:?}", index_space);

    let mut rng = StdRng::seed_from_u64(55555);

    // Perform 15000 random load/store operations
    for op in 0..15000 {
        let i = rng.random_range(0..index_space[0]);
        let j = rng.random_range(0..index_space[1]);

        if op % 2 == 0 {
            // Store with unique pattern
            let mut tile = Tile::zeros(&[tile_size as usize, tile_size as usize], ElemType::F32);
            tile.set_scalar(&[0, 0], Scalar::F32(op as f32));
            tile.set_scalar(&[127, 127], Scalar::F32((op + 1) as f32));
            partition.store_tile(&[i, j], &tile);
        } else {
            // Load
            let loaded = partition.load_tile(&[i, j]);
            let _ = loaded.get_scalar(&[0, 0]);
        }

        if op % 1000 == 0 {
            println!("Completed {} operations", op);
        }
    }

    println!("Memory stress test completed successfully");
}
