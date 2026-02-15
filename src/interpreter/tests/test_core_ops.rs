use crate::interpreter::data_structures::{
    elem_type::{ElemType, Scalar},
    tile::Tile,
};
use ndrange::ndrange;
use rand::{RngExt, SeedableRng};

#[test]
fn test_core_1() {
    let tile = Tile::iota(128, ElemType::I16);
    dbg!(&tile);

    let tile2 = Tile::zeros(&[8, 4, 1, 32], ElemType::F16);
    dbg!(tile2.shape());

    let tile3 = tile2.broadcast(&[2, 8, 4, 16, 32]);
    dbg!(tile3.shape());

    let tile4_shape = [2, 16, 4];
    let tile4 = tile.clone().reshape(&tile4_shape);
    dbg!(&tile4);

    let tile5 = tile4.permute(&[2, 0, 1]);
    dbg!(&tile5);

    for [i, j, k] in ndrange(&tile4_shape) {
        assert_eq!(
            tile4.get_scalar(&[i, j, k]),
            // permuted
            tile5.get_scalar(&[k, i, j]),
            "Mismatch at tile4[({}, {}, {})] and tile5[({}, {}, {})]: {:?} != {:?}",
            i,
            j,
            k,
            k,
            i,
            j,
            tile4.get_scalar(&[i, j, k]),
            tile5.get_scalar(&[k, i, j]),
        );
    }
}

#[test]
fn test_select_dtypes() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Test I8 type with 2-D tiles [2, 4]
    {
        let shape = [2usize, 4];
        let size: usize = shape.iter().product();

        // Generate random boolean condition
        let mut cond_data = Vec::with_capacity(size);
        for _ in 0..size {
            cond_data.push(rng.random());
        }
        let cond = Tile::I1(
            ndarray::arr1(&cond_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        // Generate positive random values for true branch
        let mut true_data = Vec::with_capacity(size);
        for _ in 0..size {
            true_data.push(rng.random_range(1i8..127));
        }
        let true_vals = Tile::I8(
            ndarray::arr1(&true_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        // Generate negative random values for false branch
        let mut false_data = Vec::with_capacity(size);
        for _ in 0..size {
            false_data.push(rng.random_range(-128i8..0));
        }
        let false_vals = Tile::I8(
            ndarray::arr1(&false_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        let result = cond.select(&true_vals, &false_vals);

        // Verify against input tiles
        for [i, j] in ndrange(&shape) {
            let idx = (i * shape[1] as i64 + j) as usize;
            let expected = if cond_data[idx] {
                true_data[idx]
            } else {
                false_data[idx]
            };
            match result.get_scalar(&[i, j]) {
                Scalar::I8(v) => {
                    assert_eq!(v, expected);
                }
                _ => panic!("Expected I8 scalar"),
            }
        }
    }

    // Test I32 type with 3-D tiles [2, 4, 8]
    {
        let shape = [2usize, 4, 8];
        let size: usize = shape.iter().product();

        // Generate random boolean condition
        let mut cond_data = Vec::with_capacity(size);
        for _ in 0..size {
            cond_data.push(rng.random());
        }
        let cond = Tile::I1(
            ndarray::arr1(&cond_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        // Generate positive random values for true branch
        let mut true_data = Vec::with_capacity(size);
        for _ in 0..size {
            true_data.push(rng.random_range(1i32..2147483647));
        }
        let true_vals = Tile::I32(
            ndarray::arr1(&true_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        // Generate negative random values for false branch
        let mut false_data = Vec::with_capacity(size);
        for _ in 0..size {
            false_data.push(rng.random_range(-2147483648i32..0));
        }
        let false_vals = Tile::I32(
            ndarray::arr1(&false_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        let result = cond.select(&true_vals, &false_vals);

        // Verify against input tiles
        for [i, j, k] in ndrange(&shape) {
            let idx = (i * shape[1] as i64 * shape[2] as i64 + j * shape[2] as i64 + k) as usize;
            let expected = if cond_data[idx] {
                true_data[idx]
            } else {
                false_data[idx]
            };
            match result.get_scalar(&[i, j, k]) {
                Scalar::I32(v) => {
                    assert_eq!(v, expected);
                }
                _ => panic!("Expected I32 scalar"),
            }
        }
    }

    // Test F16 type with 4-D tiles [2, 4, 8, 16]
    {
        let shape = [2usize, 4, 8, 16];
        let size: usize = shape.iter().product();

        // Generate random boolean condition
        let mut cond_data = Vec::with_capacity(size);
        for _ in 0..size {
            cond_data.push(rng.random());
        }
        let cond = Tile::I1(
            ndarray::arr1(&cond_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        // Generate positive random values for true branch
        let mut true_data = Vec::with_capacity(size);
        for _ in 0..size {
            true_data.push(rng.random_range(1.0f32..1000.0) as f16);
        }
        let true_vals = Tile::F16(
            ndarray::arr1(&true_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        // Generate negative random values for false branch
        let mut false_data = Vec::with_capacity(size);
        for _ in 0..size {
            false_data.push(rng.random_range(-1000.0f32..0.0) as f16);
        }
        let false_vals = Tile::F16(
            ndarray::arr1(&false_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        let result = cond.select(&true_vals, &false_vals);

        // Verify against input tiles
        for [i, j, k, l] in ndrange(&shape) {
            let idx = (i * shape[1] as i64 * shape[2] as i64 * shape[3] as i64
                + j * shape[2] as i64 * shape[3] as i64
                + k * shape[3] as i64
                + l) as usize;
            let expected = if cond_data[idx] {
                true_data[idx]
            } else {
                false_data[idx]
            };
            match result.get_scalar(&[i, j, k, l]) {
                Scalar::F16(v) => {
                    let diff = ((v as f32) - (expected as f32)).abs();
                    assert!(
                        diff < 1e-2,
                        "Mismatch at [{}, {}, {}, {}]: {} != {}",
                        i,
                        j,
                        k,
                        l,
                        v,
                        expected
                    );
                }
                _ => panic!("Expected F16 scalar"),
            }
        }
    }

    // Test I64 type with 4-D tiles [4, 8, 16, 32]
    {
        let shape = [4usize, 8, 16, 32];
        let size: usize = shape.iter().product();

        // Generate random boolean condition
        let mut cond_data = Vec::with_capacity(size);
        for _ in 0..size {
            cond_data.push(rng.random());
        }
        let cond = Tile::I1(
            ndarray::arr1(&cond_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        // Generate positive random values for true branch
        let mut true_data = Vec::with_capacity(size);
        for _ in 0..size {
            true_data.push(rng.random_range(1i64..9223372036854775807));
        }
        let true_vals = Tile::I64(
            ndarray::arr1(&true_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        // Generate negative random values for false branch
        let mut false_data = Vec::with_capacity(size);
        for _ in 0..size {
            false_data.push(rng.random_range(-9223372036854775808i64..0));
        }
        let false_vals = Tile::I64(
            ndarray::arr1(&false_data)
                .into_shape_with_order(ndarray::IxDyn(&shape))
                .unwrap(),
        );

        let result = cond.select(&true_vals, &false_vals);

        // Verify against input tiles (sample a few points due to large size)
        for [i, j, k, l] in ndrange(&[4usize, 2, 4, 8]) {
            let idx = (i * shape[1] as i64 * shape[2] as i64 * shape[3] as i64
                + j * shape[2] as i64 * shape[3] as i64
                + k * shape[3] as i64
                + l) as usize;
            let expected = if cond_data[idx] {
                true_data[idx]
            } else {
                false_data[idx]
            };
            match result.get_scalar(&[i, j, k, l]) {
                Scalar::I64(v) => {
                    assert_eq!(v, expected);
                }
                _ => panic!("Expected I64 scalar"),
            }
        }
    }
}

#[test]
fn test_cat() {
    let tile1_orig = Tile::iota(128, ElemType::I32);
    let tile1_shape = [2, 1, 16, 4];
    let tile1 = tile1_orig.reshape(&tile1_shape);

    let tile2_orig = Tile::iota(128, ElemType::I32);
    let tile2_shape = [2, 1, 4, 16];
    let tile2_reshaped = tile2_orig.reshape(&tile2_shape);
    let tile2 = tile2_reshaped.permute(&[0, 1, 3, 2]);

    let tile3 = tile1.cat(&tile2, 2);
    dbg!(tile3.shape());
    assert_eq!(tile3.shape(), vec![2, 1, 32, 4]);

    for [ia, ib, ic, id] in ndrange(&tile1_shape) {
        assert_eq!(
            tile3.get_scalar(&[ia, ib, ic, id]),
            tile1.get_scalar(&[ia, ib, ic, id])
        );
        assert_eq!(
            tile3.get_scalar(&[ia, ib, ic + 16, id]),
            // NOTE: tile2_reshaped is Not permuted
            tile2_reshaped.get_scalar(&[ia, ib, id, ic])
        );
        println!("({ia}, {ib}, {ic}, {id}) test passed")
    }
}

#[should_panic(expected = "Broadcast failed")]
#[test]
fn test_core_invalid_broadcast() {
    let tile2 = Tile::zeros(&[8, 4, 1, 32], ElemType::F16);
    let tile4 = tile2.broadcast(&[2, 8, 4, 16, 4]);
    dbg!(tile4);
}

#[should_panic(expected = "Permutation is invalid")]
#[test]
fn test_core_invalid_permute() {
    let tile2 = Tile::zeros(&[8, 4, 1, 32], ElemType::F16);
    let tile4 = tile2.permute(&[0, 1, 3, 3]);
    dbg!(tile4);
}

#[test]
fn test_extract() {
    // Test 2-D tiles: shape [64, 32], extracting subtiles of shape [16, 8]
    {
        let src_shape = [64usize, 32];
        let subtile_shape = [16usize, 8];
        let tile = Tile::iota(src_shape.iter().product(), ElemType::I32).reshape(&src_shape);

        // Extract at [0, 0]
        let result = tile.extract(&[0, 0], &subtile_shape);
        assert_eq!(result.shape(), vec![16, 8]);
        for [i, j] in ndrange(&subtile_shape) {
            let expected = (i as i32) * 32 + (j as i32);
            match result.get_scalar(&[i, j]) {
                Scalar::I32(v) => assert_eq!(v, expected),
                _ => panic!("Expected I32 scalar"),
            }
        }

        // Extract at [1, 2]
        let result = tile.extract(&[1, 2], &subtile_shape);
        assert_eq!(result.shape(), vec![16, 8]);
        for [i, j] in ndrange(&subtile_shape) {
            let src_i = 1 * 16 + i;
            let src_j = 2 * 8 + j;
            let expected = (src_i as i32) * 32 + (src_j as i32);
            match result.get_scalar(&[i, j]) {
                Scalar::I32(v) => assert_eq!(v, expected),
                _ => panic!("Expected I32 scalar"),
            }
        }

        // Extract at [3, 1]
        let result = tile.extract(&[3, 1], &subtile_shape);
        assert_eq!(result.shape(), vec![16, 8]);
        for [i, j] in ndrange(&subtile_shape) {
            let src_i = 3 * 16 + i;
            let src_j = 1 * 8 + j;
            let expected = (src_i as i32) * 32 + (src_j as i32);
            match result.get_scalar(&[i, j]) {
                Scalar::I32(v) => assert_eq!(v, expected),
                _ => panic!("Expected I32 scalar"),
            }
        }
    }

    // Test 3-D tiles: shape [32, 16, 8], extracting subtiles of shape [8, 4, 2]
    {
        let src_shape = [32usize, 16, 8];
        let subtile_shape = [8usize, 4, 2];
        let tile = Tile::iota(src_shape.iter().product(), ElemType::I16).reshape(&src_shape);

        // Extract at [0, 0, 0]
        let result = tile.extract(&[0, 0, 0], &subtile_shape);
        assert_eq!(result.shape(), vec![8, 4, 2]);
        for [i, j, k] in ndrange(&subtile_shape) {
            let expected = ((i as i16) * 16 * 8 + (j as i16) * 8 + (k as i16)) as i16;
            match result.get_scalar(&[i, j, k]) {
                Scalar::I16(v) => assert_eq!(v, expected),
                _ => panic!("Expected I16 scalar"),
            }
        }

        // Extract at [2, 1, 3]
        let result = tile.extract(&[2, 1, 3], &subtile_shape);
        assert_eq!(result.shape(), vec![8, 4, 2]);
        for [i, j, k] in ndrange(&subtile_shape) {
            let src_i = 2 * 8 + i;
            let src_j = 1 * 4 + j;
            let src_k = 3 * 2 + k;
            let expected = ((src_i as i16) * 16 * 8 + (src_j as i16) * 8 + (src_k as i16)) as i16;
            match result.get_scalar(&[i, j, k]) {
                Scalar::I16(v) => assert_eq!(v, expected),
                _ => panic!("Expected I16 scalar"),
            }
        }

        // Extract at [1, 3, 1]
        let result = tile.extract(&[1, 3, 1], &subtile_shape);
        assert_eq!(result.shape(), vec![8, 4, 2]);
        for [i, j, k] in ndrange(&subtile_shape) {
            let src_i = 1 * 8 + i;
            let src_j = 3 * 4 + j;
            let src_k = 1 * 2 + k;
            let expected = ((src_i as i16) * 16 * 8 + (src_j as i16) * 8 + (src_k as i16)) as i16;
            match result.get_scalar(&[i, j, k]) {
                Scalar::I16(v) => assert_eq!(v, expected),
                _ => panic!("Expected I16 scalar"),
            }
        }
    }

    // Test 4-D tiles: shape [16, 8, 4, 32], extracting subtiles of shape [4, 2, 2, 8]
    {
        let src_shape = [16usize, 8, 4, 32];
        let subtile_shape = [4usize, 2, 2, 8];
        let tile = Tile::iota(src_shape.iter().product(), ElemType::I64).reshape(&src_shape);

        // Extract at [0, 0, 0, 0]
        let result = tile.extract(&[0, 0, 0, 0], &subtile_shape);
        assert_eq!(result.shape(), vec![4, 2, 2, 8]);
        for [i, j, k, l] in ndrange(&subtile_shape) {
            let expected =
                ((i as i64) * 8 * 4 * 32 + (j as i64) * 4 * 32 + (k as i64) * 32 + (l as i64))
                    as i64;
            match result.get_scalar(&[i, j, k, l]) {
                Scalar::I64(v) => assert_eq!(v, expected),
                _ => panic!("Expected I64 scalar"),
            }
        }

        // Extract at [2, 3, 1, 1]
        let result = tile.extract(&[2, 3, 1, 1], &subtile_shape);
        assert_eq!(result.shape(), vec![4, 2, 2, 8]);
        for [i, j, k, l] in ndrange(&subtile_shape) {
            let src_i = 2 * 4 + i;
            let src_j = 3 * 2 + j;
            let src_k = 1 * 2 + k;
            let src_l = 1 * 8 + l;
            let expected = ((src_i as i64) * 8 * 4 * 32
                + (src_j as i64) * 4 * 32
                + (src_k as i64) * 32
                + (src_l as i64)) as i64;
            match result.get_scalar(&[i, j, k, l]) {
                Scalar::I64(v) => assert_eq!(v, expected),
                _ => panic!("Expected I64 scalar"),
            }
        }

        // Extract at [1, 1, 0, 3]
        let result = tile.extract(&[1, 1, 0, 3], &subtile_shape);
        assert_eq!(result.shape(), vec![4, 2, 2, 8]);
        for [i, j, k, l] in ndrange(&subtile_shape) {
            let src_i = 1 * 4 + i;
            let src_j = 1 * 2 + j;
            let src_k = 0 * 2 + k;
            let src_l = 3 * 8 + l;
            let expected = ((src_i as i64) * 8 * 4 * 32
                + (src_j as i64) * 4 * 32
                + (src_k as i64) * 32
                + (src_l as i64)) as i64;
            match result.get_scalar(&[i, j, k, l]) {
                Scalar::I64(v) => assert_eq!(v, expected),
                _ => panic!("Expected I64 scalar"),
            }
        }
    }

    // Test F32 type with 4-D tiles
    {
        let src_shape = [8usize, 4, 2, 16];
        let subtile_shape = [2usize, 2, 1, 4];
        let tile = Tile::iota(src_shape.iter().product(), ElemType::I32).reshape(&src_shape);

        // Convert I32 tile to F32 for testing
        let Tile::I32(arr) = tile else {
            panic!("Expected I32 tile")
        };
        let f32_tile = Tile::F32(arr.mapv(|v| v as f32));

        // Extract at [1, 1, 1, 2]
        let result = f32_tile.extract(&[1, 1, 1, 2], &subtile_shape);
        assert_eq!(result.shape(), vec![2, 2, 1, 4]);
        for [i, j, k, l] in ndrange(&subtile_shape) {
            let src_i = 1 * 2 + i;
            let src_j = 1 * 2 + j;
            let src_k = 1 * 1 + k;
            let src_l = 2 * 4 + l;
            let expected = ((src_i as i32) * 4 * 2 * 16
                + (src_j as i32) * 2 * 16
                + (src_k as i32) * 16
                + (src_l as i32)) as f32;
            match result.get_scalar(&[i, j, k, l]) {
                Scalar::F32(v) => {
                    let diff = (v - expected).abs();
                    assert!(
                        diff < 1e-6,
                        "Mismatch at [{}, {}, {}, {}]: {} != {}",
                        i,
                        j,
                        k,
                        l,
                        v,
                        expected
                    );
                }
                _ => panic!("Expected F32 scalar"),
            }
        }
    }
}
