// Tests for bitwise operations (Section 8.9)
//
// Tests covering all 3 bitwise operations:
// - Binary (3): andi, ori, xori
//
// Supported types: i1 (boolean), i8, i16, i32, i64

use crate::interpreter::data_structures::tile::Tile;
use ndrange::ndrange;
use rand::{RngExt, SeedableRng};

// ============================================================================
// ANDI Tests
// ============================================================================

#[test]
fn test_andi_i32_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();
    let rhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.andi(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = i * 8 + j;
        let expected = lhs_data[idx] & rhs_data[idx];
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "andi i32 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_andi_i16_3d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(43);
    let shape = ndarray::IxDyn(&[2, 4, 8]);
    let size = 64;

    let lhs_data: Vec<i16> = (0..size).map(|_| rng.random::<i16>()).collect();
    let rhs_data: Vec<i16> = (0..size).map(|_| rng.random::<i16>()).collect();

    let lhs = Tile::I16(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I16(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.andi(&rhs);

    for [i, j, k] in ndrange(&[2, 4, 8]) {
        let idx = (i * 4 * 8) + (j * 8) + k;
        let expected = lhs_data[idx] & rhs_data[idx];
        if let Tile::I16(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j, k]],
                expected,
                "andi i16 mismatch at [{},{},{}]: expected {}, got {}",
                i,
                j,
                k,
                expected,
                result_arr[[i, j, k]]
            );
        }
    }
}

#[test]
fn test_andi_i8_4d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(44);
    let shape = ndarray::IxDyn(&[2, 4, 8, 16]);
    let size = 1024;

    let lhs_data: Vec<i8> = (0..size).map(|_| rng.random::<i8>()).collect();
    let rhs_data: Vec<i8> = (0..size).map(|_| rng.random::<i8>()).collect();

    let lhs = Tile::I8(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I8(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.andi(&rhs);

    for [i, j, k, l] in ndrange(&[2, 4, 8, 16]) {
        let idx = (i * 4 * 8 * 16) + (j * 8 * 16) + (k * 16) + l;
        let expected = lhs_data[idx] & rhs_data[idx];
        if let Tile::I8(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j, k, l]],
                expected,
                "andi i8 mismatch at [{},{},{},{}]: expected {}, got {}",
                i,
                j,
                k,
                l,
                expected,
                result_arr[[i, j, k, l]]
            );
        }
    }
}

#[test]
fn test_andi_i64_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(45);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i64> = (0..size).map(|_| rng.random::<i64>()).collect();
    let rhs_data: Vec<i64> = (0..size).map(|_| rng.random::<i64>()).collect();

    let lhs = Tile::I64(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I64(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.andi(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = i * 8 + j;
        let expected = lhs_data[idx] & rhs_data[idx];
        if let Tile::I64(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "andi i64 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_andi_i1_boolean() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(46);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<bool> = (0..size).map(|_| rng.random::<bool>()).collect();
    let rhs_data: Vec<bool> = (0..size).map(|_| rng.random::<bool>()).collect();

    let lhs = Tile::I1(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I1(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.andi(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = i * 8 + j;
        let expected = lhs_data[idx] & rhs_data[idx];
        if let Tile::I1(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "andi i1 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

// ============================================================================
// ORI Tests
// ============================================================================

#[test]
fn test_ori_i32_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(47);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();
    let rhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.ori(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = i * 8 + j;
        let expected = lhs_data[idx] | rhs_data[idx];
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "ori i32 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_ori_i16_3d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(48);
    let shape = ndarray::IxDyn(&[2, 4, 8]);
    let size = 64;

    let lhs_data: Vec<i16> = (0..size).map(|_| rng.random::<i16>()).collect();
    let rhs_data: Vec<i16> = (0..size).map(|_| rng.random::<i16>()).collect();

    let lhs = Tile::I16(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I16(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.ori(&rhs);

    for [i, j, k] in ndrange(&[2, 4, 8]) {
        let idx = (i * 4 * 8) + (j * 8) + k;
        let expected = lhs_data[idx] | rhs_data[idx];
        if let Tile::I16(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j, k]],
                expected,
                "ori i16 mismatch at [{},{},{}]: expected {}, got {}",
                i,
                j,
                k,
                expected,
                result_arr[[i, j, k]]
            );
        }
    }
}

#[test]
fn test_ori_i8_4d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(49);
    let shape = ndarray::IxDyn(&[2, 4, 8, 16]);
    let size = 1024;

    let lhs_data: Vec<i8> = (0..size).map(|_| rng.random::<i8>()).collect();
    let rhs_data: Vec<i8> = (0..size).map(|_| rng.random::<i8>()).collect();

    let lhs = Tile::I8(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I8(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.ori(&rhs);

    for [i, j, k, l] in ndrange(&[2, 4, 8, 16]) {
        let idx = (i * 4 * 8 * 16) + (j * 8 * 16) + (k * 16) + l;
        let expected = lhs_data[idx] | rhs_data[idx];
        if let Tile::I8(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j, k, l]],
                expected,
                "ori i8 mismatch at [{},{},{},{}]: expected {}, got {}",
                i,
                j,
                k,
                l,
                expected,
                result_arr[[i, j, k, l]]
            );
        }
    }
}

#[test]
fn test_ori_i64_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(50);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i64> = (0..size).map(|_| rng.random::<i64>()).collect();
    let rhs_data: Vec<i64> = (0..size).map(|_| rng.random::<i64>()).collect();

    let lhs = Tile::I64(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I64(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.ori(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = i * 8 + j;
        let expected = lhs_data[idx] | rhs_data[idx];
        if let Tile::I64(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "ori i64 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_ori_i1_boolean() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(51);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<bool> = (0..size).map(|_| rng.random::<bool>()).collect();
    let rhs_data: Vec<bool> = (0..size).map(|_| rng.random::<bool>()).collect();

    let lhs = Tile::I1(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I1(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.ori(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = i * 8 + j;
        let expected = lhs_data[idx] | rhs_data[idx];
        if let Tile::I1(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "ori i1 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

// ============================================================================
// XORI Tests
// ============================================================================

#[test]
fn test_xori_i32_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(52);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();
    let rhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.xori(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = i * 8 + j;
        let expected = lhs_data[idx] ^ rhs_data[idx];
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "xori i32 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_xori_i16_3d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(53);
    let shape = ndarray::IxDyn(&[2, 4, 8]);
    let size = 64;

    let lhs_data: Vec<i16> = (0..size).map(|_| rng.random::<i16>()).collect();
    let rhs_data: Vec<i16> = (0..size).map(|_| rng.random::<i16>()).collect();

    let lhs = Tile::I16(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I16(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.xori(&rhs);

    for [i, j, k] in ndrange(&[2, 4, 8]) {
        let idx = (i * 4 * 8) + (j * 8) + k;
        let expected = lhs_data[idx] ^ rhs_data[idx];
        if let Tile::I16(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j, k]],
                expected,
                "xori i16 mismatch at [{},{},{}]: expected {}, got {}",
                i,
                j,
                k,
                expected,
                result_arr[[i, j, k]]
            );
        }
    }
}

#[test]
fn test_xori_i8_4d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(54);
    let shape = ndarray::IxDyn(&[2, 4, 8, 16]);
    let size = 1024;

    let lhs_data: Vec<i8> = (0..size).map(|_| rng.random::<i8>()).collect();
    let rhs_data: Vec<i8> = (0..size).map(|_| rng.random::<i8>()).collect();

    let lhs = Tile::I8(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I8(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.xori(&rhs);

    for [i, j, k, l] in ndrange(&[2, 4, 8, 16]) {
        let idx = (i * 4 * 8 * 16) + (j * 8 * 16) + (k * 16) + l;
        let expected = lhs_data[idx] ^ rhs_data[idx];
        if let Tile::I8(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j, k, l]],
                expected,
                "xori i8 mismatch at [{},{},{},{}]: expected {}, got {}",
                i,
                j,
                k,
                l,
                expected,
                result_arr[[i, j, k, l]]
            );
        }
    }
}

#[test]
fn test_xori_i64_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(55);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i64> = (0..size).map(|_| rng.random::<i64>()).collect();
    let rhs_data: Vec<i64> = (0..size).map(|_| rng.random::<i64>()).collect();

    let lhs = Tile::I64(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I64(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.xori(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = i * 8 + j;
        let expected = lhs_data[idx] ^ rhs_data[idx];
        if let Tile::I64(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "xori i64 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_xori_i1_boolean() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(56);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<bool> = (0..size).map(|_| rng.random::<bool>()).collect();
    let rhs_data: Vec<bool> = (0..size).map(|_| rng.random::<bool>()).collect();

    let lhs = Tile::I1(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I1(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.xori(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = i * 8 + j;
        let expected = lhs_data[idx] ^ rhs_data[idx];
        if let Tile::I1(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "xori i1 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_andi_all_ones() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|i| i as i32 + 1).collect();
    let rhs_data: Vec<i32> = vec![-1; size];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data).unwrap());

    let result = lhs.andi(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = i * 8 + j;
        let expected = lhs_data[idx];
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "andi all ones mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_andi_all_zeros() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|i| i as i32 + 1).collect();
    let rhs_data: Vec<i32> = vec![0; size];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data).unwrap());

    let result = lhs.andi(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]], 0,
                "andi all zeros mismatch at [{},{}]: expected 0, got {}",
                i, j, result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_ori_all_zeros() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|i| i as i32 + 1).collect();
    let rhs_data: Vec<i32> = vec![0; size];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data).unwrap());

    let result = lhs.ori(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = i * 8 + j;
        let expected = lhs_data[idx];
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "ori all zeros mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_xori_same_input() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(57);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), data).unwrap());

    let result = lhs.xori(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]], 0,
                "xori same input mismatch at [{},{}]: expected 0, got {}",
                i, j, result_arr[[i, j]]
            );
        }
    }
}

// ============================================================================
// Type Mismatch Panic Tests
// ============================================================================

#[test]
#[should_panic(expected = "AndI requires matching integer types")]
fn test_andi_type_mismatch() {
    let lhs = Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i32; 32]).unwrap());
    let rhs = Tile::I64(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i64; 32]).unwrap());

    let _ = lhs.andi(&rhs);
}

#[test]
#[should_panic(expected = "OrI requires matching integer types")]
fn test_ori_type_mismatch() {
    let lhs = Tile::I16(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i16; 32]).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i32; 32]).unwrap());

    let _ = lhs.ori(&rhs);
}

#[test]
#[should_panic(expected = "XOrI requires matching integer types")]
fn test_xori_type_mismatch() {
    let lhs = Tile::I8(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i8; 32]).unwrap());
    let rhs = Tile::I64(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i64; 32]).unwrap());

    let _ = lhs.xori(&rhs);
}

#[test]
#[should_panic(expected = "Shape mismatch")]
fn test_andi_shape_mismatch() {
    let lhs = Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i32; 32]).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[8, 4]), vec![0i32; 32]).unwrap());

    let _ = lhs.andi(&rhs);
}

#[test]
#[should_panic(expected = "Shape mismatch")]
fn test_ori_shape_mismatch() {
    let lhs = Tile::I16(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[2, 4, 8]), vec![0i16; 64]).unwrap());
    let rhs = Tile::I16(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8, 2]), vec![0i16; 64]).unwrap());

    let _ = lhs.ori(&rhs);
}

#[test]
#[should_panic(expected = "Shape mismatch")]
fn test_xori_shape_mismatch() {
    let lhs = Tile::I64(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i64; 32]).unwrap());
    let rhs = Tile::I64(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8, 2]), vec![0i64; 64]).unwrap());

    let _ = lhs.xori(&rhs);
}

// ============================================================================
// All Bitwise Operations on All Types
// ============================================================================

#[test]
fn test_all_bitwise_ops_on_all_integer_types() {
    let shape_2d = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    // Test i8
    {
        let lhs_data: Vec<i8> = vec![0b0011_0011, 0b0101_0101, 0b0000_1111, 0b0000_0000,
                                      0b0001_0010, 0b0011_0100, -1, 0];
        let rhs_data: Vec<i8> = vec![0b0101_0101, 0b0011_0011, 0b0000_0000, 0b0000_1111,
                                      0b0011_0100, 0b0001_0010, 0, -1];

        let lhs = Tile::I8(ndarray::Array::from_shape_vec(shape_2d.clone(), lhs_data.clone()).unwrap());
        let rhs = Tile::I8(ndarray::Array::from_shape_vec(shape_2d.clone(), rhs_data.clone()).unwrap());

        let and_result = lhs.andi(&rhs);
        let or_result = lhs.ori(&rhs);
        let xor_result = lhs.xori(&rhs);

        for i in 0..8 {
            let expected_and = lhs_data[i] & rhs_data[i];
            let expected_or = lhs_data[i] | rhs_data[i];
            let expected_xor = lhs_data[i] ^ rhs_data[i];

            if let Tile::I8(arr) = &and_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_and, "i8 andi mismatch at index {}", i);
            }
            if let Tile::I8(arr) = &or_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_or, "i8 ori mismatch at index {}", i);
            }
            if let Tile::I8(arr) = &xor_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_xor, "i8 xori mismatch at index {}", i);
            }
        }
    }

    // Test i16
    {
        let lhs_data: Vec<i16> = vec![0b0011_1100_0011_1100, 0b0101_0101_0101_0101, 0b0000_1111_0000_1111, 0b0000_0000_0000_0000,
                                      0b0001_0010_0011_0100, 0b0101_0110_0111_1000, -1, 0];
        let rhs_data: Vec<i16> = vec![0b0101_0101_0101_0101, 0b0011_1100_0011_1100, 0b0000_0000_0000_0000, 0b0000_1111_0000_1111,
                                      0b0101_0110_0111_1000, 0b0001_0010_0011_0100, 0, -1];

        let lhs = Tile::I16(ndarray::Array::from_shape_vec(shape_2d.clone(), lhs_data.clone()).unwrap());
        let rhs = Tile::I16(ndarray::Array::from_shape_vec(shape_2d.clone(), rhs_data.clone()).unwrap());

        let and_result = lhs.andi(&rhs);
        let or_result = lhs.ori(&rhs);
        let xor_result = lhs.xori(&rhs);

        for i in 0..8 {
            let expected_and = lhs_data[i] & rhs_data[i];
            let expected_or = lhs_data[i] | rhs_data[i];
            let expected_xor = lhs_data[i] ^ rhs_data[i];

            if let Tile::I16(arr) = &and_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_and, "i16 andi mismatch at index {}", i);
            }
            if let Tile::I16(arr) = &or_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_or, "i16 ori mismatch at index {}", i);
            }
            if let Tile::I16(arr) = &xor_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_xor, "i16 xori mismatch at index {}", i);
            }
        }
    }

    // Test i32
    {
        let lhs_data: Vec<i32> = vec![-0x66666666, 0x55555555, -1, 0,
                                      0x12345678, -0x789ABCDF, -0x21524111, -0x35145442];
        let rhs_data: Vec<i32> = vec![0x55555555, -0x66666666, 0, -1,
                                      -0x789ABCDF, 0x12345678, -0x35145442, -0x21524111];

        let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape_2d.clone(), lhs_data.clone()).unwrap());
        let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape_2d.clone(), rhs_data.clone()).unwrap());

        let and_result = lhs.andi(&rhs);
        let or_result = lhs.ori(&rhs);
        let xor_result = lhs.xori(&rhs);

        for i in 0..8 {
            let expected_and = lhs_data[i] & rhs_data[i];
            let expected_or = lhs_data[i] | rhs_data[i];
            let expected_xor = lhs_data[i] ^ rhs_data[i];

            if let Tile::I32(arr) = &and_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_and, "i32 andi mismatch at index {}", i);
            }
            if let Tile::I32(arr) = &or_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_or, "i32 ori mismatch at index {}", i);
            }
            if let Tile::I32(arr) = &xor_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_xor, "i32 xori mismatch at index {}", i);
            }
        }
    }

    // Test i64
    {
        let lhs_data: Vec<i64> = vec![-0x6666666666666666, 0x5555555555555555, -1, 0,
                                      0x123456789ABCDEF0, 0x0FEDCBA987654321, -0x2152411135014542, 0x3456789ABCDEF012];
        let rhs_data: Vec<i64> = vec![0x5555555555555555, -0x6666666666666666, 0, -1,
                                      0x0FEDCBA987654321, 0x123456789ABCDEF0, 0x3456789ABCDEF012, -0x2152411135014542];

        let lhs = Tile::I64(ndarray::Array::from_shape_vec(shape_2d.clone(), lhs_data.clone()).unwrap());
        let rhs = Tile::I64(ndarray::Array::from_shape_vec(shape_2d.clone(), rhs_data.clone()).unwrap());

        let and_result = lhs.andi(&rhs);
        let or_result = lhs.ori(&rhs);
        let xor_result = lhs.xori(&rhs);

        for i in 0..8 {
            let expected_and = lhs_data[i] & rhs_data[i];
            let expected_or = lhs_data[i] | rhs_data[i];
            let expected_xor = lhs_data[i] ^ rhs_data[i];

            if let Tile::I64(arr) = &and_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_and, "i64 andi mismatch at index {}", i);
            }
            if let Tile::I64(arr) = &or_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_or, "i64 ori mismatch at index {}", i);
            }
            if let Tile::I64(arr) = &xor_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_xor, "i64 xori mismatch at index {}", i);
            }
        }
    }

    // Test i1 (boolean)
    {
        let lhs_data: Vec<bool> = vec![true, false, true, false, true, false, true, false];
        let rhs_data: Vec<bool> = vec![true, true, false, false, true, true, false, false];

        let lhs = Tile::I1(ndarray::Array::from_shape_vec(shape_2d.clone(), lhs_data.clone()).unwrap());
        let rhs = Tile::I1(ndarray::Array::from_shape_vec(shape_2d.clone(), rhs_data.clone()).unwrap());

        let and_result = lhs.andi(&rhs);
        let or_result = lhs.ori(&rhs);
        let xor_result = lhs.xori(&rhs);

        for i in 0..8 {
            let expected_and = lhs_data[i] & rhs_data[i];
            let expected_or = lhs_data[i] | rhs_data[i];
            let expected_xor = lhs_data[i] ^ rhs_data[i];

            if let Tile::I1(arr) = &and_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_and, "i1 andi mismatch at index {}", i);
            }
            if let Tile::I1(arr) = &or_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_or, "i1 ori mismatch at index {}", i);
            }
            if let Tile::I1(arr) = &xor_result {
                assert_eq!(arr[[i / 4, i % 4]], expected_xor, "i1 xori mismatch at index {}", i);
            }
        }
    }
}
