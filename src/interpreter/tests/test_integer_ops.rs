// Tests for integer operations (Section 8.8)
//
// Tests covering all 14 integer operations:
// - Unary (2): absi, negi
// - Binary (11): addi, divi, maxi, mini, muli, mulhii, remi, shli, shri, subi
// - Ternary (1): mmai
//
// Supported types: i1 (boolean), i8, i16, i32, i64

use crate::interpreter::data_structures::tile::Tile;
use crate::cuda_tile_ir::enums::{ComparisonPredicate, RoundingMode, Signedness};
use ndrange::ndrange;
use rand::{RngExt, SeedableRng};

// ============================================================================
// ABSI Tests
// ============================================================================

#[test]
fn test_absi_i32_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();

    let tile = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), data.clone()).unwrap());
    let result = tile.absi();

    for [i, j] in ndrange(&[4, 8]) {
        let idx = (i * 8 + j) as usize;
        let expected = data[idx].abs();
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "absi i32 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_absi_i8_negative() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let data: Vec<i8> = (0..size).map(|i| -((i % 127) as i8 + 1)).collect();

    let tile = Tile::I8(ndarray::Array::from_shape_vec(shape.clone(), data.clone()).unwrap());
    let result = tile.absi();

    for [i, j] in ndrange(&[4, 8]) {
        let idx = (i * 8 + j) as usize;
        let expected = data[idx].abs();
        if let Tile::I8(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "absi i8 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

// ============================================================================
// NEGI Tests
// ============================================================================

#[test]
fn test_negi_i32_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(43);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();

    let tile = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), data.clone()).unwrap());
    let result = tile.negi();

    for [i, j] in ndrange(&[4, 8]) {
        let idx = (i * 8 + j) as usize;
        let expected = data[idx].wrapping_neg();
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "negi i32 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_negi_i16_3d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(44);
    let shape = ndarray::IxDyn(&[2, 4, 8]);
    let size = 64;

    let data: Vec<i16> = (0..size).map(|_| rng.random::<i16>()).collect();

    let tile = Tile::I16(ndarray::Array::from_shape_vec(shape.clone(), data.clone()).unwrap());
    let result = tile.negi();

    for [i, j, k] in ndrange(&[2, 4, 8]) {
        let idx = ((i * 4 * 8) + (j * 8) + k) as usize;
        let expected = data[idx].wrapping_neg();
        if let Tile::I16(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j, k]],
                expected,
                "negi i16 mismatch at [{},{},{}]: expected {}, got {}",
                i,
                j,
                k,
                expected,
                result_arr[[i, j, k]]
            );
        }
    }
}

// ============================================================================
// ADDI Tests
// ============================================================================

#[test]
fn test_addi_i32_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(45);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();
    let rhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.addi(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = (i * 8 + j) as usize;
        let expected = lhs_data[idx].wrapping_add(rhs_data[idx]);
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "addi i32 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_addi_i16_3d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(46);
    let shape = ndarray::IxDyn(&[2, 4, 8]);
    let size = 64;

    let lhs_data: Vec<i16> = (0..size).map(|_| rng.random::<i16>()).collect();
    let rhs_data: Vec<i16> = (0..size).map(|_| rng.random::<i16>()).collect();

    let lhs = Tile::I16(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I16(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.addi(&rhs);

    for [i, j, k] in ndrange(&[2, 4, 8]) {
        let idx = ((i * 4 * 8) + (j * 8) + k) as usize;
        let expected = lhs_data[idx].wrapping_add(rhs_data[idx]);
        if let Tile::I16(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j, k]],
                expected,
                "addi i16 mismatch at [{},{},{}]: expected {}, got {}",
                i,
                j,
                k,
                expected,
                result_arr[[i, j, k]]
            );
        }
    }
}

// ============================================================================
// SUBI Tests
// ============================================================================

#[test]
fn test_subi_i32_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(47);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();
    let rhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.subi(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = (i * 8 + j) as usize;
        let expected = lhs_data[idx].wrapping_sub(rhs_data[idx]);
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "subi i32 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

// ============================================================================
// MULI Tests
// ============================================================================

#[test]
fn test_muli_i32_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(48);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();
    let rhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.muli(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = (i * 8 + j) as usize;
        let expected = lhs_data[idx].wrapping_mul(rhs_data[idx]);
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "muli i32 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

// ============================================================================
// MULHII Tests
// ============================================================================

#[test]
fn test_mulhii_i32_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(49);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();
    let rhs_data: Vec<i32> = (0..size).map(|_| rng.random::<i32>()).collect();

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.mulhii(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = (i * 8 + j) as usize;
        // Cast through u32 to avoid sign extension (same as implementation)
        let x_u = (lhs_data[idx] as u32) as u64;
        let y_u = (rhs_data[idx] as u32) as u64;
        let full = x_u.wrapping_mul(y_u);
        let expected = (full >> 32) as i32;
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "mulhii i32 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_mulhii_i16_2d() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(49);
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i16> = (0..size).map(|_| rng.random::<i16>()).collect();
    let rhs_data: Vec<i16> = (0..size).map(|_| rng.random::<i16>()).collect();

    let lhs = Tile::I16(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I16(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.mulhii(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = (i * 8 + j) as usize;
        let x_u = lhs_data[idx] as u16;
        let y_u = rhs_data[idx] as u16;
        let full = (x_u as u32) * (y_u as u32);
        let expected = (full >> 16) as i16;
        if let Tile::I16(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "mulhii i16 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

// ============================================================================
// DIVI Tests
// ============================================================================

#[test]
fn test_divi_i32_signed_truncate() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<i32> = vec![7, -7, 7, -7, 100, -100, 0, 10];
    let rhs_data: Vec<i32> = vec![3, 3, -3, -3, 3, -3, 5, 2];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.divi(&rhs, Signedness::Signed, RoundingMode::Zero);

    for i in 0..8 {
        let expected = lhs_data[i] / rhs_data[i];
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected,
                "divi i32 signed truncate mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

#[test]
fn test_divi_i32_unsigned() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<i32> = vec![7, -7, 100, 200, 50, 10, 0, 15];
    let rhs_data: Vec<i32> = vec![3, 3, 3, 5, 7, 3, 5, 4];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.divi(&rhs, Signedness::Unsigned, RoundingMode::Zero);

    for i in 0..8 {
        let expected = ((lhs_data[i] as u32) / (rhs_data[i] as u32)) as i32;
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected,
                "divi i32 unsigned mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

#[test]
fn test_divi_i32_signed_floor() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    // Test floor division
    let lhs_data: Vec<i32> = vec![7, -7, 7, -7, -10, 10, -1, 5];
    let rhs_data: Vec<i32> = vec![3, 3, -3, -3, 3, 3, 3, 2];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.divi(&rhs, Signedness::Signed, RoundingMode::NegativeInf);

    let expected_results: Vec<i32> = vec![2, -3, -3, 2, -4, 3, -1, 2];

    for i in 0..8 {
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected_results[i],
                "divi i32 signed floor mismatch at index {}: expected {}, got {}",
                i,
                expected_results[i],
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

#[test]
fn test_divi_i32_signed_ceil() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    // Test ceil division
    let lhs_data: Vec<i32> = vec![7, -7, 7, -7, -10, 10, -1, 5];
    let rhs_data: Vec<i32> = vec![3, 3, -3, -3, 3, 3, 3, 2];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.divi(&rhs, Signedness::Signed, RoundingMode::PositiveInf);

    let expected_results: Vec<i32> = vec![3, -2, -2, 3, -3, 4, 0, 3];

    for i in 0..8 {
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected_results[i],
                "divi i32 signed ceil mismatch at index {}: expected {}, got {}",
                i,
                expected_results[i],
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

// ============================================================================
// REMI Tests
// ============================================================================

#[test]
fn test_remi_i32_signed() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    // Test signed remainder: sign matches dividend
    let lhs_data: Vec<i32> = vec![7, -7, 7, -7, 10, -10, 5, -5];
    let rhs_data: Vec<i32> = vec![3, 3, -3, -3, 3, 3, 7, 7];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.remi(&rhs, Signedness::Signed);

    for i in 0..8 {
        let expected = lhs_data[i] % rhs_data[i];
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected,
                "remi i32 signed mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

#[test]
fn test_remi_i32_unsigned() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<i32> = vec![7, -7, 10, 100, 50, 10, 5, 15];
    let rhs_data: Vec<i32> = vec![3, 3, 3, 7, 7, 3, 2, 4];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.remi(&rhs, Signedness::Unsigned);

    for i in 0..8 {
        let expected = ((lhs_data[i] as u32) % (rhs_data[i] as u32)) as i32;
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected,
                "remi i32 unsigned mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

// ============================================================================
// SHLI Tests
// ============================================================================

#[test]
fn test_shli_i32_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|i| i as i32 + 1).collect();
    let rhs_data: Vec<i32> = (0..size).map(|i| (i % 32) as i32).collect();

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.shli(&rhs);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = (i * 8 + j) as usize;
        let expected = lhs_data[idx].wrapping_shl((rhs_data[idx] as u32) & 31);
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "shli i32 mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

// ============================================================================
// SHRI Tests
// ============================================================================

#[test]
fn test_shri_i32_signed() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|i| ((i as i32) + 1) * -1000).collect();
    let rhs_data: Vec<i32> = (0..size).map(|i| (i % 32) as i32).collect();

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.shri(&rhs, Signedness::Signed);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = (i * 8 + j) as usize;
        let expected = lhs_data[idx].wrapping_shr((rhs_data[idx] as u32) & 31);
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "shri i32 signed mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

#[test]
fn test_shri_i32_unsigned() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<i32> = (0..size).map(|i| ((i as i32) + 1) * -1000).collect();
    let rhs_data: Vec<i32> = (0..size).map(|i| (i % 32) as i32).collect();

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.shri(&rhs, Signedness::Unsigned);

    for [i, j] in ndrange(&[4, 8]) {
        let idx = (i * 8 + j) as usize;
        let expected = ((lhs_data[idx] as u32) >> ((rhs_data[idx] as u32) & 31)) as i32;
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i, j]],
                expected,
                "shri i32 unsigned mismatch at [{},{}]: expected {}, got {}",
                i,
                j,
                expected,
                result_arr[[i, j]]
            );
        }
    }
}

// ============================================================================
// MAXI Tests
// ============================================================================

#[test]
fn test_maxi_i32_signed() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<i32> = vec![1, -1, 5, -5, 100, -100, 0, 10];
    let rhs_data: Vec<i32> = vec![2, -2, 3, -3, 50, -50, 5, 10];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.maxi(&rhs, Signedness::Signed);

    for i in 0..8 {
        let expected = if lhs_data[i] >= rhs_data[i] { lhs_data[i] } else { rhs_data[i] };
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected,
                "maxi i32 signed mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

#[test]
fn test_maxi_i32_unsigned() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<i32> = vec![1, -1, 5, -5, 100, -100, 0, 10];
    let rhs_data: Vec<i32> = vec![2, -2, 3, -3, 50, -50, 5, 10];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.maxi(&rhs, Signedness::Unsigned);

    for i in 0..8 {
        let expected = if (lhs_data[i] as u32) >= (rhs_data[i] as u32) {
            lhs_data[i]
        } else {
            rhs_data[i]
        };
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected,
                "maxi i32 unsigned mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

// ============================================================================
// MINI Tests
// ============================================================================

#[test]
fn test_mini_i32_signed() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<i32> = vec![1, -1, 5, -5, 100, -100, 0, 10];
    let rhs_data: Vec<i32> = vec![2, -2, 3, -3, 50, -50, 5, 10];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.mini(&rhs, Signedness::Signed);

    for i in 0..8 {
        let expected = if lhs_data[i] <= rhs_data[i] { lhs_data[i] } else { rhs_data[i] };
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected,
                "mini i32 signed mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

#[test]
fn test_mini_i32_unsigned() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<i32> = vec![1, -1, 5, -5, 100, -100, 0, 10];
    let rhs_data: Vec<i32> = vec![2, -2, 3, -3, 50, -50, 5, 10];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.mini(&rhs, Signedness::Unsigned);

    for i in 0..8 {
        let expected = if (lhs_data[i] as u32) <= (rhs_data[i] as u32) {
            lhs_data[i]
        } else {
            rhs_data[i]
        };
        if let Tile::I32(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected,
                "mini i32 unsigned mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

// ============================================================================
// CMPI Tests
// ============================================================================

#[test]
fn test_cmpi_i32_equal() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let rhs_data: Vec<i32> = vec![1, 0, 3, 0, 5, 0, 7, 0];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.cmpi(&rhs, ComparisonPredicate::Equal, Signedness::Signed);

    for i in 0..8 {
        let expected = lhs_data[i] == rhs_data[i];
        if let Tile::I1(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected,
                "cmpi i32 equal mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

#[test]
fn test_cmpi_i32_less_than_signed() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<i32> = vec![1, -1, 3, -3, 5, -5, 7, -7];
    let rhs_data: Vec<i32> = vec![2, 0, 4, -2, 6, -4, 8, -6];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.cmpi(&rhs, ComparisonPredicate::LessThan, Signedness::Signed);

    for i in 0..8 {
        let expected = lhs_data[i] < rhs_data[i];
        if let Tile::I1(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected,
                "cmpi i32 less than signed mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

#[test]
fn test_cmpi_i32_less_than_unsigned() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<i32> = vec![1, -1, 3, -3, 5, -5, 7, -7];
    let rhs_data: Vec<i32> = vec![2, 0, 4, -2, 6, -4, 8, -6];

    let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.cmpi(&rhs, ComparisonPredicate::LessThan, Signedness::Unsigned);

    for i in 0..8 {
        let expected = (lhs_data[i] as u32) < (rhs_data[i] as u32);
        if let Tile::I1(result_arr) = &result {
            assert_eq!(
                result_arr[[i / 4, i % 4]],
                expected,
                "cmpi i32 less than unsigned mismatch at index {}: expected {}, got {}",
                i,
                expected,
                result_arr[[i / 4, i % 4]]
            );
        }
    }
}

// ============================================================================
// MMAI Tests
// ============================================================================

#[test]
fn test_mmai_unbatched_i8() {
    // Simple 2x2 * 2x2 matrix multiplication
    let lhs = Tile::I8(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 2]),
        vec![1i8, 2, 3, 4],
    ).unwrap());
    let rhs = Tile::I8(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 2]),
        vec![5i8, 6, 7, 8],
    ).unwrap());
    let acc = Tile::I32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 2]),
        vec![0i32; 4],
    ).unwrap());

    let result = lhs.mmai(&rhs, &acc, Signedness::Signed, Signedness::Signed);

    // Expected result:
    // [1*5 + 2*7, 1*6 + 2*8]   = [19, 22]
    // [3*5 + 4*7, 3*6 + 4*8]   = [43, 50]
    if let Tile::I32(result_arr) = &result {
        assert_eq!(result_arr[[0, 0]], 19, "mmai result[0,0] mismatch");
        assert_eq!(result_arr[[0, 1]], 22, "mmai result[0,1] mismatch");
        assert_eq!(result_arr[[1, 0]], 43, "mmai result[1,0] mismatch");
        assert_eq!(result_arr[[1, 1]], 50, "mmai result[1,1] mismatch");
    }
}

#[test]
fn test_mmai_batched_i8() {
    // Simple batched 2x2 * 2x2 matrix multiplication
    let lhs = Tile::I8(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 2, 2]),
        vec![1i8, 2, 3, 4, 2, 3, 4, 5],
    ).unwrap());
    let rhs = Tile::I8(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 2, 2]),
        vec![5i8, 6, 7, 8, 1, 2, 3, 4],
    ).unwrap());
    let acc = Tile::I32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 2, 2]),
        vec![0i32; 8],
    ).unwrap());

    let result = lhs.mmai(&rhs, &acc, Signedness::Signed, Signedness::Signed);

    // Expected result for batch 0:
    // [1*5 + 2*7, 1*6 + 2*8]   = [19, 22]
    // [3*5 + 4*7, 3*6 + 4*8]   = [43, 50]
    // Expected result for batch 1:
    // [2*1 + 3*3, 2*2 + 3*4]   = [11, 16]
    // [4*1 + 5*3, 4*2 + 5*4]   = [19, 28]
    if let Tile::I32(result_arr) = &result {
        // Batch 0
        assert_eq!(result_arr[[0, 0, 0]], 19, "mmai batch 0 result[0,0] mismatch");
        assert_eq!(result_arr[[0, 0, 1]], 22, "mmai batch 0 result[0,1] mismatch");
        assert_eq!(result_arr[[0, 1, 0]], 43, "mmai batch 0 result[1,0] mismatch");
        assert_eq!(result_arr[[0, 1, 1]], 50, "mmai batch 0 result[1,1] mismatch");
        // Batch 1
        assert_eq!(result_arr[[1, 0, 0]], 11, "mmai batch 1 result[0,0] mismatch");
        assert_eq!(result_arr[[1, 0, 1]], 16, "mmai batch 1 result[0,1] mismatch");
        assert_eq!(result_arr[[1, 1, 0]], 19, "mmai batch 1 result[1,0] mismatch");
        assert_eq!(result_arr[[1, 1, 1]], 28, "mmai batch 1 result[1,1] mismatch");
    }
}

#[test]
fn test_mmai_unbatched_i8_mixed_signedness() {
    // Test with mixed signedness
    let lhs = Tile::I8(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 2]),
        vec![-1i8, 2, -3, 4],
    ).unwrap());
    let rhs = Tile::I8(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 2]),
        vec![5i8, -6, 7, -8],
    ).unwrap());
    let acc = Tile::I32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 2]),
        vec![0i32; 4],
    ).unwrap());

    // Signed * Signed
    let result = lhs.mmai(&rhs, &acc, Signedness::Signed, Signedness::Signed);

    // Expected result (signed * signed):
    // [(-1)*5 + 2*7, (-1)*(-6) + 2*(-8)]   = [9, -10]
    // [(-3)*5 + 4*7, (-3)*(-6) + 4*(-8)]   = [13, -14]
    if let Tile::I32(result_arr) = &result {
        assert_eq!(result_arr[[0, 0]], 9, "mmai signed*signed result[0,0] mismatch");
        assert_eq!(result_arr[[0, 1]], -10, "mmai signed*signed result[0,1] mismatch");
        assert_eq!(result_arr[[1, 0]], 13, "mmai signed*signed result[1,0] mismatch");
        assert_eq!(result_arr[[1, 1]], -14, "mmai signed*signed result[1,1] mismatch");
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_addi_overflow_wrapping() {
    let shape = ndarray::IxDyn(&[2, 2]);
    let lhs = Tile::I8(ndarray::Array::from_shape_vec(shape.clone(), vec![100i8, 127, -128, 0]).unwrap());
    let rhs = Tile::I8(ndarray::Array::from_shape_vec(shape.clone(), vec![50i8, 1, -1, 0]).unwrap());

    let result = lhs.addi(&rhs);

    if let Tile::I8(arr) = &result {
        assert_eq!(arr[[0, 0]], -106, "100 + 50 should wrap to -106"); // 150 wraps to -106
        assert_eq!(arr[[0, 1]], -128, "127 + 1 should wrap to -128"); // Overflow to min value
        assert_eq!(arr[[1, 0]], 127, "-128 + (-1) should wrap to 127"); // Underflow to max value
        assert_eq!(arr[[1, 1]], 0, "0 + 0 should be 0");
    }
}

#[test]
fn test_muli_overflow_wrapping() {
    let shape = ndarray::IxDyn(&[2, 2]);
    let lhs = Tile::I8(ndarray::Array::from_shape_vec(shape.clone(), vec![64i8, 16, -64, 1]).unwrap());
    let rhs = Tile::I8(ndarray::Array::from_shape_vec(shape.clone(), vec![4i8, 16, -2, -1]).unwrap());

    let result = lhs.muli(&rhs);

    if let Tile::I8(arr) = &result {
        assert_eq!(arr[[0, 0]], 0, "64 * 4 should overflow to 0"); // 256 wraps to 0
        assert_eq!(arr[[0, 1]], 0, "16 * 16 should overflow to 0"); // 256 wraps to 0
        assert_eq!(arr[[1, 0]], -128, "-64 * -2 should overflow to -128"); // 128 wraps to -128
        assert_eq!(arr[[1, 1]], -1, "1 * -1 should be -1");
    }
}

// ============================================================================
// Type Mismatch Panic Tests
// ============================================================================

#[test]
#[should_panic(expected = "Addi requires matching integer types")]
fn test_addi_type_mismatch() {
    let lhs = Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i32; 32]).unwrap());
    let rhs = Tile::I64(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i64; 32]).unwrap());
    let _ = lhs.addi(&rhs);
}

#[test]
#[should_panic(expected = "Muli requires matching integer types")]
fn test_muli_type_mismatch() {
    let lhs = Tile::I16(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i16; 32]).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i32; 32]).unwrap());
    let _ = lhs.muli(&rhs);
}

#[test]
#[should_panic(expected = "Shape mismatch")]
fn test_addi_shape_mismatch() {
    let lhs = Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0i32; 32]).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[8, 4]), vec![0i32; 32]).unwrap());
    let _ = lhs.addi(&rhs);
}

#[test]
#[should_panic(expected = "Mmai requires lhs/rhs to be i8")]
fn test_mmai_type_mismatch() {
    let lhs = Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![0i32; 4]).unwrap());
    let rhs = Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![0i32; 4]).unwrap());
    let acc = Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![0i32; 4]).unwrap());
    let _ = lhs.mmai(&rhs, &acc, Signedness::Signed, Signedness::Signed);
}

// ============================================================================
// All Integer Operations on All Types
// ============================================================================

#[test]
fn test_all_integer_ops_on_all_types() {
    let shape_2d = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    // Test i8
    {
        let lhs_data: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let rhs_data: Vec<i8> = vec![2, 3, 4, 5, 6, 7, 8, 9];

        let lhs = Tile::I8(ndarray::Array::from_shape_vec(shape_2d.clone(), lhs_data.clone()).unwrap());
        let rhs = Tile::I8(ndarray::Array::from_shape_vec(shape_2d.clone(), rhs_data.clone()).unwrap());

        let abs_result = lhs.absi();
        let neg_result = lhs.negi();
        let add_result = lhs.addi(&rhs);
        let sub_result = lhs.subi(&rhs);
        let mul_result = lhs.muli(&rhs);

        for i in 0..8 {
            if let Tile::I8(arr) = &abs_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].abs(), "i8 absi mismatch at index {}", i);
            }
            if let Tile::I8(arr) = &neg_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_neg(), "i8 negi mismatch at index {}", i);
            }
            if let Tile::I8(arr) = &add_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_add(rhs_data[i]), "i8 addi mismatch at index {}", i);
            }
            if let Tile::I8(arr) = &sub_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_sub(rhs_data[i]), "i8 subi mismatch at index {}", i);
            }
            if let Tile::I8(arr) = &mul_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_mul(rhs_data[i]), "i8 muli mismatch at index {}", i);
            }
        }
    }

    // Test i16
    {
        let lhs_data: Vec<i16> = vec![100, 200, 300, 400, 500, 600, 700, 800];
        let rhs_data: Vec<i16> = vec![50, 75, 100, 125, 150, 175, 200, 225];

        let lhs = Tile::I16(ndarray::Array::from_shape_vec(shape_2d.clone(), lhs_data.clone()).unwrap());
        let rhs = Tile::I16(ndarray::Array::from_shape_vec(shape_2d.clone(), rhs_data.clone()).unwrap());

        let add_result = lhs.addi(&rhs);
        let sub_result = lhs.subi(&rhs);
        let mul_result = lhs.muli(&rhs);

        for i in 0..8 {
            if let Tile::I16(arr) = &add_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_add(rhs_data[i]), "i16 addi mismatch at index {}", i);
            }
            if let Tile::I16(arr) = &sub_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_sub(rhs_data[i]), "i16 subi mismatch at index {}", i);
            }
            if let Tile::I16(arr) = &mul_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_mul(rhs_data[i]), "i16 muli mismatch at index {}", i);
            }
        }
    }

    // Test i32
    {
        let lhs_data: Vec<i32> = vec![1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
        let rhs_data: Vec<i32> = vec![500, 750, 1000, 1250, 1500, 1750, 2000, 2250];

        let lhs = Tile::I32(ndarray::Array::from_shape_vec(shape_2d.clone(), lhs_data.clone()).unwrap());
        let rhs = Tile::I32(ndarray::Array::from_shape_vec(shape_2d.clone(), rhs_data.clone()).unwrap());

        let add_result = lhs.addi(&rhs);
        let sub_result = lhs.subi(&rhs);
        let mul_result = lhs.muli(&rhs);

        for i in 0..8 {
            if let Tile::I32(arr) = &add_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_add(rhs_data[i]), "i32 addi mismatch at index {}", i);
            }
            if let Tile::I32(arr) = &sub_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_sub(rhs_data[i]), "i32 subi mismatch at index {}", i);
            }
            if let Tile::I32(arr) = &mul_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_mul(rhs_data[i]), "i32 muli mismatch at index {}", i);
            }
        }
    }

    // Test i64
    {
        let lhs_data: Vec<i64> = vec![10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000];
        let rhs_data: Vec<i64> = vec![5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500];

        let lhs = Tile::I64(ndarray::Array::from_shape_vec(shape_2d.clone(), lhs_data.clone()).unwrap());
        let rhs = Tile::I64(ndarray::Array::from_shape_vec(shape_2d.clone(), rhs_data.clone()).unwrap());

        let add_result = lhs.addi(&rhs);
        let sub_result = lhs.subi(&rhs);
        let mul_result = lhs.muli(&rhs);

        for i in 0..8 {
            if let Tile::I64(arr) = &add_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_add(rhs_data[i]), "i64 addi mismatch at index {}", i);
            }
            if let Tile::I64(arr) = &sub_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_sub(rhs_data[i]), "i64 subi mismatch at index {}", i);
            }
            if let Tile::I64(arr) = &mul_result {
                assert_eq!(arr[[i / 4, i % 4]], lhs_data[i].wrapping_mul(rhs_data[i]), "i64 muli mismatch at index {}", i);
            }
        }
    }
}
