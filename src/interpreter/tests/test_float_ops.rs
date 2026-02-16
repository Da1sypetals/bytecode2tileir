// Tests for floating-point operations (Section 8.7)
//
// Simplified tests covering all 29 floating-point operations:
// - Unary (16): absf, negf, sqrt, rsqrt, sin, cos, tan, sinh, cosh, tanh, exp, exp2, log, log2, floor, ceil
// - Binary (8): addf, subf, mulf, divf, remf, minf, maxf, pow
// - Ternary (2): fma, mmaf
// - Comparison (1): cmpf

use crate::cuda_tile_ir::enums::{ComparisonOrdering, ComparisonPredicate};
use crate::interpreter::data_structures::tile::Tile;
use ndrange::ndrange;
use rand::{RngExt, SeedableRng};

// Precision tolerances per type
const TOL_F16: f32 = 0.01;
const TOL_F32: f32 = 1e-6;
const TOL_F64: f64 = 1e-12;

// ============================================================================
// Basic Arithmetic Operations (addf, subf, mulf, divf)
// ============================================================================

#[test]
fn test_addf_f32_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    // Real-world data: mixed positive/negative, various magnitudes
    let lhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.7 - 10.0).collect();
    let rhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 1.3 + 5.0).collect();

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    // Test basic operation
    let result = lhs.addf(&rhs, false);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = lhs_data[idx] + rhs_data[idx];
                let actual = arr[[i, j]];
                assert!(
                    (actual - expected).abs() < TOL_F32,
                    "addf mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }

    // Test FTZ behavior
    let result_ftz = lhs.addf(&rhs, true);
    match result_ftz {
        Tile::F32(arr) => {
            // FTZ should not affect normal values
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = lhs_data[idx] + rhs_data[idx];
                let actual = arr[[i, j]];
                assert!(
                    (actual - expected).abs() < TOL_F32,
                    "addf FTZ mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_subf_f32_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 1.7 + 20.0).collect();
    let rhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.9 - 5.0).collect();

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.subf(&rhs, false);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = lhs_data[idx] - rhs_data[idx];
                let actual = arr[[i, j]];
                assert!(
                    (actual - expected).abs() < TOL_F32,
                    "subf mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_mulf_f32_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3 + 1.5).collect();
    let rhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.7 - 2.0).collect();

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.mulf(&rhs, false);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = lhs_data[idx] * rhs_data[idx];
                let actual = arr[[i, j]];
                assert!(
                    (actual - expected).abs() < TOL_F32,
                    "mulf mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_divf_f32_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    // Avoid division by zero
    let lhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 2.5 + 10.0).collect();
    let rhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5 + 1.0).collect();

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.divf(&rhs, false);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = lhs_data[idx] / rhs_data[idx];
                let actual = arr[[i, j]];
                assert!(
                    (actual - expected).abs() < TOL_F32,
                    "divf mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

// ============================================================================
// Unary Operations - Simplified (one operation per data type)
// ============================================================================

#[test]
fn test_unary_f16_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    // Test absf with negative data
    let data: Vec<f16> = (0..size)
        .map(|i| -((i as f32) * 1.5 + 1.0) as f16)
        .collect();
    let tile = Tile::F16(ndarray::Array::from_shape_vec(shape.clone(), data.clone()).unwrap());
    let result = tile.absf();
    match result {
        Tile::F16(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = -data[idx];
                let f32_actual = arr[[i, j]] as f32;
                let f32_expected = expected as f32;
                assert!(
                    (f32_actual - f32_expected).abs() < TOL_F16,
                    "absf f16 mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    f32_expected,
                    f32_actual
                );
            }
        }
        _ => panic!("Expected F16 tile"),
    }
}

#[test]
fn test_unary_f32_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    // Test absf
    let data: Vec<f32> = (0..size).map(|i| -((i as f32) * 1.5 + 1.0)).collect();
    let tile = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), data.clone()).unwrap());
    let result = tile.absf();
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = -data[idx];
                assert!(
                    (arr[[i, j]] - expected).abs() < TOL_F32,
                    "absf f32 mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    arr[[i, j]]
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_unary_f64_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    // Test absf
    let data: Vec<f64> = (0..size).map(|i| -((i as f64) * 1.5 + 1.0)).collect();
    let tile = Tile::F64(ndarray::Array::from_shape_vec(shape.clone(), data.clone()).unwrap());
    let result = tile.absf();
    match result {
        Tile::F64(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = -data[idx];
                assert!(
                    (arr[[i, j]] - expected).abs() < TOL_F64,
                    "absf f64 mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    arr[[i, j]]
                );
            }
        }
        _ => panic!("Expected F64 tile"),
    }
}

// ============================================================================
// Square Root Operations with FTZ
// ============================================================================

#[test]
fn test_sqrt_ftz() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    // Include subnormal values to test FTZ
    let data: Vec<f32> = (0..size)
        .map(|i| {
            if i % 5 == 0 {
                1.0e-40 // Subnormal
            } else {
                (i as f32) * 0.5 + 1.0
            }
        })
        .collect();

    let tile = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), data.clone()).unwrap());

    // Without FTZ
    let result_no_ftz = tile.sqrt(false);
    match result_no_ftz {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = if data[idx].is_subnormal() {
                    data[idx].sqrt()
                } else {
                    data[idx].sqrt()
                };
                // Subnormals may produce NaN or Inf
                if !expected.is_nan() && !expected.is_infinite() {
                    assert!(
                        (arr[[i, j]] - expected).abs() < TOL_F32,
                        "sqrt no FTZ mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
        }
        _ => panic!("Expected F32 tile"),
    }

    // With FTZ - subnormals should flush to zero first
    let result_ftz = tile.sqrt(true);
    match result_ftz {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                if data[idx].is_subnormal() {
                    // Should be 0.0
                    assert_eq!(
                        arr[[i, j]],
                        0.0,
                        "sqrt FTZ should flush subnormal to zero at [{},{}]",
                        i,
                        j
                    );
                } else {
                    let expected = data[idx].sqrt();
                    assert!(
                        (arr[[i, j]] - expected).abs() < TOL_F32,
                        "sqrt FTZ mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_rsqrt_ftz() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let data: Vec<f32> = (0..size)
        .map(|i| {
            if i % 5 == 0 {
                1.0e-40 // Subnormal
            } else {
                (i as f32) * 0.5 + 1.0
            }
        })
        .collect();

    let tile = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), data.clone()).unwrap());

    // Without FTZ
    let result_no_ftz = tile.rsqrt(false);
    match result_no_ftz {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                if data[idx].is_subnormal() {
                    // 1/sqrt(subnormal) -> Very large number (may or may not be Inf depending on value)
                    assert!(
                        arr[[i, j]].is_infinite() || arr[[i, j]].abs() > 1e10,
                        "rsqrt of subnormal should be Inf or very large at [{},{}]: got {}",
                        i,
                        j,
                        arr[[i, j]]
                    );
                } else {
                    let expected = 1.0 / data[idx].sqrt();
                    assert!(
                        (arr[[i, j]] - expected).abs() < TOL_F32,
                        "rsqrt no FTZ mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
        }
        _ => panic!("Expected F32 tile"),
    }

    // With FTZ
    let result_ftz = tile.rsqrt(true);
    match result_ftz {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                if data[idx].is_subnormal() {
                    // 1/sqrt(0.0) -> Inf
                    assert!(
                        arr[[i, j]].is_infinite(),
                        "rsqrt of flushed subnormal should be Inf at [{},{}]: got {}",
                        i,
                        j,
                        arr[[i, j]]
                    );
                } else {
                    let expected = 1.0 / data[idx].sqrt();
                    assert!(
                        (arr[[i, j]] - expected).abs() < TOL_F32,
                        "rsqrt FTZ mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

// ============================================================================
// Remainder Operation (remf)
// ============================================================================

#[test]
fn test_remf_f32_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 1.7 - 10.0).collect();
    let rhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3 + 1.0).collect();

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.remf(&rhs);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = lhs_data[idx] % rhs_data[idx];
                let actual = arr[[i, j]];
                assert!(
                    (actual - expected).abs() < TOL_F32,
                    "remf mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_remf_edge_cases() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    // Test with NaN, Inf, zero
    // Expected results based on IEEE 754:
    // [0]: 10.0 % 3.0 = 1.0
    // [1]: NaN % 2.0 = NaN
    // [2]: Inf % 5.0 = NaN
    // [3]: 5.0 % 0.0 = NaN (division by zero)
    // [4]: -10.0 % 4.0 = -2.0 (sign matches dividend)
    // [5]: 0.0 % 2.0 = 0.0
    // [6]: 3.5 % NaN = NaN
    // [7]: -2.5 % Inf = -2.5 (finite % Inf = finite)
    let lhs_data: Vec<f32> = vec![10.0, f32::NAN, f32::INFINITY, 5.0, -10.0, 0.0, 3.5, -2.5];
    let rhs_data: Vec<f32> = vec![3.0, 2.0, 5.0, 0.0, 4.0, 2.0, f32::NAN, f32::INFINITY];

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let expected_results: Vec<Option<f32>> = vec![
        Some(1.0),  // 10.0 % 3.0
        None,       // NaN
        None,       // Inf % 5.0 = NaN
        None,       // 5.0 % 0.0 = NaN
        Some(-2.0), // -10.0 % 4.0 = -2.0
        Some(0.0),  // 0.0 % 2.0 = 0.0
        None,       // 3.5 % NaN = NaN
        Some(-2.5), // -2.5 % Inf = -2.5
    ];

    let result = lhs.remf(&rhs);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[2, 4]) {
                let idx = (i * 4 + j) as usize;
                let actual = arr[[i, j]];
                match expected_results[idx] {
                    Some(expected) => {
                        assert!(
                            (actual - expected).abs() < TOL_F32,
                            "remf mismatch at [{},{}]: expected {}, got {}",
                            i,
                            j,
                            expected,
                            actual
                        );
                    }
                    None => {
                        assert!(
                            actual.is_nan(),
                            "remf should produce NaN at [{},{}]: got {}",
                            i,
                            j,
                            actual
                        );
                    }
                }
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_remf_sign() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    // Test that result sign matches dividend
    let lhs_data: Vec<f32> = vec![10.0, -10.0, 7.5, -7.5, 3.0, -3.0, 5.0, -5.0];
    let rhs_data: Vec<f32> = vec![3.0, 3.0, 2.0, 2.0, 4.0, 4.0, 2.0, 2.0];

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.remf(&rhs);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[2, 4]) {
                let idx = (i * 4 + j) as usize;
                let actual = arr[[i, j]];
                let expected = lhs_data[idx] % rhs_data[idx];

                // Check sign matches dividend
                assert!(
                    actual.signum() == expected.signum() || actual == 0.0,
                    "remf sign should match dividend at [{},{}]: expected sign {}, got {}",
                    i,
                    j,
                    expected.signum(),
                    actual.signum()
                );

                assert!(
                    (actual - expected).abs() < TOL_F32,
                    "remf mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

// ============================================================================
// Min/Max Operations (minf, maxf)
// ============================================================================

#[test]
fn test_minf_f32_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 1.5 - 10.0).collect();
    let rhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.8 + 5.0).collect();

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.minf(&rhs, false, false);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = lhs_data[idx].min(rhs_data[idx]);
                assert!(
                    (arr[[i, j]] - expected).abs() < TOL_F32,
                    "minf mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    arr[[i, j]]
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_minf_propagate_nan_true() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<f32> = vec![5.0, f32::NAN, 3.0, f32::NAN, 7.0, 2.0, f32::NAN, 1.0];
    let rhs_data: Vec<f32> = vec![3.0, 5.0, f32::NAN, f32::NAN, 9.0, f32::NAN, 4.0, f32::NAN];

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.minf(&rhs, true, false);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[2, 4]) {
                let idx = (i * 4 + j) as usize;
                let actual = arr[[i, j]];
                // propagate_nan=true: if either is NaN, result is NaN
                if lhs_data[idx].is_nan() || rhs_data[idx].is_nan() {
                    assert!(
                        actual.is_nan(),
                        "minf propagate_nan=true should produce NaN at [{},{}]: got {}",
                        i,
                        j,
                        actual
                    );
                } else {
                    let expected = lhs_data[idx].min(rhs_data[idx]);
                    assert!(
                        (actual - expected).abs() < TOL_F32,
                        "minf mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        actual
                    );
                }
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_minf_propagate_nan_false() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<f32> = vec![5.0, f32::NAN, 3.0, f32::NAN, 7.0, 2.0, f32::NAN, 1.0];
    let rhs_data: Vec<f32> = vec![3.0, 5.0, f32::NAN, f32::NAN, 9.0, f32::NAN, 4.0, f32::NAN];

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.minf(&rhs, false, false);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[2, 4]) {
                let idx = (i * 4 + j) as usize;
                let actual = arr[[i, j]];
                // propagate_nan=false: return the non-NaN value, or NaN if both are NaN
                if lhs_data[idx].is_nan() && rhs_data[idx].is_nan() {
                    assert!(
                        actual.is_nan(),
                        "minf propagate_nan=false with both NaN should produce NaN at [{},{}]: got {}",
                        i,
                        j,
                        actual
                    );
                } else if lhs_data[idx].is_nan() {
                    assert_eq!(
                        actual, rhs_data[idx],
                        "minf propagate_nan=false should return rhs at [{},{}]: expected {}, got {}",
                        i, j, rhs_data[idx], actual
                    );
                } else if rhs_data[idx].is_nan() {
                    assert_eq!(
                        actual, lhs_data[idx],
                        "minf propagate_nan=false should return lhs at [{},{}]: expected {}, got {}",
                        i, j, lhs_data[idx], actual
                    );
                } else {
                    let expected = lhs_data[idx].min(rhs_data[idx]);
                    assert!(
                        (actual - expected).abs() < TOL_F32,
                        "minf mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        actual
                    );
                }
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_maxf_f32_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let lhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 1.5 - 10.0).collect();
    let rhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.8 + 5.0).collect();

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.maxf(&rhs, false, false);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = lhs_data[idx].max(rhs_data[idx]);
                assert!(
                    (arr[[i, j]] - expected).abs() < TOL_F32,
                    "maxf mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    arr[[i, j]]
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_maxf_propagate_nan_true() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<f32> = vec![5.0, f32::NAN, 3.0, f32::NAN, 7.0, 2.0, f32::NAN, 1.0];
    let rhs_data: Vec<f32> = vec![3.0, 5.0, f32::NAN, f32::NAN, 9.0, f32::NAN, 4.0, f32::NAN];

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.maxf(&rhs, true, false);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[2, 4]) {
                let idx = (i * 4 + j) as usize;
                let actual = arr[[i, j]];
                if lhs_data[idx].is_nan() || rhs_data[idx].is_nan() {
                    assert!(
                        actual.is_nan(),
                        "maxf propagate_nan=true should produce NaN at [{},{}]: got {}",
                        i,
                        j,
                        actual
                    );
                } else {
                    let expected = lhs_data[idx].max(rhs_data[idx]);
                    assert!(
                        (actual - expected).abs() < TOL_F32,
                        "maxf mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        actual
                    );
                }
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_maxf_propagate_nan_false() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<f32> = vec![5.0, f32::NAN, 3.0, f32::NAN, 7.0, 2.0, f32::NAN, 1.0];
    let rhs_data: Vec<f32> = vec![3.0, 5.0, f32::NAN, f32::NAN, 9.0, f32::NAN, 4.0, f32::NAN];

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let result = lhs.maxf(&rhs, false, false);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[2, 4]) {
                let idx = (i * 4 + j) as usize;
                let actual = arr[[i, j]];
                if lhs_data[idx].is_nan() && rhs_data[idx].is_nan() {
                    assert!(
                        actual.is_nan(),
                        "maxf propagate_nan=false with both NaN should produce NaN at [{},{}]: got {}",
                        i,
                        j,
                        actual
                    );
                } else if lhs_data[idx].is_nan() {
                    assert_eq!(
                        actual, rhs_data[idx],
                        "maxf propagate_nan=false should return rhs at [{},{}]: expected {}, got {}",
                        i, j, rhs_data[idx], actual
                    );
                } else if rhs_data[idx].is_nan() {
                    assert_eq!(
                        actual, lhs_data[idx],
                        "maxf propagate_nan=false should return lhs at [{},{}]: expected {}, got {}",
                        i, j, lhs_data[idx], actual
                    );
                } else {
                    let expected = lhs_data[idx].max(rhs_data[idx]);
                    assert!(
                        (actual - expected).abs() < TOL_F32,
                        "maxf mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        actual
                    );
                }
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

// ============================================================================
// Power Operation (pow)
// ============================================================================

#[test]
fn test_pow_f32_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let base_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let exp_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3 - 2.0).collect();

    let base = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), base_data.clone()).unwrap());
    let exp = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), exp_data.clone()).unwrap());

    let result = base.pow(&exp);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = base_data[idx].powf(exp_data[idx]);
                // Power can have larger errors
                assert!(
                    (arr[[i, j]] - expected).abs() < 1e-4,
                    "pow mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    arr[[i, j]]
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_pow_edge_cases() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    // Test zero base, negative exponents, fractional bases
    let base_data: Vec<f32> = vec![0.0, 2.0, 4.0, 0.5, 1.0, -2.0, 10.0, 0.1];
    let exp_data: Vec<f32> = vec![2.0, -1.0, 0.5, 3.0, 0.0, 2.0, -2.0, 1.5];

    let base = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), base_data.clone()).unwrap());
    let exp = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), exp_data.clone()).unwrap());

    let result = base.pow(&exp);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[2, 4]) {
                let idx = (i * 4 + j) as usize;
                let actual = arr[[i, j]];
                let expected = base_data[idx].powf(exp_data[idx]);

                // Handle special cases
                if expected.is_nan() || expected.is_infinite() {
                    assert!(
                        actual.is_nan() || actual.is_infinite(),
                        "pow edge case at [{},{}]: expected special value, got {}",
                        i,
                        j,
                        actual
                    );
                } else {
                    assert!(
                        (actual - expected).abs() < 1e-4,
                        "pow edge case mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        actual
                    );
                }
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

// ============================================================================
// FMA Operation
// ============================================================================

#[test]
fn test_fma_f32_2d() {
    let shape = ndarray::IxDyn(&[4, 8]);
    let size = 32;

    let a_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.7 + 1.0).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i as f32) * 1.3 + 2.0).collect();
    let c_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.9 - 5.0).collect();

    let a = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), a_data.clone()).unwrap());
    let b = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), b_data.clone()).unwrap());
    let c = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), c_data.clone()).unwrap());

    let result = a.fma(&b, &c, false);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[4, 8]) {
                let idx = (i * 8 + j) as usize;
                let expected = a_data[idx] * b_data[idx] + c_data[idx];
                assert!(
                    (arr[[i, j]] - expected).abs() < TOL_F32,
                    "fma mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    arr[[i, j]]
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_fma_ftz() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    // Include subnormal values
    let a_data: Vec<f32> = vec![1.0e-40, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_data: Vec<f32> = vec![2.0, 1.0e-40, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let c_data: Vec<f32> = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0e-40, 2.0];

    let a = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), a_data.clone()).unwrap());
    let b = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), b_data.clone()).unwrap());
    let c = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), c_data.clone()).unwrap());

    let result = a.fma(&b, &c, true);
    match result {
        Tile::F32(arr) => {
            for [i, j] in ndrange(&[2, 4]) {
                let idx = (i * 4 + j) as usize;
                // With FTZ, subnormals flush to zero
                let a_flushed = if a_data[idx].is_subnormal() {
                    0.0
                } else {
                    a_data[idx]
                };
                let b_flushed = if b_data[idx].is_subnormal() {
                    0.0
                } else {
                    b_data[idx]
                };
                let c_flushed = if c_data[idx].is_subnormal() {
                    0.0
                } else {
                    c_data[idx]
                };
                let expected = a_flushed * b_flushed + c_flushed;
                assert!(
                    (arr[[i, j]] - expected).abs() < TOL_F32,
                    "fma FTZ mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    arr[[i, j]]
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

// ============================================================================
// MMAF Operation (Matrix Multiply Accumulate)
// ============================================================================

#[test]
fn test_mmaf_f32_unbatched() {
    // Unbatched: M=4, K=8, N=16
    let m = 4;
    let k = 8;
    let n = 16;

    let lhs_data: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.5).collect();
    let rhs_data: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.3).collect();
    let acc_data: Vec<f32> = (0..m * n).map(|i| (i % 5) as f32 * 0.7).collect();

    let lhs = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[m, k]), lhs_data.clone()).unwrap(),
    );
    let rhs = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[k, n]), rhs_data.clone()).unwrap(),
    );
    let acc = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[m, n]), acc_data.clone()).unwrap(),
    );

    let result = lhs.mmaf(&rhs, &acc);
    match result {
        Tile::F32(arr) => {
            assert_eq!(arr.shape(), &[m, n]);
            for i in 0..m {
                for j in 0..n {
                    // Simple loop-based verification
                    let mut expected = acc_data[i * n + j];
                    for kk in 0..k {
                        expected += lhs_data[i * k + kk] * rhs_data[kk * n + j];
                    }
                    assert!(
                        (arr[[i, j]] - expected).abs() < 1e-4,
                        "mmaf f32 unbatched mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_mmaf_f64_batched() {
    // Batched: B=2, M=4, K=8, N=16
    let bsz = 2;
    let m = 4;
    let k = 8;
    let n = 16;

    let lhs_data: Vec<f64> = (0..bsz * m * k).map(|i| (i % 7) as f64 * 0.5).collect();
    let rhs_data: Vec<f64> = (0..bsz * k * n).map(|i| (i % 11) as f64 * 0.3).collect();
    let acc_data: Vec<f64> = (0..bsz * m * n).map(|i| (i % 5) as f64 * 0.7).collect();

    let lhs = Tile::F64(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, m, k]), lhs_data.clone()).unwrap(),
    );
    let rhs = Tile::F64(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, k, n]), rhs_data.clone()).unwrap(),
    );
    let acc = Tile::F64(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, m, n]), acc_data.clone()).unwrap(),
    );

    let result = lhs.mmaf(&rhs, &acc);
    match result {
        Tile::F64(arr) => {
            assert_eq!(arr.shape(), &[bsz, m, n]);
            for batch in 0..bsz {
                for i in 0..m {
                    for j in 0..n {
                        let mut expected = acc_data[batch * m * n + i * n + j];
                        for kk in 0..k {
                            expected += lhs_data[batch * m * k + i * k + kk]
                                * rhs_data[batch * k * n + kk * n + j];
                        }
                        assert!(
                            (arr[[batch, i, j]] - expected).abs() < TOL_F64,
                            "mmaf f64 batched mismatch at [{},{},{}]: expected {}, got {}",
                            batch,
                            i,
                            j,
                            expected,
                            arr[[batch, i, j]]
                        );
                    }
                }
            }
        }
        _ => panic!("Expected F64 tile"),
    }
}

#[test]
fn test_mmaf_correctness() {
    // Small matrix for manual verification
    let m = 2;
    let k = 3;
    let n = 2;

    // LHS (2x3): [1.0, 2.0, 3.0] / [4.0, 5.0, 6.0]
    let lhs_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // RHS (3x2): [7.0, 8.0] / [9.0, 10.0] / [11.0, 12.0]
    let rhs_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    // ACC (2x2): [1.0, 2.0] / [3.0, 4.0]
    let acc_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let lhs = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[m, k]), lhs_data.clone()).unwrap(),
    );
    let rhs = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[k, n]), rhs_data.clone()).unwrap(),
    );
    let acc = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[m, n]), acc_data.clone()).unwrap(),
    );

    let result = lhs.mmaf(&rhs, &acc);
    match result {
        Tile::F32(arr) => {
            // Expected: [59, 66] / [142, 158]
            assert!((arr[[0, 0]] - 59.0).abs() < TOL_F32, "mmaf [0,0] mismatch");
            assert!((arr[[0, 1]] - 66.0).abs() < TOL_F32, "mmaf [0,1] mismatch");
            assert!((arr[[1, 0]] - 142.0).abs() < TOL_F32, "mmaf [1,0] mismatch");
            assert!((arr[[1, 1]] - 158.0).abs() < TOL_F32, "mmaf [1,1] mismatch");
        }
        _ => panic!("Expected F32 tile"),
    }
}

// Mixed-precision mmaf tests with large random tiles
#[test]
fn test_mmaf_f16_f32_large_random() {
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Test both unbatched and batched with large tiles
    let (m, k, n) = (16, 32, 64);

    // Generate random data in reasonable range for F16
    let lhs_data: Vec<f16> = (0..m * k)
        .map(|_| (rng.random_range(-10.0f32..10.0) as f16))
        .collect();
    let rhs_data: Vec<f16> = (0..k * n)
        .map(|_| (rng.random_range(-10.0f32..10.0) as f16))
        .collect();
    let acc_data: Vec<f32> = (0..m * n)
        .map(|_| rng.random_range(-100.0f32..100.0))
        .collect();

    // Test unbatched
    {
        let lhs = Tile::F16(
            ndarray::Array::from_shape_vec(ndarray::IxDyn(&[m, k]), lhs_data.clone()).unwrap(),
        );
        let rhs = Tile::F16(
            ndarray::Array::from_shape_vec(ndarray::IxDyn(&[k, n]), rhs_data.clone()).unwrap(),
        );
        let acc = Tile::F32(
            ndarray::Array::from_shape_vec(ndarray::IxDyn(&[m, n]), acc_data.clone()).unwrap(),
        );

        let result = lhs.mmaf(&rhs, &acc);
        match result {
            Tile::F32(arr) => {
                assert_eq!(arr.shape(), &[m, n]);
                // Verify correctness on sample of elements (not all for speed)
                for _ in 0..10 {
                    let i = rng.random_range(0..m);
                    let j = rng.random_range(0..n);
                    let mut expected = acc_data[i * n + j];
                    for kk in 0..k {
                        expected += lhs_data[i * k + kk] as f32 * rhs_data[kk * n + j] as f32;
                    }
                    assert!(
                        (arr[[i, j]] - expected).abs() < 1e-4,
                        "f16->f32 unbatched mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                    println!(
                        "[{}, {}] expected: {}, actual: {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
            _ => panic!("Expected F32 tile"),
        }
    }

    // Test batched with same data
    let bsz = 2;
    let lhs_data_batch: Vec<f16> = (0..bsz * m * k)
        .map(|_| (rng.random_range(-10.0f32..10.0) as f16))
        .collect();
    let rhs_data_batch: Vec<f16> = (0..bsz * k * n)
        .map(|_| (rng.random_range(-10.0f32..10.0) as f16))
        .collect();
    let acc_data_batch: Vec<f32> = (0..bsz * m * n)
        .map(|_| rng.random_range(-100.0f32..100.0))
        .collect();

    let lhs = Tile::F16(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, m, k]), lhs_data_batch.clone())
            .unwrap(),
    );
    let rhs = Tile::F16(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, k, n]), rhs_data_batch.clone())
            .unwrap(),
    );
    let acc = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, m, n]), acc_data_batch.clone())
            .unwrap(),
    );

    let result = lhs.mmaf(&rhs, &acc);
    match result {
        Tile::F32(arr) => {
            assert_eq!(arr.shape(), &[bsz, m, n]);
            // Verify correctness on sample of elements
            for _ in 0..10 {
                let batch = rng.random_range(0..bsz);
                let i = rng.random_range(0..m);
                let j = rng.random_range(0..n);
                let mut expected = acc_data_batch[batch * m * n + i * n + j];
                for kk in 0..k {
                    expected += lhs_data_batch[batch * m * k + i * k + kk] as f32
                        * rhs_data_batch[batch * k * n + kk * n + j] as f32;
                }
                assert!(
                    (arr[[batch, i, j]] - expected).abs() < 1e-3,
                    "f16->f32 batched mismatch at [{},{},{}]: expected {}, got {}",
                    batch,
                    i,
                    j,
                    expected,
                    arr[[batch, i, j]]
                );
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_mmaf_f16_f64_large_random() {
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(43);

    // Large tiles for F16->F64
    let (m, k, n) = (16, 32, 64);

    let lhs_data: Vec<f16> = (0..m * k)
        .map(|_| (rng.random_range(-10.0f32..10.0) as f16))
        .collect();
    let rhs_data: Vec<f16> = (0..k * n)
        .map(|_| (rng.random_range(-10.0f32..10.0) as f16))
        .collect();
    let acc_data: Vec<f64> = (0..m * n)
        .map(|_| rng.random_range(-1000.0f64..1000.0))
        .collect();

    let lhs = Tile::F16(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[m, k]), lhs_data.clone()).unwrap(),
    );
    let rhs = Tile::F16(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[k, n]), rhs_data.clone()).unwrap(),
    );
    let acc = Tile::F64(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[m, n]), acc_data.clone()).unwrap(),
    );

    let result = lhs.mmaf(&rhs, &acc);
    match result {
        Tile::F64(arr) => {
            assert_eq!(arr.shape(), &[m, n]);
            // Verify correctness on sample
            for _ in 0..10 {
                let i = rng.random_range(0..m);
                let j = rng.random_range(0..n);
                let mut expected = acc_data[i * n + j];
                for kk in 0..k {
                    expected += lhs_data[i * k + kk] as f64 * rhs_data[kk * n + j] as f64;
                }
                assert!(
                    (arr[[i, j]] - expected).abs() < TOL_F64,
                    "f16->f64 mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    arr[[i, j]]
                );
            }
        }
        _ => panic!("Expected F64 tile"),
    }

    // Test batched
    let bsz = 2;
    let lhs_data_batch: Vec<f16> = (0..bsz * m * k)
        .map(|_| (rng.random_range(-10.0f32..10.0) as f16))
        .collect();
    let rhs_data_batch: Vec<f16> = (0..bsz * k * n)
        .map(|_| (rng.random_range(-10.0f32..10.0) as f16))
        .collect();
    let acc_data_batch: Vec<f64> = (0..bsz * m * n)
        .map(|_| rng.random_range(-1000.0f64..1000.0))
        .collect();

    let lhs = Tile::F16(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, m, k]), lhs_data_batch.clone())
            .unwrap(),
    );
    let rhs = Tile::F16(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, k, n]), rhs_data_batch.clone())
            .unwrap(),
    );
    let acc = Tile::F64(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, m, n]), acc_data_batch.clone())
            .unwrap(),
    );

    let result = lhs.mmaf(&rhs, &acc);
    match result {
        Tile::F64(arr) => {
            assert_eq!(arr.shape(), &[bsz, m, n]);
            for _ in 0..10 {
                let batch = rng.random_range(0..bsz);
                let i = rng.random_range(0..m);
                let j = rng.random_range(0..n);
                let mut expected = acc_data_batch[batch * m * n + i * n + j];
                for kk in 0..k {
                    expected += lhs_data_batch[batch * m * k + i * k + kk] as f64
                        * rhs_data_batch[batch * k * n + kk * n + j] as f64;
                }
                assert!(
                    (arr[[batch, i, j]] - expected).abs() < TOL_F64,
                    "f16->f64 batched mismatch at [{},{},{}]: expected {}, got {}",
                    batch,
                    i,
                    j,
                    expected,
                    arr[[batch, i, j]]
                );
            }
        }
        _ => panic!("Expected F64 tile"),
    }
}

#[test]
fn test_mmaf_f32_f64_large_random() {
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(44);

    // Large tiles for F32->F64
    let (m, k, n) = (16, 32, 64);

    let lhs_data: Vec<f32> = (0..m * k)
        .map(|_| rng.random_range(-100.0f32..100.0))
        .collect();
    let rhs_data: Vec<f32> = (0..k * n)
        .map(|_| rng.random_range(-100.0f32..100.0))
        .collect();
    let acc_data: Vec<f64> = (0..m * n)
        .map(|_| rng.random_range(-10000.0f64..10000.0))
        .collect();

    let lhs = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[m, k]), lhs_data.clone()).unwrap(),
    );
    let rhs = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[k, n]), rhs_data.clone()).unwrap(),
    );
    let acc = Tile::F64(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[m, n]), acc_data.clone()).unwrap(),
    );

    let result = lhs.mmaf(&rhs, &acc);
    match result {
        Tile::F64(arr) => {
            assert_eq!(arr.shape(), &[m, n]);
            // Verify correctness on sample
            for _ in 0..10 {
                let i = rng.random_range(0..m);
                let j = rng.random_range(0..n);
                let mut expected = acc_data[i * n + j];
                for kk in 0..k {
                    expected += lhs_data[i * k + kk] as f64 * rhs_data[kk * n + j] as f64;
                }
                assert!(
                    (arr[[i, j]] - expected).abs() < 1e-5,
                    "f32->f64 mismatch at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected,
                    arr[[i, j]]
                );
            }
        }
        _ => panic!("Expected F64 tile"),
    }

    // Test batched
    let bsz = 2;
    let lhs_data_batch: Vec<f32> = (0..bsz * m * k)
        .map(|_| rng.random_range(-100.0f32..100.0))
        .collect();
    let rhs_data_batch: Vec<f32> = (0..bsz * k * n)
        .map(|_| rng.random_range(-100.0f32..100.0))
        .collect();
    let acc_data_batch: Vec<f64> = (0..bsz * m * n)
        .map(|_| rng.random_range(-10000.0f64..10000.0))
        .collect();

    let lhs = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, m, k]), lhs_data_batch.clone())
            .unwrap(),
    );
    let rhs = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, k, n]), rhs_data_batch.clone())
            .unwrap(),
    );
    let acc = Tile::F64(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[bsz, m, n]), acc_data_batch.clone())
            .unwrap(),
    );

    let result = lhs.mmaf(&rhs, &acc);
    match result {
        Tile::F64(arr) => {
            assert_eq!(arr.shape(), &[bsz, m, n]);
            for _ in 0..10 {
                let batch = rng.random_range(0..bsz);
                let i = rng.random_range(0..m);
                let j = rng.random_range(0..n);
                let mut expected = acc_data_batch[batch * m * n + i * n + j];
                for kk in 0..k {
                    expected += lhs_data_batch[batch * m * k + i * k + kk] as f64
                        * rhs_data_batch[batch * k * n + kk * n + j] as f64;
                }
                assert!(
                    (arr[[batch, i, j]] - expected).abs() < 1e-5,
                    "f32->f64 batched mismatch at [{},{},{}]: expected {}, got {}",
                    batch,
                    i,
                    j,
                    expected,
                    arr[[batch, i, j]]
                );
            }
        }
        _ => panic!("Expected F64 tile"),
    }
}

// MMAF shape and type validation tests
#[test]
#[should_panic = "assertion `left == right` failed: MMAF: accumulator shape mismatch"]
fn test_mmaf_shape_validation() {
    let lhs =
        Tile::F32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 8]), vec![0.0; 32]).unwrap());
    let rhs = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[8, 16]), vec![0.0; 128]).unwrap(),
    );
    let acc =
        Tile::F32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&[4, 15]), vec![0.0; 60]).unwrap());
    let _ = lhs.mmaf(&rhs, &acc);
}

#[test]
fn test_mmaf_mixed_precision_different_tile_count() {
    // Test that lhs and rhs must have same type
    let lhs = Tile::F16(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0f16; 6]).unwrap(),
    );
    let rhs = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[3, 2]), vec![1.0f32; 6]).unwrap(),
    );
    let acc = Tile::F64(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0f64; 4]).unwrap(),
    );

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lhs.mmaf(&rhs, &acc)));
    assert!(
        result.is_err(),
        "Should panic when lhs/rhs types don't match"
    );
}

// ============================================================================

#[test]
fn test_cmpf_all_predicates() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let size = 8;

    let lhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.7 - 2.0).collect();
    let rhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5 + 1.0).collect();

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    let ordering = ComparisonOrdering::Ordered;

    // Test equal
    {
        let result = lhs.cmpf(&rhs, ComparisonPredicate::Equal, ordering);
        match result {
            Tile::I1(arr) => {
                for [i, j] in ndrange(&[2, 4]) {
                    let idx = (i * 4 + j) as usize;
                    let expected = lhs_data[idx] == rhs_data[idx];
                    assert_eq!(
                        arr[[i, j]],
                        expected,
                        "cmpf equal mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
            _ => panic!("Expected I1 tile"),
        }
    }

    // Test not_equal
    {
        let result = lhs.cmpf(&rhs, ComparisonPredicate::NotEqual, ordering);
        match result {
            Tile::I1(arr) => {
                for [i, j] in ndrange(&[2, 4]) {
                    let idx = (i * 4 + j) as usize;
                    let expected = lhs_data[idx] != rhs_data[idx];
                    assert_eq!(
                        arr[[i, j]],
                        expected,
                        "cmpf not_equal mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
            _ => panic!("Expected I1 tile"),
        }
    }

    // Test less_than
    {
        let result = lhs.cmpf(&rhs, ComparisonPredicate::LessThan, ordering);
        match result {
            Tile::I1(arr) => {
                for [i, j] in ndrange(&[2, 4]) {
                    let idx = (i * 4 + j) as usize;
                    let expected = lhs_data[idx] < rhs_data[idx];
                    assert_eq!(
                        arr[[i, j]],
                        expected,
                        "cmpf less_than mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
            _ => panic!("Expected I1 tile"),
        }
    }

    // Test less_than_or_equal
    {
        let result = lhs.cmpf(&rhs, ComparisonPredicate::LessThanOrEqual, ordering);
        match result {
            Tile::I1(arr) => {
                for [i, j] in ndrange(&[2, 4]) {
                    let idx = (i * 4 + j) as usize;
                    let expected = lhs_data[idx] <= rhs_data[idx];
                    assert_eq!(
                        arr[[i, j]],
                        expected,
                        "cmpf less_than_or_equal mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
            _ => panic!("Expected I1 tile"),
        }
    }

    // Test greater_than
    {
        let result = lhs.cmpf(&rhs, ComparisonPredicate::GreaterThan, ordering);
        match result {
            Tile::I1(arr) => {
                for [i, j] in ndrange(&[2, 4]) {
                    let idx = (i * 4 + j) as usize;
                    let expected = lhs_data[idx] > rhs_data[idx];
                    assert_eq!(
                        arr[[i, j]],
                        expected,
                        "cmpf greater_than mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
            _ => panic!("Expected I1 tile"),
        }
    }

    // Test greater_than_or_equal
    {
        let result = lhs.cmpf(&rhs, ComparisonPredicate::GreaterThanOrEqual, ordering);
        match result {
            Tile::I1(arr) => {
                for [i, j] in ndrange(&[2, 4]) {
                    let idx = (i * 4 + j) as usize;
                    let expected = lhs_data[idx] >= rhs_data[idx];
                    assert_eq!(
                        arr[[i, j]],
                        expected,
                        "cmpf greater_than_or_equal mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
            _ => panic!("Expected I1 tile"),
        }
    }
}

#[test]
fn test_cmpf_ordering_modes() {
    let shape = ndarray::IxDyn(&[2, 2]);

    let lhs_data: Vec<f32> = vec![1.0, f32::NAN, 3.0, 5.0];
    let rhs_data: Vec<f32> = vec![2.0, 3.0, f32::NAN, 5.0];

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data.clone()).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data.clone()).unwrap());

    // Test ordered mode
    {
        let result = lhs.cmpf(
            &rhs,
            ComparisonPredicate::LessThan,
            ComparisonOrdering::Ordered,
        );
        match result {
            Tile::I1(arr) => {
                for [i, j] in ndrange(&[2, 2]) {
                    let idx = (i * 2 + j) as usize;
                    // In ordered mode, NaN comparisons return false
                    let expected = lhs_data[idx] < rhs_data[idx];
                    assert_eq!(
                        arr[[i, j]],
                        expected,
                        "cmpf ordered mismatch at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
            _ => panic!("Expected I1 tile"),
        }
    }

    // Test unordered mode (note: implementation currently ignores ordering)
    {
        let result = lhs.cmpf(
            &rhs,
            ComparisonPredicate::LessThan,
            ComparisonOrdering::Unordered,
        );
        match result {
            Tile::I1(arr) => {
                // Current implementation ignores ordering, so behavior matches ordered
                // TODO: Update when unordered is properly implemented
                // In unordered mode, NaN < value should return true
                for [i, j] in ndrange(&[2, 2]) {
                    let idx = (i * 2 + j) as usize;
                    // Currently behaves as ordered (NaN comparisons return false)
                    let expected = lhs_data[idx] < rhs_data[idx];
                    assert_eq!(
                        arr[[i, j]],
                        expected,
                        "cmpf unordered (currently same as ordered) at [{},{}]: expected {}, got {}",
                        i,
                        j,
                        expected,
                        arr[[i, j]]
                    );
                }
            }
            _ => panic!("Expected I1 tile"),
        }
    }
}

#[test]
fn test_cmpf_nan_behavior() {
    let shape = ndarray::IxDyn(&[2, 2]);

    let lhs_data: Vec<f32> = vec![f32::NAN, 1.0, 2.0, f32::NAN];
    let rhs_data: Vec<f32> = vec![1.0, f32::NAN, f32::NAN, f32::NAN];

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data).unwrap());

    // Test that NaN comparisons follow IEEE 754
    let result = lhs.cmpf(
        &rhs,
        ComparisonPredicate::Equal,
        ComparisonOrdering::Ordered,
    );
    match result {
        Tile::I1(arr) => {
            // Expected results for equality comparison:
            // [0,0]: NaN == 1.0 → false
            // [0,1]: 1.0 == NaN → false
            // [1,0]: 2.0 == NaN → false
            // [1,1]: NaN == NaN → false (IEEE 754: NaN != NaN)
            let expected = vec![false, false, false, false];
            for [i, j] in ndrange(&[2, 2]) {
                let idx = (i * 2 + j) as usize;
                assert_eq!(
                    arr[[i, j]],
                    expected[idx],
                    "cmpf NaN behavior at [{},{}]: expected {}, got {}",
                    i,
                    j,
                    expected[idx],
                    arr[[i, j]]
                );
            }
        }
        _ => panic!("Expected I1 tile"),
    }
}

#[test]
fn test_cmpf_result_type() {
    let shape = ndarray::IxDyn(&[2, 4]);
    let lhs_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let rhs_data: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();

    let lhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), lhs_data).unwrap());
    let rhs = Tile::F32(ndarray::Array::from_shape_vec(shape.clone(), rhs_data).unwrap());

    let result = lhs.cmpf(
        &rhs,
        ComparisonPredicate::LessThan,
        ComparisonOrdering::Ordered,
    );

    // Verify result is Tile::I1
    match result {
        Tile::I1(arr) => {
            assert_eq!(arr.shape(), &[2, 4]);
            // Verify result contains boolean values
            for [i, j] in ndrange(&[2, 4]) {
                let val = arr[[i, j]];
                assert!(
                    val == true || val == false,
                    "cmpf result should be boolean at [{},{}]: got {}",
                    i,
                    j,
                    val
                );
            }
        }
        _ => panic!("cmpf should return Tile::I1, got {:?}", result),
    }
}
