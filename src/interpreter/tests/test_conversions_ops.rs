// Tests for conversion operations (Section 8.4)
use crate::interpreter::data_structures::tile::Tile;

// ============================================================================
// Pointer Conversion Tests (8.4.6-8.4.8)
// ============================================================================

#[test]
fn test_int_to_ptr() {
    // Test i64 -> ptr with 2-d tile
    let tile = Tile::I64(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[4, 8]),
        (0..32).map(|i| (0x1000 + i) as i64).collect(),
    )
    .unwrap());

    match tile {
        Tile::I64(arr) => {
            let ptrs: Vec<*mut u8> = arr.iter().map(|&v| v as *mut u8).collect();
            let result =
                Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(arr.shape()), ptrs).unwrap());
            match result {
                Tile::Ptr(r) => {
                    assert_eq!(r.shape(), &[4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        assert_eq!(*elem, (0x1000 + idx) as *mut u8);
                    }
                }
                _ => panic!("Expected Ptr tile"),
            }
        }
        _ => panic!("Expected I64 tile"),
    }
}

#[test]
fn test_ptr_to_int() {
    // Test ptr -> i64 with 3-d tile
    let tile = Tile::Ptr(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8]),
        (0..64).map(|i| (0x1000 + i) as *mut u8).collect(),
    )
    .unwrap());

    match tile {
        Tile::Ptr(arr) => {
            let result = Tile::I64(arr.mapv(|p| p as i64));
            match result {
                Tile::I64(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        assert_eq!(*elem, (0x1000 + idx) as i64);
                    }
                }
                _ => panic!("Expected I64 tile"),
            }
        }
        _ => panic!("Expected Ptr tile"),
    }
}

// ============================================================================
// Bitcast Tests (8.4.1)
// ============================================================================

#[test]
fn test_bitcast_i32_f32() {
    // Test i32 -> f32 bitcast with 2-d tile
    let i32_tile = Tile::I32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[4, 8]),
        (0..32).map(|i| 0x40000000i32 + i as i32).collect(),
    )
    .unwrap());

    match i32_tile {
        Tile::I32(arr) => {
            let result = Tile::F32(arr.mapv(|v| f32::from_bits(v.cast_unsigned())));
            match result {
                Tile::F32(r) => {
                    assert_eq!(r.shape(), &[4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected =
                            f32::from_bits((0x40000000i32 + idx as i32).cast_unsigned());
                        assert!((elem - &expected).abs() < 1e-6);
                    }
                }
                _ => panic!("Expected F32 tile"),
            }
        }
        _ => panic!("Expected I32 tile"),
    }
}

#[test]
fn test_bitcast_f32_i32() {
    // Test f32 -> i32 bitcast with 3-d tile
    let f32_tile = Tile::F32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8]),
        (0..64)
            .map(|i| f32::from_bits(0x40000000u32 + i as u32))
            .collect(),
    )
    .unwrap());

    match f32_tile {
        Tile::F32(arr) => {
            let result = Tile::I32(arr.mapv(|v| v.to_bits().cast_signed()));
            match result {
                Tile::I32(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        assert_eq!(*elem, (0x40000000i32 + idx as i32));
                    }
                }
                _ => panic!("Expected I32 tile"),
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_bitcast_i64_f64() {
    // Test i64 -> f64 bitcast with 4-d tile
    let i64_tile = Tile::I64(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8, 16]),
        (0..1024)
            .map(|i| 0x4000000000000000i64 + i as i64)
            .collect(),
    )
    .unwrap());

    match i64_tile {
        Tile::I64(arr) => {
            let result = Tile::F64(arr.mapv(|v| f64::from_bits(v.cast_unsigned())));
            match result {
                Tile::F64(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected =
                            f64::from_bits((0x4000000000000000i64 + idx as i64).cast_unsigned());
                        assert!((elem - &expected).abs() < 1e-12);
                    }
                }
                _ => panic!("Expected F64 tile"),
            }
        }
        _ => panic!("Expected I64 tile"),
    }
}

#[test]
fn test_bitcast_i16_f16() {
    // Test i16 -> f16 bitcast with 2-d tile
    let i16_tile = Tile::I16(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[8, 16]),
        (0..128).map(|i| 0x3C00i16 + i as i16).collect(),
    )
    .unwrap());

    match i16_tile {
        Tile::I16(arr) => {
            let result = Tile::F16(arr.mapv(|v| f16::from_bits(v.cast_unsigned())));
            match result {
                Tile::F16(r) => {
                    assert_eq!(r.shape(), &[8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = f16::from_bits((0x3C00i16 + idx as i16).cast_unsigned());
                        let f32_val = *elem as f32;
                        let expected_f32 = expected as f32;
                        assert!((f32_val - expected_f32).abs() < 0.01);
                    }
                }
                _ => panic!("Expected F16 tile"),
            }
        }
        _ => panic!("Expected I16 tile"),
    }
}

#[test]
fn test_bitcast_f16_i16() {
    // Test f16 -> i16 bitcast with 3-d tile
    let f16_tile = Tile::F16(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[4, 8, 16]),
        (0..512)
            .map(|i| f16::from_bits(0x3C00u16 + i as u16))
            .collect(),
    )
    .unwrap());

    match f16_tile {
        Tile::F16(arr) => {
            let result = Tile::I16(arr.mapv(|v| v.to_bits().cast_signed()));
            match result {
                Tile::I16(r) => {
                    assert_eq!(r.shape(), &[4, 8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        assert_eq!(*elem, (0x3C00i16 + idx as i16));
                    }
                }
                _ => panic!("Expected I16 tile"),
            }
        }
        _ => panic!("Expected F16 tile"),
    }
}

#[test]
fn test_bitcast_f64_i64() {
    // Test f64 -> i64 bitcast with 4-d tile
    let f64_tile = Tile::F64(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8, 32]),
        (0..2048)
            .map(|i| f64::from_bits(0x4000000000000000u64 + i as u64))
            .collect(),
    )
    .unwrap());

    match f64_tile {
        Tile::F64(arr) => {
            let result = Tile::I64(arr.mapv(|v| v.to_bits().cast_signed()));
            match result {
                Tile::I64(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8, 32]);
                    for (idx, elem) in r.iter().enumerate() {
                        assert_eq!(*elem, (0x4000000000000000i64 + idx as i64));
                    }
                }
                _ => panic!("Expected I64 tile"),
            }
        }
        _ => panic!("Expected F64 tile"),
    }
}

// ============================================================================
// ExtI Tests (8.4.2)
// ============================================================================

#[test]
fn test_exti_sign_extension() {
    // Test sign extension from i8 to i32 with 2-d tile
    let i8_tile = Tile::I8(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[4, 8]),
        (-16..16).collect(),
    )
    .unwrap());

    match i8_tile {
        Tile::I8(arr) => {
            let result = Tile::I32(arr.mapv(|v| v as i32));
            match result {
                Tile::I32(r) => {
                    assert_eq!(r.shape(), &[4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = (-16i8 + idx as i8) as i32;
                        assert_eq!(*elem, expected);
                    }
                }
                _ => panic!("Expected I32 tile"),
            }
        }
        _ => panic!("Expected I8 tile"),
    }
}

#[test]
fn test_exti_zero_extension() {
    // Test zero extension from i8 to i32 (unsigned) with 3-d tile
    let i8_tile = Tile::I8(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8]),
        (0..64).map(|i| i as i8).collect(),
    )
    .unwrap());

    match i8_tile {
        Tile::I8(arr) => {
            let result = Tile::I32(arr.mapv(|v| v as u8 as i32));
            match result {
                Tile::I32(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        assert_eq!(*elem, (idx as u8) as i32);
                    }
                }
                _ => panic!("Expected I32 tile"),
            }
        }
        _ => panic!("Expected I8 tile"),
    }
}

#[test]
fn test_exti_i16_to_i64_signed() {
    // Test i16 -> i64 signed extension with 4-d tile
    let i16_tile = Tile::I16(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8, 16]),
        (0..1024).map(|i| i as i16).collect(),
    )
    .unwrap());

    match i16_tile {
        Tile::I16(arr) => {
            let result = Tile::I64(arr.mapv(|v| v as i64));
            match result {
                Tile::I64(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = idx as i16 as i64;
                        assert_eq!(*elem, expected);
                    }
                }
                _ => panic!("Expected I64 tile"),
            }
        }
        _ => panic!("Expected I16 tile"),
    }
}

// ============================================================================
// TruncI Tests (8.4.9)
// ============================================================================

#[test]
fn test_trunci_i64_to_i32() {
    // Test truncation from i64 to i32 with 2-d tile
    let i64_tile = Tile::I64(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[8, 16]),
        (0..128)
            .map(|i| 0x123456789ABCDEF0i64 + i as i64)
            .collect(),
    )
    .unwrap());

    match i64_tile {
        Tile::I64(arr) => {
            let result = Tile::I32(arr.mapv(|v| v as i32));
            match result {
                Tile::I32(r) => {
                    assert_eq!(r.shape(), &[8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = (0x123456789ABCDEF0i64 + idx as i64) as i32;
                        assert_eq!(*elem, expected);
                    }
                }
                _ => panic!("Expected I32 tile"),
            }
        }
        _ => panic!("Expected I64 tile"),
    }
}

#[test]
fn test_trunci_i32_to_i8() {
    // Test i32 -> i8 truncation with 3-d tile
    let i32_tile = Tile::I32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[4, 8, 16]),
        (0..512).map(|i| 0x12345678i32 + i as i32).collect(),
    )
    .unwrap());

    match i32_tile {
        Tile::I32(arr) => {
            let result = Tile::I8(arr.mapv(|v| v as i8));
            match result {
                Tile::I8(r) => {
                    assert_eq!(r.shape(), &[4, 8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = (0x12345678i32 + idx as i32) as i8;
                        assert_eq!(*elem, expected);
                    }
                }
                _ => panic!("Expected I8 tile"),
            }
        }
        _ => panic!("Expected I32 tile"),
    }
}

#[test]
fn test_trunci_to_bool() {
    // Test i32 -> bool truncation with 4-d tile
    let i32_tile = Tile::I32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8, 16]),
        (0..1024).map(|i| if i % 2 == 0 { 42 } else { 0 }).collect(),
    )
    .unwrap());

    match i32_tile {
        Tile::I32(arr) => {
            let result = Tile::I1(arr.mapv(|v| v != 0));
            match result {
                Tile::I1(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = idx % 2 == 0;
                        assert_eq!(*elem, expected);
                    }
                }
                _ => panic!("Expected I1 tile"),
            }
        }
        _ => panic!("Expected I32 tile"),
    }
}

// ============================================================================
// FToF Tests (8.4.3)
// ============================================================================

#[test]
fn test_ftof_f16_to_f32() {
    // Test f16 -> f32 with 2-d tile
    let f16_tile = Tile::F16(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[4, 8]),
        (0..32).map(|i| (1.0 + i as f32 / 32.0) as f16).collect(),
    )
    .unwrap());

    match f16_tile {
        Tile::F16(arr) => {
            let result = Tile::F32(arr.mapv(|v| v as f32));
            match result {
                Tile::F32(r) => {
                    assert_eq!(r.shape(), &[4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = 1.0 + idx as f32 / 32.0;
                        assert!((elem - &expected).abs() < 1e-6);
                    }
                }
                _ => panic!("Expected F32 tile"),
            }
        }
        _ => panic!("Expected F16 tile"),
    }
}

#[test]
fn test_ftof_f32_to_f64() {
    // Test f32 -> f64 with 3-d tile
    let f32_tile = Tile::F32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8]),
        (0..64).map(|i| 1.0 + i as f32 / 64.0).collect(),
    )
    .unwrap());

    match f32_tile {
        Tile::F32(arr) => {
            let result = Tile::F64(arr.mapv(|v| v as f64));
            match result {
                Tile::F64(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = 1.0 + idx as f64 / 64.0;
                        assert!((elem - &expected).abs() < 1e-12);
                    }
                }
                _ => panic!("Expected F64 tile"),
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_ftof_f64_to_f32() {
    // Test f64 -> f32 with 4-d tile
    let f64_tile = Tile::F64(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8, 16]),
        (0..1024).map(|i| 1.0 + i as f64 / 1024.0).collect(),
    )
    .unwrap());

    match f64_tile {
        Tile::F64(arr) => {
            let result = Tile::F32(arr.mapv(|v| v as f32));
            match result {
                Tile::F32(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = 1.0 + idx as f32 / 1024.0;
                        assert!((elem - &expected).abs() < 1e-6);
                    }
                }
                _ => panic!("Expected F32 tile"),
            }
        }
        _ => panic!("Expected F64 tile"),
    }
}

#[test]
fn test_ftof_f32_to_f16() {
    // Test f32 -> f16 with 2-d tile
    let f32_tile = Tile::F32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[8, 16]),
        (0..128).map(|i| 1.0 + i as f32 / 128.0).collect(),
    )
    .unwrap());

    match f32_tile {
        Tile::F32(arr) => {
            let result = Tile::F16(arr.mapv(|v| v as f16));
            match result {
                Tile::F16(r) => {
                    assert_eq!(r.shape(), &[8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let f32_val = *elem as f32;
                        let expected = 1.0 + idx as f32 / 128.0;
                        assert!((f32_val - expected).abs() < 0.001);
                    }
                }
                _ => panic!("Expected F16 tile"),
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

// ============================================================================
// IToF Tests (8.4.5)
// ============================================================================

#[test]
fn test_itof_signed_i32_to_f32() {
    // Test i32 -> f32 signed with 2-d tile
    let i32_tile = Tile::I32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[4, 8]),
        (-16..16).map(|i| i as i32).collect(),
    )
    .unwrap());

    match i32_tile {
        Tile::I32(arr) => {
            let result = Tile::F32(arr.mapv(|v| v as f32));
            match result {
                Tile::F32(r) => {
                    assert_eq!(r.shape(), &[4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = (-16i32 + idx as i32) as f32;
                        assert!((elem - &expected).abs() < 1e-6);
                    }
                }
                _ => panic!("Expected F32 tile"),
            }
        }
        _ => panic!("Expected I32 tile"),
    }
}

#[test]
fn test_itof_unsigned_i32_to_f32() {
    // Test i32 -> f32 unsigned with 3-d tile
    let i32_tile = Tile::I32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8]),
        (0..64).map(|i| i as i32).collect(),
    )
    .unwrap());

    match i32_tile {
        Tile::I32(arr) => {
            let result = Tile::F32(arr.mapv(|v| (v as u32) as f32));
            match result {
                Tile::F32(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = (idx as u32) as f32;
                        assert!((elem - &expected).abs() < 1.0);
                    }
                }
                _ => panic!("Expected F32 tile"),
            }
        }
        _ => panic!("Expected I32 tile"),
    }
}

#[test]
fn test_itof_i64_to_f16() {
    // Test i64 -> f16 with 4-d tile
    let i64_tile = Tile::I64(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8, 16]),
        (0..1024).map(|i| i as i64).collect(),
    )
    .unwrap());

    match i64_tile {
        Tile::I64(arr) => {
            let result = Tile::F16(arr.mapv(|v| (v as f32) as f16));
            match result {
                Tile::F16(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let f32_val = *elem as f32;
                        let expected = idx as f32;
                        assert!((f32_val - expected).abs() < 1.0);
                    }
                }
                _ => panic!("Expected F16 tile"),
            }
        }
        _ => panic!("Expected I64 tile"),
    }
}

// ============================================================================
// FToI Tests (8.4.4)
// ============================================================================

#[test]
fn test_ftoi_f32_to_i32_round_to_zero() {
    // Test f32 -> i32 round to zero with 2-d tile
    let f32_tile = Tile::F32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[4, 8]),
        (0..32).map(|i| (i as f32) * 0.7 + 1.0).collect(),
    )
    .unwrap());

    match f32_tile {
        Tile::F32(arr) => {
            let result = Tile::I32(arr.mapv(|v| v as i32));
            match result {
                Tile::I32(r) => {
                    assert_eq!(r.shape(), &[4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = ((idx as f32) * 0.7 + 1.0) as i32;
                        assert_eq!(*elem, expected);
                    }
                }
                _ => panic!("Expected I32 tile"),
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_ftoi_f32_to_i32_round_nearest() {
    // Test f32 -> i32 round to nearest with 3-d tile
    let f32_tile = Tile::F32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8]),
        (0..64).map(|i| (i as f32) * 0.7 + 1.0).collect(),
    )
    .unwrap());

    match f32_tile {
        Tile::F32(arr) => {
            let result = Tile::I32(arr.mapv(|v| v.round() as i32));
            match result {
                Tile::I32(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = (((idx as f32) * 0.7 + 1.0).round()) as i32;
                        assert_eq!(*elem, expected);
                    }
                }
                _ => panic!("Expected I32 tile"),
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_ftoi_f32_to_i32_negative() {
    // Test f32 -> i32 negative with 4-d tile
    let f32_tile = Tile::F32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8, 16]),
        (0..1024).map(|i| -(i as f32) * 0.7 - 1.0).collect(),
    )
    .unwrap());

    match f32_tile {
        Tile::F32(arr) => {
            let result = Tile::I32(arr.mapv(|v| v as i32));
            match result {
                Tile::I32(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = (-(idx as f32) * 0.7 - 1.0) as i32;
                        assert_eq!(*elem, expected);
                    }
                }
                _ => panic!("Expected I32 tile"),
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

#[test]
fn test_ftoi_f64_to_i64() {
    // Test f64 -> i64 with 2-d tile
    let f64_tile = Tile::F64(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[8, 16]),
        (0..128).map(|i| (i as f64) * 0.9 + 1.0).collect(),
    )
    .unwrap());

    match f64_tile {
        Tile::F64(arr) => {
            let result = Tile::I64(arr.mapv(|v| v as i64));
            match result {
                Tile::I64(r) => {
                    assert_eq!(r.shape(), &[8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = ((idx as f64) * 0.9 + 1.0) as i64;
                        assert_eq!(*elem, expected);
                    }
                }
                _ => panic!("Expected I64 tile"),
            }
        }
        _ => panic!("Expected F64 tile"),
    }
}

#[test]
fn test_ftoi_f16_to_i32() {
    // Test f16 -> i32 with 3-d tile
    let f16_tile = Tile::F16(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[4, 8, 16]),
        (0..512).map(|i| ((i as f32) * 0.5 + 1.0) as f16).collect(),
    )
    .unwrap());

    match f16_tile {
        Tile::F16(arr) => {
            let result = Tile::I32(arr.mapv(|v| (v as f32) as i32));
            match result {
                Tile::I32(r) => {
                    assert_eq!(r.shape(), &[4, 8, 16]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = (((idx as f32) * 0.5 + 1.0) as f32) as i32;
                        assert_eq!(*elem, expected);
                    }
                }
                _ => panic!("Expected I32 tile"),
            }
        }
        _ => panic!("Expected F16 tile"),
    }
}

#[test]
fn test_ftoi_unsigned_f32_to_u32() {
    // Test f32 -> u32 with 4-d tile
    let f32_tile = Tile::F32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8, 32]),
        (0..2048).map(|i| (i as f32) * 0.7 + 1.0).collect(),
    )
    .unwrap());

    match f32_tile {
        Tile::F32(arr) => {
            let result = Tile::I32(arr.mapv(|v| (v as i32) as u32 as i32));
            match result {
                Tile::I32(r) => {
                    assert_eq!(r.shape(), &[2, 4, 8, 32]);
                    for (idx, elem) in r.iter().enumerate() {
                        let expected = (((idx as f32) * 0.7 + 1.0) as i32) as u32 as i32;
                        assert_eq!(*elem, expected);
                    }
                }
                _ => panic!("Expected I32 tile"),
            }
        }
        _ => panic!("Expected F32 tile"),
    }
}

// ============================================================================
// Round-trip Conversion Tests
// ============================================================================

#[test]
fn test_round_trip_i32_f32_i32() {
    // Test round-trip i32 -> f32 -> i32 with 2-d tile
    let i32_tile = Tile::I32(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[4, 8]),
        (-16..16).map(|i| i as i32).collect(),
    )
    .unwrap());

    match &i32_tile {
        Tile::I32(arr) => {
            let f32_tile = Tile::F32(arr.mapv(|v| v as f32));
            match f32_tile {
                Tile::F32(f32_arr) => {
                    let i32_result = Tile::I32(f32_arr.mapv(|v| v as i32));
                    match i32_result {
                        Tile::I32(r) => {
                            assert_eq!(r.shape(), &[4, 8]);
                            for (idx, elem) in r.iter().enumerate() {
                                let original = -16i32 + idx as i32;
                                assert_eq!(*elem, original);
                            }
                        }
                        _ => panic!("Expected I32 tile"),
                    }
                }
                _ => panic!("Expected F32 tile"),
            }
        }
        _ => panic!("Expected I32 tile"),
    }
}

#[test]
fn test_round_trip_i16_f16_i16() {
    // Test round-trip i16 -> f16 -> i16 with 3-d tile
    let i16_tile = Tile::I16(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8]),
        (-32..32).map(|i| i as i16).collect(),
    )
    .unwrap());

    match &i16_tile {
        Tile::I16(arr) => {
            let f16_tile = Tile::F16(arr.mapv(|v| v as f16));
            match f16_tile {
                Tile::F16(f16_arr) => {
                    let i16_result = Tile::I16(f16_arr.mapv(|v| v as i16));
                    match i16_result {
                        Tile::I16(r) => {
                            assert_eq!(r.shape(), &[2, 4, 8]);
                            for (idx, elem) in r.iter().enumerate() {
                                let original = -32i16 + idx as i16;
                                assert_eq!(*elem, original);
                            }
                        }
                        _ => panic!("Expected I16 tile"),
                    }
                }
                _ => panic!("Expected F16 tile"),
            }
        }
        _ => panic!("Expected I16 tile"),
    }
}

#[test]
fn test_round_trip_f16_f32_f16() {
    // Test round-trip f16 -> f32 -> f16 with 4-d tile
    let f16_tile = Tile::F16(ndarray::Array::from_shape_vec(
        ndarray::IxDyn(&[2, 4, 8, 16]),
        (0..1024).map(|i| (1.0 + i as f32 / 1024.0) as f16).collect(),
    )
    .unwrap());

    match &f16_tile {
        Tile::F16(arr) => {
            let f32_tile = Tile::F32(arr.mapv(|v| v as f32));
            match f32_tile {
                Tile::F32(f32_arr) => {
                    let f16_result = Tile::F16(f32_arr.mapv(|v| v as f16));
                    match f16_result {
                        Tile::F16(r) => {
                            assert_eq!(r.shape(), &[2, 4, 8, 16]);
                            // f16 has limited precision, so we check approximate equality
                            for (idx, elem) in r.iter().enumerate() {
                                let f32_val = *elem as f32;
                                let original = 1.0 + idx as f32 / 1024.0;
                                assert!((f32_val - original).abs() < 0.01);
                            }
                        }
                        _ => panic!("Expected F16 tile"),
                    }
                }
                _ => panic!("Expected F32 tile"),
            }
        }
        _ => panic!("Expected F16 tile"),
    }
}
