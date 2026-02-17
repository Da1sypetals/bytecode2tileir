// Tests for view operations (Section 8.11 of TileIR)
//
// This module tests the helper methods used by view operations.
// Full integration tests would require bytecode files.

use crate::interpreter::data_structures::elem_type::{ElemType, Scalar};
use crate::interpreter::data_structures::tile::Tile;

#[test]
fn test_tile_from_scalar_i32() {
    let scalar = Scalar::I32(42);
    let tile = Tile::from_scalar(scalar, ElemType::I32);

    assert!(tile.is_scalar());
    assert_eq!(tile.elem_type(), ElemType::I32);
    assert_eq!(tile.get_scalar(&[]), Scalar::I32(42));
}

#[test]
fn test_tile_from_scalar_f32() {
    let scalar = Scalar::F32(3.14159);
    let tile = Tile::from_scalar(scalar, ElemType::F32);

    assert!(tile.is_scalar());
    assert_eq!(tile.elem_type(), ElemType::F32);
    match tile.get_scalar(&[]) {
        Scalar::F32(v) => {
            assert!((v - 3.14159).abs() < 1e-6);
        }
        _ => panic!("Expected F32 scalar"),
    }
}

#[test]
fn test_tile_from_scalar_i64() {
    let scalar = Scalar::I64(9223372036854775807);
    let tile = Tile::from_scalar(scalar, ElemType::I64);

    assert!(tile.is_scalar());
    assert_eq!(tile.elem_type(), ElemType::I64);
    assert_eq!(tile.get_scalar(&[]), Scalar::I64(9223372036854775807));
}

#[test]
fn test_tile_from_scalar_f16() {
    let scalar = Scalar::F16(1.5f16);
    let tile = Tile::from_scalar(scalar, ElemType::F16);

    assert!(tile.is_scalar());
    assert_eq!(tile.elem_type(), ElemType::F16);
    match tile.get_scalar(&[]) {
        Scalar::F16(_) => {
            // Successfully created and retrieved F16 scalar
        }
        _ => panic!("Expected F16 scalar"),
    }
}

#[test]
fn test_tile_from_scalar_bool() {
    let scalar = Scalar::Bool(true);
    let tile = Tile::from_scalar(scalar, ElemType::Bool);

    assert!(tile.is_scalar());
    assert_eq!(tile.elem_type(), ElemType::Bool);
    assert_eq!(tile.get_scalar(&[]), Scalar::Bool(true));
}

#[test]
fn test_tile_from_scalar_ptr() {
    let ptr = 0x12345678 as *mut u8;
    let scalar = Scalar::Ptr(ptr);
    let tile = Tile::from_scalar(scalar, ElemType::Ptr);

    assert!(tile.is_scalar());
    assert_eq!(tile.elem_type(), ElemType::Ptr);
    assert_eq!(tile.get_scalar(&[]), Scalar::Ptr(ptr));
}

#[test]
fn test_scalar_from_i64_width_8() {
    let scalar = Scalar::from_i64(127, 8);
    assert_eq!(scalar, Scalar::I8(127));
}

#[test]
fn test_scalar_from_i64_width_16() {
    let scalar = Scalar::from_i64(32767, 16);
    assert_eq!(scalar, Scalar::I16(32767));
}

#[test]
fn test_scalar_from_i64_width_32() {
    let scalar = Scalar::from_i64(2147483647, 32);
    assert_eq!(scalar, Scalar::I32(2147483647));
}

#[test]
fn test_scalar_from_i64_width_64() {
    let scalar = Scalar::from_i64(9223372036854775807, 64);
    assert_eq!(scalar, Scalar::I64(9223372036854775807));
}

#[test]
fn test_scalar_from_i64_width_1() {
    let scalar_true = Scalar::from_i64(1, 1);
    assert_eq!(scalar_true, Scalar::Bool(true));

    let scalar_false = Scalar::from_i64(0, 1);
    assert_eq!(scalar_false, Scalar::Bool(false));

    // Any non-zero value should become true
    let scalar_nonzero = Scalar::from_i64(42, 1);
    assert_eq!(scalar_nonzero, Scalar::Bool(true));
}

#[test]
fn test_scalar_from_i64_negative() {
    let scalar = Scalar::from_i64(-128, 8);
    assert_eq!(scalar, Scalar::I8(-128));
}

#[test]
fn test_scalar_from_i64_truncation() {
    // Test that values are properly truncated to fit in the target width
    let scalar = Scalar::from_i64(0x1234567812345678, 32);
    assert_eq!(scalar, Scalar::I32(0x12345678));
}
