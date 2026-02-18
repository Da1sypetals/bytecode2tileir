use crate::interpreter::data_structures::elem_type::{ElemType, Scalar};
use crate::interpreter::data_structures::tile::Tile;
use log::debug;
use ndrange::ndrange;
use rand::{Rng, SeedableRng};
use rand::{RngExt, rngs::StdRng};

// ============================================================================
// Load Operations (9 tests)
// ============================================================================

#[test]
fn test_load_i8_unmasked_2d() {
    let shape = [8usize, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size];
    let ptr = buffer.as_mut_ptr();

    for i in 0..size {
        let value = if i < 64 { i as i8 } else { -(i as i8) };
        unsafe {
            *((ptr as *mut i8).offset(i as isize)) = value;
        }
    }

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset(i as isize) })
        .collect();
    let ptr_tile = Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let result = ptr_tile.load_from_ptrs(None, None, &shape, ElemType::I8);

    for [i, j] in ndrange::<2, i64>(&shape) {
        let idx = (i * 16i64 + j) as usize;
        let expected = if idx < 64 { idx as i8 } else { -(idx as i8) };
        match result.get_scalar(&[i, j]) {
            Scalar::I8(v) => assert_eq!(
                v, expected,
                "Mismatch at [{}, {}]: expected {}, got {}",
                i, j, expected, v
            ),
            _ => panic!("Expected I8 at [{}, {}]", i, j),
        }
    }
}

#[test]
fn test_load_i16_unmasked_3d() {
    let shape = [4usize, 8, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 2];
    let ptr = buffer.as_mut_ptr();

    for i in 0..size {
        let value = if i % 2 == 0 { i16::MIN } else { i16::MAX };
        unsafe {
            *((ptr as *mut i16).offset(i as isize)) = value;
        }
    }

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 2) as isize) })
        .collect();
    let ptr_tile = Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let result = ptr_tile.load_from_ptrs(None, None, &shape, ElemType::I16);

    for [i, j, k] in ndrange::<3, i64>(&shape) {
        let idx = (i * 8i64 * 16i64 + j * 16i64 + k) as usize;
        let expected = if idx % 2 == 0 { i16::MIN } else { i16::MAX };
        match result.get_scalar(&[i, j, k]) {
            Scalar::I16(v) => assert_eq!(
                v, expected,
                "Mismatch at [{}, {}, {}]: expected {}, got {}",
                i, j, k, expected, v
            ),
            _ => panic!("Expected I16 at [{}, {}, {}]", i, j, k),
        }
    }
}

#[test]
fn test_load_i32_masked_4d() {
    let shape = [2usize, 4, 8, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 4];
    let ptr = buffer.as_mut_ptr();

    let mut rng = StdRng::seed_from_u64(42);
    let random_data: Vec<i32> = (0..size)
        .map(|_| rng.random_range(i32::MIN..=i32::MAX))
        .collect();

    for i in 0..size {
        let value = random_data[i];
        unsafe {
            *((ptr as *mut i32).offset(i as isize)) = value;
        }
    }

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 4) as isize) })
        .collect();
    let ptr_tile = Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let mask_data: Vec<bool> = (0..size).map(|i| i % 2 == 0).collect();
    let mask_tile =
        Tile::I1(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), mask_data).unwrap());

    let padding_data: Vec<i32> = vec![0xDEADBEEFu32 as i32; size];
    let padding_tile =
        Tile::I32(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), padding_data).unwrap());

    let result =
        ptr_tile.load_from_ptrs(Some(&mask_tile), Some(&padding_tile), &shape, ElemType::I32);

    for [i, j, k, l] in ndrange::<4, i64>(&shape) {
        let idx = (i * 4i64 * 8i64 * 16i64 + j * 8i64 * 16i64 + k * 16i64 + l) as usize;
        let is_masked = idx % 2 == 0;
        let expected = if is_masked {
            random_data[idx]
        } else {
            0xDEADBEEFu32 as i32
        };
        match result.get_scalar(&[i, j, k, l]) {
            Scalar::I32(v) => {
                assert_eq!(
                    v, expected,
                    "Mismatch at [{}, {}, {}, {}]: expected {}, got {}",
                    i, j, k, l, expected, v
                );

                debug!("Masked: {}, value: expected {} vs actual {}", is_masked, expected, v);
            }
            _ => panic!("Expected I32 at [{}, {}, {}, {}]", i, j, k, l),
        }
    }
}

#[test]
fn test_load_i64_unmasked_3d() {
    let shape = [8usize, 16, 4];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 8];
    let ptr = buffer.as_mut_ptr();

    for i in 0..size {
        let value: i64 = 0x1000000000000000i64 + i as i64;
        unsafe {
            *((ptr as *mut i64).offset(i as isize)) = value;
        }
    }

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 8) as isize) })
        .collect();
    let ptr_tile = Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let result = ptr_tile.load_from_ptrs(None, None, &shape, ElemType::I64);

    for [i, j, k] in ndrange::<3, i64>(&shape) {
        let idx = (i * 16i64 * 4i64 + j * 4i64 + k) as usize;
        let expected = 0x1000000000000000i64 + idx as i64;
        match result.get_scalar(&[i, j, k]) {
            Scalar::I64(v) => assert_eq!(
                v, expected,
                "Mismatch at [{}, {}, {}]: expected {}, got {}",
                i, j, k, expected, v
            ),
            _ => panic!("Expected I64 at [{}, {}, {}]", i, j, k),
        }
    }
}

#[test]
fn test_load_f16_masked_2d() {
    let shape = [16usize, 32];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 2];
    let ptr = buffer.as_mut_ptr();

    for i in 0..size {
        let value: f16 = 1.0 + (i as f16) * 0.5;
        unsafe {
            *((ptr as *mut u16).offset(i as isize)) = value.to_bits();
        }
    }

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 2) as isize) })
        .collect();
    let ptr_tile = Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let mask_data: Vec<bool> = (0..size)
        .map(|i| {
            let row = i / 32;
            row % 2 == 0
        })
        .collect();
    let mask_tile =
        Tile::I1(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), mask_data).unwrap());

    let padding_value: f16 = 3.14;
    let padding_data: Vec<f16> = vec![padding_value; size];
    let padding_tile =
        Tile::F16(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), padding_data).unwrap());

    let result =
        ptr_tile.load_from_ptrs(Some(&mask_tile), Some(&padding_tile), &shape, ElemType::F16);

    for [i, j] in ndrange::<2, i64>(&shape) {
        let idx = (i * 32i64 + j) as usize;
        let is_masked = (i as usize) % 2 == 0;
        let expected = if is_masked {
            1.0f16 + (idx as f16) * 0.5
        } else {
            3.14f16
        };
        match result.get_scalar(&[i, j]) {
            Scalar::F16(v) => {
                let v_f32: f32 = v as f32;
                let expected_f32: f32 = expected as f32;
                let diff = (v_f32 - expected_f32).abs();
                assert!(
                    diff < 0.01,
                    "Mismatch at [{}, {}]: expected {:.4}, got {:.4} (diff={:.4})",
                    i,
                    j,
                    expected_f32,
                    v_f32,
                    diff
                );
            }
            _ => panic!("Expected F16 at [{}, {}]", i, j),
        }
    }
}

#[test]
fn test_load_f32_unmasked_4d() {
    let shape = [2usize, 4, 8, 8];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 4];
    let ptr = buffer.as_mut_ptr();

    for i in 0..size {
        let value: f32 = (i as f32) * 0.7;
        unsafe {
            *((ptr as *mut f32).offset(i as isize)) = value;
        }
    }

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 4) as isize) })
        .collect();
    let ptr_tile = Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let result = ptr_tile.load_from_ptrs(None, None, &shape, ElemType::F32);

    for [i, j, k, l] in ndrange::<4, i64>(&shape) {
        let idx = (i * 4i64 * 8i64 * 8i64 + j * 8i64 * 8i64 + k * 8i64 + l) as usize;
        let expected = (idx as f32) * 0.7;
        match result.get_scalar(&[i, j, k, l]) {
            Scalar::F32(v) => {
                let diff = (v - expected).abs();
                assert!(
                    diff < 1e-6,
                    "Mismatch at [{}, {}, {}, {}]: expected {}, got {} (diff={})",
                    i,
                    j,
                    k,
                    l,
                    expected,
                    v,
                    diff
                );
            }
            _ => panic!("Expected F32 at [{}, {}, {}, {}]", i, j, k, l),
        }
    }
}

#[test]
fn test_load_f64_unmasked_3d() {
    let shape = [8usize, 4, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 8];
    let ptr = buffer.as_mut_ptr();

    for i in 0..size {
        let value: f64 = 1.0 / ((i + 1) as f64);
        unsafe {
            *((ptr as *mut f64).offset(i as isize)) = value;
        }
    }

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 8) as isize) })
        .collect();
    let ptr_tile = Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let result = ptr_tile.load_from_ptrs(None, None, &shape, ElemType::F64);

    for [i, j, k] in ndrange::<3, i64>(&shape) {
        let idx = (i * 4i64 * 16i64 + j * 16i64 + k) as usize;
        let expected = 1.0 / ((idx + 1) as f64);
        match result.get_scalar(&[i, j, k]) {
            Scalar::F64(v) => {
                let diff = (v - expected).abs();
                assert!(
                    diff < 1e-12,
                    "Mismatch at [{}, {}, {}]: expected {:.15}, got {:.15} (diff={:.15})",
                    i,
                    j,
                    k,
                    expected,
                    v,
                    diff
                );
            }
            _ => panic!("Expected F64 at [{}, {}, {}]", i, j, k),
        }
    }
}

#[test]
fn test_load_i1_unmasked_2d() {
    let shape = [8usize, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size];
    let ptr = buffer.as_mut_ptr();

    for i in 0..size {
        let value: u8 = if i % 2 == 0 { 1 } else { 0 };
        unsafe {
            *((ptr as *mut u8).offset(i as isize)) = value;
        }
    }

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset(i as isize) })
        .collect();
    let ptr_tile = Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let result = ptr_tile.load_from_ptrs(None, None, &shape, ElemType::Bool);

    for [i, j] in ndrange::<2, i64>(&shape) {
        let idx = (i * 16i64 + j) as usize;
        let expected = idx % 2 == 0;
        match result.get_scalar(&[i, j]) {
            Scalar::Bool(v) => assert_eq!(
                v, expected,
                "Mismatch at [{}, {}]: expected {}, got {}",
                i, j, expected, v
            ),
            _ => panic!("Expected I1 at [{}, {}]", i, j),
        }
    }
}

#[test]
fn test_load_i32_full_mask_3d() {
    let shape = [4usize, 8, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 4];
    let ptr = buffer.as_mut_ptr();

    let mut rng = StdRng::seed_from_u64(123);
    let random_data: Vec<i32> = (0..size)
        .map(|_| rng.random_range(i32::MIN..=i32::MAX))
        .collect();

    for i in 0..size {
        let value = random_data[i];
        unsafe {
            *((ptr as *mut i32).offset(i as isize)) = value;
        }
    }

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 4) as isize) })
        .collect();
    let ptr_tile = Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let mask_data: Vec<bool> = vec![true; size];
    let mask_tile =
        Tile::I1(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), mask_data).unwrap());

    let result = ptr_tile.load_from_ptrs(Some(&mask_tile), None, &shape, ElemType::I32);

    for [i, j, k] in ndrange::<3, i64>(&shape) {
        let idx = (i * 8i64 * 16i64 + j * 16i64 + k) as usize;
        let expected = random_data[idx];
        match result.get_scalar(&[i, j, k]) {
            Scalar::I32(v) => assert_eq!(
                v, expected,
                "Mismatch at [{}, {}, {}]: expected {}, got {}",
                i, j, k, expected, v
            ),
            _ => panic!("Expected I32 at [{}, {}, {}]", i, j, k),
        }
    }
}

// ============================================================================
// Store Operations (9 tests)
// ============================================================================

#[test]
fn test_store_i8_unmasked_2d() {
    let shape = [8usize, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size];
    let ptr = buffer.as_mut_ptr();

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset(i as isize) })
        .collect();
    let dest_tile =
        Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let value_data: Vec<i8> = (0..size)
        .map(|i| {
            if i < 64 {
                (i % 128) as i8
            } else {
                ((255 - i) % 128) as i8 - 64
            }
        })
        .collect();
    let value_tile = Tile::I8(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), value_data.clone()).unwrap(),
    );

    value_tile.store_to_ptrs(&dest_tile, None);

    for [i, j] in ndrange::<2, i64>(&shape) {
        let i_usize = i as usize;
        let j_usize = j as usize;
        let idx = i_usize * 16 + j_usize;
        let expected = value_data[idx];
        let actual = unsafe { *((ptr as *mut i8).offset(idx as isize)) };
        assert_eq!(
            actual, expected,
            "Mismatch at [{}, {}] (idx={}): expected {}, got {}",
            i, j, idx, expected, actual
        );
    }
}

#[test]
fn test_store_i16_unmasked_3d() {
    let shape = [4usize, 8, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 2];
    let ptr = buffer.as_mut_ptr();

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 2) as isize) })
        .collect();
    let dest_tile =
        Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let mut rng = StdRng::seed_from_u64(456);
    let value_data: Vec<i16> = (0..size)
        .map(|_| rng.random_range(i16::MIN..=i16::MAX))
        .collect();
    let value_tile = Tile::I16(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), value_data.clone()).unwrap(),
    );

    value_tile.store_to_ptrs(&dest_tile, None);

    for [i, j, k] in ndrange::<3, i64>(&shape) {
        let i_usize = i as usize;
        let j_usize = j as usize;
        let k_usize = k as usize;
        let idx = i_usize * 8 * 16 + j_usize * 16 + k_usize;
        let expected = value_data[idx];
        let actual = unsafe { *((ptr as *mut i16).offset(idx as isize)) };
        assert_eq!(
            actual, expected,
            "Mismatch at [{}, {}, {}] (idx={}): expected {}, got {}",
            i, j, k, idx, expected, actual
        );
    }
}

#[test]
fn test_store_i32_masked_4d() {
    let shape = [2usize, 4, 8, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 4];
    let ptr = buffer.as_mut_ptr();

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 4) as isize) })
        .collect();
    let dest_tile =
        Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let mut rng = StdRng::seed_from_u64(789);
    let value_data: Vec<i32> = (0..size)
        .map(|_| rng.random_range(i32::MIN..=i32::MAX))
        .collect();
    let value_tile = Tile::I32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), value_data.clone()).unwrap(),
    );

    let mask_data: Vec<bool> = (0..size).map(|i| i % 4 == 0).collect();
    let mask_tile = Tile::I1(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), mask_data.clone()).unwrap(),
    );

    for i in 0..size {
        unsafe {
            *((ptr as *mut i32).offset(i as isize)) = 0xFFFFFFFFu32 as i32;
        }
    }

    value_tile.store_to_ptrs(&dest_tile, Some(&mask_tile));

    for [i, j, k, l] in ndrange::<4, i64>(&shape) {
        let i_usize = i as usize;
        let j_usize = j as usize;
        let k_usize = k as usize;
        let l_usize = l as usize;
        let idx = i_usize * 4 * 8 * 16 + j_usize * 8 * 16 + k_usize * 16 + l_usize;
        let is_masked = mask_data[idx];
        let expected = if is_masked {
            value_data[idx]
        } else {
            0xFFFFFFFFu32 as i32
        };
        let actual = unsafe { *((ptr as *mut i32).offset(idx as isize)) };
        debug!(
            "idx={}, expected={}, actual={}, masked={}",
            idx, expected, actual, is_masked
        );
        assert_eq!(
            actual, expected,
            "Mismatch at [{}, {}, {}, {}] (idx={}): expected {}, got {}",
            i, j, k, l, idx, expected, actual
        );
    }
}

#[test]
fn test_store_i64_unmasked_3d() {
    let shape = [8usize, 16, 4];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 8];
    let ptr = buffer.as_mut_ptr();

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 8) as isize) })
        .collect();
    let dest_tile =
        Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let value_data: Vec<i64> = (0..size)
        .map(|i| 0x1000000000000000i64 + (i as i64) * 2)
        .collect();
    let value_tile = Tile::I64(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), value_data.clone()).unwrap(),
    );

    value_tile.store_to_ptrs(&dest_tile, None);

    for [i, j, k] in ndrange::<3, i64>(&shape) {
        let i_usize = i as usize;
        let j_usize = j as usize;
        let k_usize = k as usize;
        let idx = i_usize * 16 * 4 + j_usize * 4 + k_usize;
        let expected = value_data[idx];
        let actual = unsafe { *((ptr as *mut i64).offset(idx as isize)) };
        assert_eq!(
            actual, expected,
            "Mismatch at [{}, {}, {}] (idx={}): expected {}, got {}",
            i, j, k, idx, expected, actual
        );
    }
}

#[test]
fn test_store_f16_masked_2d() {
    let shape = [16usize, 32];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 2];
    let ptr = buffer.as_mut_ptr();

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 2) as isize) })
        .collect();
    let dest_tile =
        Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let test_values: Vec<f16> = vec![0.0f16, 1.0f16, -1.0f16, f16::MAX, f16::MIN, 3.14f16];
    let value_data: Vec<f16> = (0..size)
        .map(|i| test_values[i % test_values.len()])
        .collect();
    let value_tile = Tile::F16(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), value_data.clone()).unwrap(),
    );

    let mask_data: Vec<bool> = (0..size)
        .map(|i| {
            let row = i / 32;
            let col = i % 32;
            (row + col) % 2 == 0
        })
        .collect();
    let mask_tile = Tile::I1(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), mask_data.clone()).unwrap(),
    );

    for i in 0..size {
        unsafe {
            *((ptr as *mut u16).offset(i as isize)) = 0xFFFF;
        }
    }

    value_tile.store_to_ptrs(&dest_tile, Some(&mask_tile));

    for [i, j] in ndrange::<2, i64>(&shape) {
        let i_usize = i as usize;
        let j_usize = j as usize;
        let idx = i_usize * 32 + j_usize;
        let is_masked = mask_data[idx];
        let expected_bits = if is_masked {
            value_data[idx].to_bits()
        } else {
            0xFFFFu16
        };
        let actual = unsafe { *((ptr as *mut u16).offset(idx as isize)) };
        assert_eq!(
            actual, expected_bits,
            "Mismatch at [{}, {}] (idx={}): expected bits 0x{:04X}, got 0x{:04X}",
            i, j, idx, expected_bits, actual
        );
    }
}

#[test]
fn test_store_f32_unmasked_4d() {
    let shape = [2usize, 4, 8, 8];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 4];
    let ptr = buffer.as_mut_ptr();

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 4) as isize) })
        .collect();
    let dest_tile =
        Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let value_data: Vec<f32> = (0..size)
        .map(|i| std::f32::consts::PI * (i as f32))
        .collect();
    let value_tile = Tile::F32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), value_data.clone()).unwrap(),
    );

    value_tile.store_to_ptrs(&dest_tile, None);

    for [i, j, k, l] in ndrange::<4, i64>(&shape) {
        let i_usize = i as usize;
        let j_usize = j as usize;
        let k_usize = k as usize;
        let l_usize = l as usize;
        let idx = i_usize * 4 * 8 * 8 + j_usize * 8 * 8 + k_usize * 8 + l_usize;
        let expected = value_data[idx];
        let actual = unsafe { *((ptr as *mut f32).offset(idx as isize)) };
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-6,
            "Mismatch at [{}, {}, {}, {}] (idx={}): expected {}, got {} (diff={})",
            i,
            j,
            k,
            l,
            idx,
            expected,
            actual,
            diff
        );
    }
}

#[test]
fn test_store_f64_unmasked_3d() {
    let shape = [8usize, 4, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 8];
    let ptr = buffer.as_mut_ptr();

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 8) as isize) })
        .collect();
    let dest_tile =
        Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let value_data: Vec<f64> = (0..size)
        .map(|i| {
            let x = (i + 1) as f64;
            1.0 / (x * x)
        })
        .collect();
    let value_tile = Tile::F64(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), value_data.clone()).unwrap(),
    );

    value_tile.store_to_ptrs(&dest_tile, None);

    for [i, j, k] in ndrange::<3, i64>(&shape) {
        let i_usize = i as usize;
        let j_usize = j as usize;
        let k_usize = k as usize;
        let idx = i_usize * 4 * 16 + j_usize * 16 + k_usize;
        let expected = value_data[idx];
        let actual = unsafe { *((ptr as *mut f64).offset(idx as isize)) };
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-12,
            "Mismatch at [{}, {}, {}] (idx={}): expected {:.15}, got {:.15} (diff={:.15})",
            i,
            j,
            k,
            idx,
            expected,
            actual,
            diff
        );
    }
}

#[test]
fn test_store_i1_unmasked_2d() {
    let shape = [8usize, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size];
    let ptr = buffer.as_mut_ptr();

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset(i as isize) })
        .collect();
    let dest_tile =
        Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let value_data: Vec<bool> = (0..size).map(|i| i % 3 == 0).collect();
    let value_tile = Tile::I1(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), value_data.clone()).unwrap(),
    );

    value_tile.store_to_ptrs(&dest_tile, None);

    for [i, j] in ndrange::<2, i64>(&shape) {
        let i_usize = i as usize;
        let j_usize = j as usize;
        let idx = i_usize * 16 + j_usize;
        let expected = value_data[idx];
        let actual = unsafe { *((ptr as *mut bool).offset(idx as isize)) };
        assert_eq!(
            actual, expected,
            "Mismatch at [{}, {}] (idx={}): expected {}, got {}",
            i, j, idx, expected, actual
        );
    }
}

#[test]
fn test_store_i32_full_mask_3d() {
    let shape = [4usize, 8, 16];
    let size: usize = shape.iter().product();
    let mut buffer = vec![0u8; size * 4];
    let ptr = buffer.as_mut_ptr();

    let ptrs: Vec<*mut u8> = (0..size)
        .map(|i| unsafe { (ptr as *mut u8).offset((i * 4) as isize) })
        .collect();
    let dest_tile =
        Tile::Ptr(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), ptrs).unwrap());

    let mut rng = StdRng::seed_from_u64(987);
    let value_data: Vec<i32> = (0..size)
        .map(|_| rng.random_range(i32::MIN..=i32::MAX))
        .collect();
    let value_tile = Tile::I32(
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), value_data.clone()).unwrap(),
    );

    let mask_data: Vec<bool> = vec![true; size];
    let mask_tile =
        Tile::I1(ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), mask_data).unwrap());

    value_tile.store_to_ptrs(&dest_tile, Some(&mask_tile));

    for [i, j, k] in ndrange::<3, i64>(&shape) {
        let i_usize = i as usize;
        let j_usize = j as usize;
        let k_usize = k as usize;
        let idx = i_usize * 8 * 16 + j_usize * 16 + k_usize;
        let expected = value_data[idx];
        let actual = unsafe { *((ptr as *mut i32).offset(idx as isize)) };
        assert_eq!(
            actual, expected,
            "Mismatch at [{}, {}, {}] (idx={}): expected {}, got {}",
            i, j, k, idx, expected, actual
        );
    }
}
