use crate::cuda_tile_ir::OpAttrKey;
use crate::cuda_tile_ir::attrs::{Attr, DenseStorage};
use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::types::{Dim, Type};
use crate::interpreter;
use crate::interpreter::data_structures::elem_type::ElemType;
use crate::interpreter::data_structures::interpreter::ExecutionContext;
use crate::interpreter::data_structures::tile::Tile;
use crate::interpreter::data_structures::value::Value;
use ndarray::{Array, IxDyn, SliceInfo, SliceInfoElem};
use std::collections::HashSet;

impl ExecutionContext<'_> {
    pub fn execute_broadcast(&mut self, op: &Operation) {
        let src_value = self.get_value(op.operands[0]);
        let result_value_id = op.results[0];
        let result_value_data = self.arena.value_(result_value_id);
        let result_ty = self.arena.type_(result_value_data.ty());

        let result_shape = extract_shape(result_ty);

        let result = match src_value {
            Value::Tile(src_tile) => Value::Tile(src_tile.broadcast(&result_shape)),
            _ => panic!("Broadcast requires Tile operand"),
        };

        self.set_value(op.results[0], result);
    }

    pub fn execute_cat(&mut self, op: &Operation) {
        let lhs_value = self.get_value(op.operands[0]);
        let rhs_value = self.get_value(op.operands[1]);

        // Find Dim attribute in attrs
        let dim_attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::Dim)
            .map(|(_, id)| id)
            .expect("Cat operation missing Dim attribute");
        let dim_attr = self.arena.attr_(*dim_attr_id);

        let dim = if let Attr::Int { value, .. } = dim_attr {
            *value as usize
        } else {
            panic!("Cat Dim attribute must be Int, got {:?}", dim_attr)
        };

        let result = match (lhs_value, rhs_value) {
            (Value::Tile(lhs_tile), Value::Tile(rhs_tile)) => {
                Value::Tile(lhs_tile.cat(&rhs_tile, dim))
            }
            _ => panic!(
                "Cat requires Tile operands, got {:?}, {:?}",
                lhs_value, rhs_value
            ),
        };

        self.set_value(op.results[0], result);
    }

    pub fn execute_constant(&mut self, op: &Operation) {
        let result_value_id = op.results[0];
        let result_value_data = self.arena.value_(result_value_id);
        let result_ty = self.arena.type_(result_value_data.ty());

        // Find Value attribute in attrs
        let value_attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::Value)
            .map(|(_, id)| id)
            .expect("Constant operation missing Value attribute");
        let value_attr = self.arena.attr_(*value_attr_id);

        let (elem_type, shape) = match result_ty {
            Type::Tile { element, shape } => {
                let elem_ty = self.arena.type_(*element);
                let elem_type =
                    interpreter::type_conversion::type_to_elem_type(elem_ty, self.arena);
                (elem_type, extract_shape_from_tile_shape(shape))
            }
            _ => panic!("Constant result type must be Tile"),
        };

        let result_tile = match value_attr {
            Attr::DenseElements { ty, storage } => {
                // Get element type from the type reference in the attribute
                let elem_ty = self.arena.type_(*ty);
                let elem_type =
                    interpreter::type_conversion::type_to_elem_type(elem_ty, self.arena);

                // Get bytes from storage
                let bytes = match storage {
                    DenseStorage::Inline(data) => data.as_slice(),
                    DenseStorage::Const(const_id) => {
                        self.consts.get(*const_id).unwrap_or_else(|| {
                            panic!("Const ID {:?} not found in const pool", const_id)
                        })
                    }
                    DenseStorage::Strings(_) => {
                        panic!("String array DenseElements not supported for numeric constants");
                    }
                };

                // Convert bytes to Tile based on element type
                create_tile_from_bytes(elem_type, &shape, bytes)
            }
            // Uniform value
            Attr::Int { value, .. } => {
                // Integer constant - create tile directly without Scalar
                if shape.is_empty() {
                    // 0-dim tile (scalar)
                    match elem_type {
                        ElemType::I8 => {
                            Tile::I8(ndarray::Array::from_elem(ndarray::IxDyn(&[]), *value as i8))
                        }
                        ElemType::I16 => Tile::I16(ndarray::Array::from_elem(
                            ndarray::IxDyn(&[]),
                            *value as i16,
                        )),
                        ElemType::I32 => Tile::I32(ndarray::Array::from_elem(
                            ndarray::IxDyn(&[]),
                            *value as i32,
                        )),
                        ElemType::I64 => Tile::I64(ndarray::Array::from_elem(
                            ndarray::IxDyn(&[]),
                            *value as i64,
                        )),
                        _ => panic!("Invalid element type for integer constant"),
                    }
                } else {
                    // N-dim tile filled with value
                    match elem_type {
                        ElemType::I8 => Tile::I8(ndarray::Array::from_elem(
                            ndarray::IxDyn(&shape),
                            *value as i8,
                        )),
                        ElemType::I16 => Tile::I16(ndarray::Array::from_elem(
                            ndarray::IxDyn(&shape),
                            *value as i16,
                        )),
                        ElemType::I32 => Tile::I32(ndarray::Array::from_elem(
                            ndarray::IxDyn(&shape),
                            *value as i32,
                        )),
                        ElemType::I64 => Tile::I64(ndarray::Array::from_elem(
                            ndarray::IxDyn(&shape),
                            *value as i64,
                        )),
                        _ => panic!("Invalid element type for integer constant"),
                    }
                }
            }
            // Uniform value
            Attr::Float { kind, bits } => {
                // Float constant - create tile directly without Scalar
                // N-dim tile filled with uniform value
                match kind {
                    crate::cuda_tile_ir::enums::FloatKind::F16 => {
                        let value = f16::from_bits(*bits as u16);
                        Tile::F16(ndarray::Array::from_elem(ndarray::IxDyn(&shape), value))
                    }
                    crate::cuda_tile_ir::enums::FloatKind::F32 => {
                        let value = f32::from_bits(*bits as u32);
                        Tile::F32(ndarray::Array::from_elem(ndarray::IxDyn(&shape), value))
                    }
                    crate::cuda_tile_ir::enums::FloatKind::F64 => {
                        let value = f64::from_bits(*bits);
                        Tile::F64(ndarray::Array::from_elem(ndarray::IxDyn(&shape), value))
                    }
                    crate::cuda_tile_ir::enums::FloatKind::BF16 => {
                        panic!("BF16 float type not supported")
                    }
                    crate::cuda_tile_ir::enums::FloatKind::TF32 => {
                        panic!("TF32 float type not supported")
                    }
                    crate::cuda_tile_ir::enums::FloatKind::F8E4M3FN => {
                        panic!("F8E4M3FN float type not supported")
                    }
                    crate::cuda_tile_ir::enums::FloatKind::F8E5M2 => {
                        panic!("F8E5M2 float type not supported")
                    }
                }
            }
            _ => panic!("Constant operation missing valid value attribute"),
        };

        self.set_value(op.results[0], Value::Tile(result_tile));
    }

    /// ```mlir
    /// %0 = extract %t[%c1, %c2] : tile<32x8xf32> -> tile<4x2xf32>
    /// %result = extract %operand [%i0, %i1, %i2, ...]
    /// ```
    pub fn execute_extract(&mut self, op: &Operation) {
        let result_value_id = op.results[0];
        let result_value_data = self.arena.value_(result_value_id);

        let src_value = self.get_value(op.operands[0]);
        let result_ty = self.arena.type_(result_value_data.ty());

        let result_shape = extract_shape(result_ty);

        // Get indices from operands (op.operands[1+] are the indices)
        let indices: Vec<i64> = op.operands[1..]
            .iter()
            .map(|&id| {
                let idx_value = self.get_value(id);
                match idx_value {
                    Value::Tile(Tile::I8(arr)) => arr[ndarray::IxDyn(&[])] as i64,
                    Value::Tile(Tile::I16(arr)) => arr[ndarray::IxDyn(&[])] as i64,
                    Value::Tile(Tile::I32(arr)) => arr[ndarray::IxDyn(&[])] as i64,
                    Value::Tile(Tile::I64(arr)) => arr[ndarray::IxDyn(&[])] as i64,
                    _ => panic!(
                        "Extract indices must be integer Tile values, got {:?}",
                        idx_value
                    ),
                }
            })
            .collect();

        let result = match src_value {
            Value::Tile(src_tile) => Value::Tile(src_tile.extract(&indices, &result_shape)),
            _ => panic!("Extract requires Tile operand, got {:?}", src_value),
        };

        self.set_value(op.results[0], result);
    }

    pub fn execute_get_global(&mut self, op: &Operation) {
        // Find GlobalName attribute in attrs
        let name_attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::GlobalName)
            .map(|(_, id)| id)
            .expect("GetGlobal operation missing GlobalName attribute");
        let name_attr = self.arena.attr_(*name_attr_id);

        let global_name = match name_attr {
            Attr::String(s) => s.clone(),
            Attr::FlatSymbolRef(s) => s.clone(),
            _ => panic!("GetGlobal GlobalName attribute must be String or FlatSymbolRef"),
        };

        let global_value = self
            .globals
            .get(&global_name)
            .expect(&format!("Global variable '{}' not found", global_name));

        self.set_value(op.results[0], global_value.clone());
    }

    pub fn execute_get_num_tile_blocks(&mut self, op: &Operation) {
        let [gx, gy, gz] = self.grid_size;

        let x_tile = Tile::I32(ndarray::Array::from_elem(ndarray::IxDyn(&[]), gx as i32));
        let y_tile = Tile::I32(ndarray::Array::from_elem(ndarray::IxDyn(&[]), gy as i32));
        let z_tile = Tile::I32(ndarray::Array::from_elem(ndarray::IxDyn(&[]), gz as i32));

        self.set_value(op.results[0], Value::Tile(x_tile));
        self.set_value(op.results[1], Value::Tile(y_tile));
        self.set_value(op.results[2], Value::Tile(z_tile));
    }

    pub fn execute_get_tile_block_id(&mut self, op: &Operation) {
        let [bx, by, bz] = self.tile_block_id;

        let x_tile = Tile::I32(ndarray::Array::from_elem(ndarray::IxDyn(&[]), bx as i32));
        let y_tile = Tile::I32(ndarray::Array::from_elem(ndarray::IxDyn(&[]), by as i32));
        let z_tile = Tile::I32(ndarray::Array::from_elem(ndarray::IxDyn(&[]), bz as i32));

        self.set_value(op.results[0], Value::Tile(x_tile));
        self.set_value(op.results[1], Value::Tile(y_tile));
        self.set_value(op.results[2], Value::Tile(z_tile));
    }

    pub fn execute_iota(&mut self, op: &Operation) {
        let result_value_id = op.results[0];
        let result_value_data = self.arena.value_(result_value_id);
        let result_ty = self.arena.type_(result_value_data.ty());

        let (elem_type, length) = match result_ty {
            Type::Tile { element, shape } => {
                let elem_ty = self.arena.type_(*element);
                let elem_type =
                    crate::interpreter::type_conversion::type_to_elem_type(elem_ty, self.arena);
                let length = match shape.0.first() {
                    Some(Dim::Static(n)) => *n as usize,
                    Some(Dim::Dynamic) => panic!("Iota cannot have dynamic dimension"),
                    None => panic!("Iota result type must have at exactly one dimension, got none"),
                };
                (elem_type, length)
            }
            _ => panic!("Iota result type must be Tile, got {:?}", result_ty),
        };
        let result_tile = Tile::iota(length, elem_type);

        self.set_value(op.results[0], Value::Tile(result_tile));
    }

    pub fn execute_offset(&mut self, op: &Operation) {
        let ptr_value = self.get_value(op.operands[0]);
        let offset_value = self.get_value(op.operands[1]);

        // Offset operation has no pointee_size attribute - need to get it from type
        let result_value_id = op.results[0];
        let result_value_data = self.arena.value_(result_value_id);
        let result_ty = self.arena.type_(result_value_data.ty());

        // Get pointee size from result type (Tile<Ptr>)
        let pointee_size = match result_ty {
            Type::Tile { element, .. } => {
                let elem_ty = self.arena.type_(*element);
                match elem_ty {
                    Type::Ptr { pointee } => {
                        let pointee_ty = self.arena.type_(*pointee);
                        pointee_ty
                            .bit_width(self.arena)
                            .expect("Pointee type must have bit width")
                            as usize
                            / 8
                    }
                    _ => panic!("Offset result tile element type must be Ptr"),
                }
            }
            _ => panic!("Offset result type must be Tile"),
        };

        let result = match (ptr_value, offset_value) {
            (Value::Tile(ptr_tile), Value::Tile(offset_tile)) => {
                Value::Tile(ptr_tile.offset(&offset_tile, pointee_size))
            }
            _ => panic!("Offset requires Tile operands"),
        };

        self.set_value(op.results[0], result);
    }

    pub fn execute_permute(&mut self, op: &Operation) {
        let src_value = self.get_value(op.operands[0]);

        // Find Permutation attribute in attrs
        let perm_attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::Permutation)
            .map(|(_, id)| id)
            .expect("Permute operation missing Permutation attribute");
        let perm_attr = self.arena.attr_(*perm_attr_id);

        let permutation = match perm_attr {
            Attr::DenseI64Array(arr) => arr.iter().map(|&i| i as usize).collect::<Vec<_>>(),
            Attr::DenseI32Array(arr) => arr.iter().map(|&i| i as usize).collect::<Vec<_>>(),
            _ => panic!(
                "Permute Permutation attribute (dimension map) must be DenseI64Array or DenseI32Array, got {:?}",
                perm_attr
            ),
        };

        let result = match src_value {
            Value::Tile(src_tile) => Value::Tile(src_tile.permute(&permutation)),
            _ => panic!("Permute requires Tile operand, got {:?}", src_value),
        };

        self.set_value(op.results[0], result);
    }

    pub fn execute_reshape(&mut self, op: &Operation) {
        let src_value = self.get_value(op.operands[0]);
        let result_value_id = op.results[0];
        let result_value_data = self.arena.value_(result_value_id);
        let result_ty = self.arena.type_(result_value_data.ty());

        let result_shape = extract_shape(result_ty);

        let result = match src_value {
            Value::Tile(src_tile) => Value::Tile(src_tile.reshape(&result_shape)),
            _ => panic!("Reshape requires Tile operand"),
        };

        self.set_value(op.results[0], result);
    }

    pub fn execute_select(&mut self, op: &Operation) {
        let cond_value = self.get_value(op.operands[0]);
        let true_value = self.get_value(op.operands[1]);
        let false_value = self.get_value(op.operands[2]);

        let result = match (cond_value, true_value, false_value) {
            (Value::Tile(cond_tile), Value::Tile(true_tile), Value::Tile(false_tile)) => {
                Value::Tile(cond_tile.select(&true_tile, &false_tile))
            }
            _ => panic!("Select requires Tile operands"),
        };

        self.set_value(op.results[0], result);
    }

    pub fn execute_reduce(&mut self, _op: &Operation) {
        // TODO: Implement reduce operation
        panic!("Reduce operation not yet implemented")
    }

    pub fn execute_scan(&mut self, _op: &Operation) {
        // TODO: Implement scan operation
        panic!("Scan operation not yet implemented")
    }
}

/// Extract shape from a tile
pub fn extract_shape(ty: &Type) -> Vec<usize> {
    match ty {
        Type::Tile { shape, .. } => shape
            .0
            .iter()
            .map(|d| match d {
                Dim::Static(v) => *v as usize,
                Dim::Dynamic => panic!("Cannot get concrete shape for dynamic dimension"),
            })
            .collect(),
        _ => panic!("Extract shape called on non-tile type"),
    }
}

/// Get concrete tile shape as usize and panic on dynamic (should not happen)
pub fn extract_shape_from_tile_shape(shape: &crate::cuda_tile_ir::types::Shape) -> Vec<usize> {
    shape
        .0
        .iter()
        .map(|d| match d {
            Dim::Static(v) => *v as usize,
            Dim::Dynamic => panic!("Tile shape must be static, but got dynamic"),
        })
        .collect()
}

/// Create a Tile from raw bytes using the specified element type and shape.
///
/// This function is used to handle `Attr::DenseElements` which stores tensor data
/// as raw bytes in little-endian format.
fn create_tile_from_bytes(elem_type: ElemType, shape: &[usize], bytes: &[u8]) -> Tile {
    // Handle special bytecode encoding: if the first byte equals (len - 1),
    // it's a length prefix and the actual data starts from index 1.
    let actual_data = if !bytes.is_empty() && bytes[0] as usize == bytes.len().saturating_sub(1) {
        &bytes[1..]
    } else {
        bytes
    };

    let elem_size = elem_type.size_bytes();
    let expected_len = shape.iter().product::<usize>() * elem_size;

    // Handle splat constants: if data is exactly one element but multiple expected,
    // broadcast the single value to fill the entire shape
    let use_splat = actual_data.len() == elem_size && expected_len > elem_size;
    if use_splat {
        // Broadcast the single element - the code below will handle this
        // by repeatedly using the same single element
    } else {
        assert_eq!(
            actual_data.len(),
            expected_len,
            "DenseElements byte length mismatch: expected {}, got {}",
            expected_len,
            actual_data.len()
        );
    }

    match elem_type {
        ElemType::I8 => {
            let mut array: Array<i8, _> = unsafe { Array::uninit(IxDyn(shape)).assume_init() };
            for (i, &byte) in actual_data.iter().enumerate() {
                array.as_slice_mut().unwrap()[i] = byte as i8;
            }
            Tile::I8(array)
        }
        ElemType::I16 => {
            let mut array: Array<i16, _> = unsafe { Array::uninit(IxDyn(shape)).assume_init() };
            let slice = array.as_slice_mut().unwrap();
            for (i, chunk) in actual_data.chunks_exact(2).enumerate() {
                slice[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
            }
            Tile::I16(array)
        }
        ElemType::I32 => {
            let mut array: Array<i32, _> = unsafe { Array::uninit(IxDyn(shape)).assume_init() };
            let slice = array.as_slice_mut().unwrap();
            for (i, chunk) in actual_data.chunks_exact(4).enumerate() {
                slice[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Tile::I32(array)
        }
        ElemType::I64 => {
            let mut array: Array<i64, _> = unsafe { Array::uninit(IxDyn(shape)).assume_init() };
            let slice = array.as_slice_mut().unwrap();
            for (i, chunk) in actual_data.chunks_exact(8).enumerate() {
                slice[i] = i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
            }
            Tile::I64(array)
        }
        ElemType::F16 => {
            let mut array: Array<f16, _> = unsafe { Array::uninit(IxDyn(shape)).assume_init() };
            let slice = array.as_slice_mut().unwrap();
            for (i, chunk) in actual_data.chunks_exact(2).enumerate() {
                slice[i] = f16::from_le_bytes([chunk[0], chunk[1]]);
            }
            Tile::F16(array)
        }
        ElemType::F32 => {
            let mut array: Array<f32, _> = unsafe { Array::uninit(IxDyn(shape)).assume_init() };
            let slice = array.as_slice_mut().unwrap();
            if use_splat {
                // Broadcast single value
                let value = f32::from_le_bytes([
                    actual_data[0],
                    actual_data[1],
                    actual_data[2],
                    actual_data[3],
                ]);
                slice.fill(value);
            } else {
                for (i, chunk) in actual_data.chunks_exact(4).enumerate() {
                    slice[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            Tile::F32(array)
        }
        ElemType::F64 => {
            let mut array: Array<f64, _> = unsafe { Array::uninit(IxDyn(shape)).assume_init() };
            let slice = array.as_slice_mut().unwrap();
            for (i, chunk) in actual_data.chunks_exact(8).enumerate() {
                slice[i] = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
            }
            Tile::F64(array)
        }
        ElemType::Bool => {
            // Booleans stored as i1 (1 bit, but byte-aligned in storage)
            let mut array: Array<bool, _> = unsafe { Array::uninit(IxDyn(shape)).assume_init() };
            for (i, &byte) in actual_data.iter().enumerate() {
                array.as_slice_mut().unwrap()[i] = byte != 0;
            }
            Tile::I1(array)
        }
        ElemType::Ptr => {
            panic!("Pointer constants in DenseElements not supported");
        }
    }
}

impl Tile {
    pub fn zeros(shape: &[usize], elem_type: ElemType) -> Self {
        match elem_type {
            ElemType::Bool => Tile::I1(Array::default(IxDyn(shape))),
            ElemType::I8 => Tile::I8(Array::zeros(IxDyn(shape))),
            ElemType::I16 => Tile::I16(Array::zeros(IxDyn(shape))),
            ElemType::I32 => Tile::I32(Array::zeros(IxDyn(shape))),
            ElemType::I64 => Tile::I64(Array::zeros(IxDyn(shape))),
            ElemType::F16 => Tile::F16(Array::default(IxDyn(shape))),
            ElemType::F32 => Tile::F32(Array::zeros(IxDyn(shape))),
            ElemType::F64 => Tile::F64(Array::zeros(IxDyn(shape))),
            ElemType::Ptr => Tile::Ptr(Array::from_elem(IxDyn(shape), std::ptr::null_mut())),
        }
    }

    pub fn iota(length: usize, elem_type: ElemType) -> Self {
        match elem_type {
            ElemType::Bool => Tile::I1(
                Array::from_iter((0..length as i8).map(|v| v != 0))
                    .into_shape_with_order(IxDyn(&[length]))
                    .unwrap(),
            ),
            ElemType::I8 => Tile::I8(
                Array::from_iter(0..length as i8)
                    .into_shape_with_order(IxDyn(&[length]))
                    .unwrap(),
            ),
            ElemType::I16 => Tile::I16(
                Array::from_iter(0..length as i16)
                    .into_shape_with_order(IxDyn(&[length]))
                    .unwrap(),
            ),
            ElemType::I32 => Tile::I32(
                Array::from_iter(0..length as i32)
                    .into_shape_with_order(IxDyn(&[length]))
                    .unwrap(),
            ),
            ElemType::I64 => Tile::I64(
                Array::from_iter(0..length as i64)
                    .into_shape_with_order(IxDyn(&[length]))
                    .unwrap(),
            ),
            _ => panic!("Iota only supports integer types, got {:?}", elem_type),
        }
    }

    pub fn broadcast(&self, result_shape: &[usize]) -> Self {
        assert!(
            self.rank() <= result_shape.len(),
            "Rank mismatch in broadcast"
        );

        match self {
            Tile::I1(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::I1(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::I8(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::I8(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::I16(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::I16(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::I32(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::I32(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::I64(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::I64(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::F16(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::F16(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::F32(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::F32(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::F64(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::F64(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::Ptr(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::Ptr(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
        }
    }

    pub fn reshape(&self, result_shape: &[usize]) -> Self {
        assert_eq!(
            self.len(),
            result_shape.iter().product(),
            "Element count mismatch"
        );

        match self {
            Tile::I1(arr) => Tile::I1(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::I8(arr) => Tile::I8(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::I16(arr) => Tile::I16(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::I32(arr) => Tile::I32(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::I64(arr) => Tile::I64(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::F16(arr) => Tile::F16(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::F32(arr) => Tile::F32(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::F64(arr) => Tile::F64(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::Ptr(arr) => Tile::Ptr(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
        }
    }

    pub fn permute(&self, permutation: &[usize]) -> Self {
        assert_eq!(
            permutation.len(),
            self.rank(),
            "Permutation does not match tile rank: self.rank = {}, got permutation {:?}",
            self.rank(),
            permutation
        );

        // Verify permutation is valid
        let dedup_perm: HashSet<_> = permutation.iter().cloned().collect();
        assert_eq!(
            dedup_perm.len(),
            self.rank(),
            "Permutation is invalid: {:?}, require a permutation of integers from 0 to rank-1 inclusive",
            permutation
        );
        for dim in 0..self.rank() {
            assert!(
                dedup_perm.contains(&dim),
                "Permutation {:?} does not contain dimension {}",
                permutation,
                dim
            );
        }

        match self {
            Tile::I1(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::I1(result)
            }
            Tile::I8(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::I8(result)
            }
            Tile::I16(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::I16(result)
            }
            Tile::I32(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::I32(result)
            }
            Tile::I64(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::I64(result)
            }
            Tile::F16(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::F16(result)
            }
            Tile::F32(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::F32(result)
            }
            Tile::F64(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::F64(result)
            }
            Tile::Ptr(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::Ptr(result)
            }
        }
    }

    pub fn cat(&self, other: &Tile, dim: usize) -> Self {
        assert_eq!(self.rank(), other.rank(), "Rank mismatch in cat");
        assert!(dim < self.rank(), "Invalid dimension");

        for i in 0..self.rank() {
            if i != dim {
                // non-concatenated dim must match
                assert_eq!(self.shape()[i], other.shape()[i]);
            } else {
                // the concatenated dim is arbitrary
            }
        }

        match (self, other) {
            (Tile::I1(lhs), Tile::I1(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::I1(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::I8(lhs), Tile::I8(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::I8(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::I16(lhs), Tile::I16(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::I16(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::I32(lhs), Tile::I32(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::I32(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::I64(lhs), Tile::I64(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::I64(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::F16(lhs), Tile::F16(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::F16(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::F32(lhs), Tile::F32(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::F32(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::F64(lhs), Tile::F64(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::F64(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::Ptr(lhs), Tile::Ptr(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::Ptr(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            _ => panic!("Type mismatch in cat"),
        }
    }

    /// Extract sub-tile from tile.
    /// self is tiled into result_shape;
    /// indices is the index of subtile, not element.
    pub fn extract(&self, indices: &[i64], result_shape: &[usize]) -> Self {
        assert_eq!(indices.len(), self.rank(), "Index rank mismatch");

        let start_idx: Vec<usize> = indices
            .iter()
            .zip(result_shape)
            .map(|(&idx, &shape)| idx as usize * shape)
            .collect();
        let end_idx: Vec<usize> = indices
            .iter()
            .zip(result_shape)
            .map(|(&idx, &shape)| (idx + 1) as usize * shape)
            .collect();

        let slice: SliceInfo<_, IxDyn, IxDyn> = start_idx
            .into_iter()
            .zip(end_idx.into_iter())
            .map(|(start, end)| (start..end).into())
            .collect::<Vec<SliceInfoElem>>()
            .try_into()
            .expect("Failed to convert ranges to slice");

        match self {
            Tile::I1(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::I1(tile)
            }
            Tile::I8(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::I8(tile)
            }
            Tile::I16(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::I16(tile)
            }
            Tile::I32(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::I32(tile)
            }
            Tile::I64(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::I64(tile)
            }
            Tile::F16(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::F16(tile)
            }
            Tile::F32(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::F32(tile)
            }
            Tile::F64(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::F64(tile)
            }
            Tile::Ptr(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::Ptr(tile)
            }
        }
    }

    pub fn offset(&self, offset: &Tile, pointee_size: usize) -> Self {
        assert_eq!(
            self.shape(),
            offset.shape(),
            "Offset shape mismatch: {:?} vs {:?}",
            self.shape(),
            offset.shape()
        );

        let Tile::Ptr(ptrs) = self else {
            panic!("Offset requires Ptr tile and integer offset tile");
        };

        let offsets_isize: Array<isize, IxDyn> = match offset {
            Tile::I8(arr) => arr.mapv(|v| v as isize),
            Tile::I16(arr) => arr.mapv(|v| v as isize),
            Tile::I32(arr) => arr.mapv(|v| v as isize),
            Tile::I64(arr) => arr.mapv(|v| v as isize),
            _ => panic!("Offset requires Ptr tile and integer offset tile"),
        };

        let result: Vec<*mut u8> = ptrs
            .indexed_iter()
            .zip(offsets_isize.indexed_iter())
            .map(|((_, ptr), (_, off))| {
                // indexed_iter returns nd-index and value
                let addr = unsafe { ptr.offset((off).wrapping_mul(pointee_size as isize)) };
                addr as *mut u8
            })
            .collect();

        Tile::Ptr(Array::from_shape_vec(IxDyn(ptrs.shape()), result).unwrap())
    }

    pub fn select(&self, val_if_true: &Tile, val_if_false: &Tile) -> Self {
        // FIXME: should NOT use Vec to collect results
        // Should use Zip / mapv or something.
        assert_eq!(
            self.shape(),
            val_if_true.shape(),
            "Expect condition and true value to have same shapes"
        );

        assert_eq!(
            self.shape(),
            val_if_false.shape(),
            "Expect condition and false value to have same shapes"
        );

        match (self, val_if_true, val_if_false) {
            (Tile::I1(cond), Tile::I1(true_vals), Tile::I1(false_vals)) => {
                let result: Vec<bool> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c)) // flatten
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond { true_val } else { false_val }
                        },
                    )
                    .collect();
                Tile::I1(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::I8(true_vals), Tile::I8(false_vals)) => {
                let result: Vec<i8> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond { true_val } else { false_val }
                        },
                    )
                    .collect();
                Tile::I8(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::I16(true_vals), Tile::I16(false_vals)) => {
                let result: Vec<i16> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond { true_val } else { false_val }
                        },
                    )
                    .collect();
                Tile::I16(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::I32(true_vals), Tile::I32(false_vals)) => {
                let result: Vec<i32> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond { true_val } else { false_val }
                        },
                    )
                    .collect();
                Tile::I32(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::I64(true_vals), Tile::I64(false_vals)) => {
                let result: Vec<i64> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond { true_val } else { false_val }
                        },
                    )
                    .collect();
                Tile::I64(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::F16(true_vals), Tile::F16(false_vals)) => {
                let result: Vec<f16> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond { true_val } else { false_val }
                        },
                    )
                    .collect();
                Tile::F16(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::F32(true_vals), Tile::F32(false_vals)) => {
                let result: Vec<f32> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond { true_val } else { false_val }
                        },
                    )
                    .collect();
                Tile::F32(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::F64(true_vals), Tile::F64(false_vals)) => {
                let result: Vec<f64> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond { true_val } else { false_val }
                        },
                    )
                    .collect();
                Tile::F64(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            _ => panic!("Type mismatch in select"),
        }
    }
}
