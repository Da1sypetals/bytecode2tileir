use core::panic;

use crate::cuda_tile_ir::attrs::{Attr, DenseStorage};
use crate::cuda_tile_ir::ids::OpId;
use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::types::{Dim, Type};
use crate::cuda_tile_ir::{OpAttrKey, Opcode};
use crate::interpreter;
use crate::interpreter::data_structures::elem_type::ElemType;
use crate::interpreter::data_structures::interpreter::ExecutionContext;
use crate::interpreter::data_structures::tile::Tile;
use crate::interpreter::data_structures::value::Value;

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
            Attr::DenseElements { storage, .. } => {
                todo!("Dense elements constant not supported yet: {:?}", storage)

                // match storage {
                //     DenseStorage::Inline(_) => {
                //         todo!()
                //     }
                //     DenseStorage::Const(_) => {
                //         todo!()
                //     }
                //     DenseStorage::Strings(_) => {
                //         todo!()
                //     }
                // }
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
        let (gx, gy, gz) = self.grid_size;

        let x_tile = Tile::I32(ndarray::Array::from_elem(ndarray::IxDyn(&[]), gx as i32));
        let y_tile = Tile::I32(ndarray::Array::from_elem(ndarray::IxDyn(&[]), gy as i32));
        let z_tile = Tile::I32(ndarray::Array::from_elem(ndarray::IxDyn(&[]), gz as i32));

        self.set_value(op.results[0], Value::Tile(x_tile));
        self.set_value(op.results[1], Value::Tile(y_tile));
        self.set_value(op.results[2], Value::Tile(z_tile));
    }

    pub fn execute_get_tile_block_id(&mut self, op: &Operation) {
        let (bx, by, bz) = self.tile_block_id;

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
