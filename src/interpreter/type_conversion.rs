//! Type conversion utilities between IR types and interpreter types.

use crate::cuda_tile_ir::arena::IrArena;
use crate::cuda_tile_ir::enums::FloatKind;
use crate::cuda_tile_ir::types::Type;
use crate::interpreter::data_structures::elem_type::ElemType;

/// Convert a CUDA Tile IR type to an interpreter element type.
pub fn type_to_elem_type(ty: &Type, arena: &IrArena) -> ElemType {
    match ty {
        Type::Int { width: 1 } => ElemType::Bool,
        Type::Int { width: 8 } => ElemType::I8,
        Type::Int { width: 16 } => ElemType::I16,
        Type::Int { width: 32 } => ElemType::I32,
        Type::Int { width: 64 } => ElemType::I64,
        Type::Float(FloatKind::F16) => ElemType::F16,
        Type::Float(FloatKind::F32) => ElemType::F32,
        Type::Float(FloatKind::F64) => ElemType::F64,
        Type::Tile { element, .. } | Type::TensorView { element, .. } => {
            let elem_ty = arena.types.get(element.0 as usize)
                .unwrap_or_else(|| panic!("Element type {} not found", element.0));
            type_to_elem_type(&elem_ty, arena)
        }
        Type::Ptr { .. } => ElemType::Ptr,
        Type::Token => panic!("Token type has no element type"),
        Type::Func { .. } => panic!("Func type has no element type"),
        Type::PartitionView { .. } => panic!("PartitionView type has no element type"),
        _ => panic!("Type {:?} has no element type", ty),
    }
}
