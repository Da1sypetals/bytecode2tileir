//! IR type system aligned with Tile IR bytecode encoding.

use crate::cuda_tile_ir::enums::FloatKind;
use crate::cuda_tile_ir::enums::PaddingValue;
use crate::cuda_tile_ir::ids::TypeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexType {
    I32,
    I64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    Static(i64),
    Dynamic,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<Dim>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Int {
        width: u8,
    },
    Float(FloatKind),
    Token,
    Ptr {
        pointee: TypeId,
    },
    Tile {
        element: TypeId,
        shape: Shape,
    },
    TensorView {
        element: TypeId,
        shape: Shape,
        strides: Vec<i64>,
        index: IndexType,
    },
    PartitionView {
        tile_shape: Vec<i32>,
        view: TypeId,
        dim_map: Vec<i32>,
        masked: bool,
        padding_value: Option<PaddingValue>,
    },
    Func {
        params: Vec<TypeId>,
        results: Vec<TypeId>,
    },
}

impl Type {
    pub fn is_scalar(&self) -> bool {
        matches!(self, Type::Int { .. } | Type::Float(_))
    }

    pub fn bit_width(&self, arena: &crate::cuda_tile_ir::arena::IrArena) -> Option<u32> {
        match self {
            Type::Int { width } => Some(*width as u32),
            Type::Float(k) => Some(k.bit_width()),
            Type::Tile { element, .. } => arena
                .types
                .get(element.0 as usize)
                .and_then(|t| t.bit_width(arena)),
            _ => None,
        }
    }
}
