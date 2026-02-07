//! IR attribute definitions (interned by `IrArena`).

use crate::bytecode::format::{ConstId, StrId};
use crate::cuda_tile_ir::enums::{
    AtomicRMWMode, ComparisonOrdering, ComparisonPredicate, FloatKind, IntegerOverflow,
    MemoryOrdering, MemoryScope, PaddingValue, RoundingMode, Signedness,
};
use crate::cuda_tile_ir::ids::{AttrId, TypeId};

#[derive(Debug, Clone)]
pub enum DenseStorage {
    Inline(Vec<u8>),
    Const(ConstId),
    Strings(Vec<StrId>),
}

#[derive(Debug, Clone)]
pub enum Attr {
    Unit,
    Bool(bool),
    Int {
        ty: TypeId,
        value: i64,
    },
    Float {
        kind: FloatKind,
        bits: u64,
    },
    Type(TypeId),
    String(String),
    FlatSymbolRef(String),

    Array(Vec<AttrId>),
    DenseI32Array(Vec<i32>),
    DenseI64Array(Vec<i64>),
    DenseElements {
        ty: TypeId,
        storage: DenseStorage,
    },

    Dict(Vec<(String, AttrId)>),
    OptimizationHints(Vec<(String, AttrId)>),

    DivBy {
        divisor: u64,
        unsigned_int: bool,
        every: Option<i64>,
        along: Option<i64>,
    },
    SameElements(Vec<i64>),
    Bounded {
        lb: Option<i64>,
        ub: Option<i64>,
    },
    NonNegative,

    // Enum wrappers
    RoundingMode(RoundingMode),
    Signedness(Signedness),
    ComparisonPredicate(ComparisonPredicate),
    ComparisonOrdering(ComparisonOrdering),
    AtomicRMWMode(AtomicRMWMode),
    MemoryScope(MemoryScope),
    MemoryOrdering(MemoryOrdering),
    IntegerOverflow(IntegerOverflow),
    PaddingValue(PaddingValue),
}

impl Attr {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Attr::Int { value, .. } => Some(*value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Attr::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Attr::String(s) => Some(s.as_str()),
            Attr::FlatSymbolRef(s) => Some(s.as_str()),
            _ => None,
        }
    }
}
