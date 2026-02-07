//! Bytecode serialization tags.

/// Type tag for bytecode serialization
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeTag {
    I1 = 0,
    I8 = 1,
    I16 = 2,
    I32 = 3,
    I64 = 4,
    F16 = 5,
    BF16 = 6,
    F32 = 7,
    TF32 = 8,
    F64 = 9,
    F8E4M3FN = 10,
    F8E5M2 = 11,
    Pointer = 12,
    Tile = 13,
    TensorView = 14,
    PartitionView = 15,
    Func = 16,
    Token = 17,
}

impl TryFrom<u8> for TypeTag {
    type Error = ();
    fn try_from(v: u8) -> Result<Self, ()> {
        match v {
            0 => Ok(Self::I1),
            1 => Ok(Self::I8),
            2 => Ok(Self::I16),
            3 => Ok(Self::I32),
            4 => Ok(Self::I64),
            5 => Ok(Self::F16),
            6 => Ok(Self::BF16),
            7 => Ok(Self::F32),
            8 => Ok(Self::TF32),
            9 => Ok(Self::F64),
            10 => Ok(Self::F8E4M3FN),
            11 => Ok(Self::F8E5M2),
            12 => Ok(Self::Pointer),
            13 => Ok(Self::Tile),
            14 => Ok(Self::TensorView),
            15 => Ok(Self::PartitionView),
            16 => Ok(Self::Func),
            17 => Ok(Self::Token),
            _ => Err(()),
        }
    }
}

/// Attribute tag for bytecode serialization
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeTag {
    Integer = 1,
    Float = 2,
    Bool = 3,
    Type = 4,
    String = 5,
    Array = 6,
    DenseElements = 7,
    DivBy = 8,
    SameElements = 9,
    Dictionary = 10,
    OptimizationHints = 11,
    NonNegative = 12,
}

impl TryFrom<u8> for AttributeTag {
    type Error = ();
    fn try_from(v: u8) -> Result<Self, ()> {
        match v {
            1 => Ok(Self::Integer),
            2 => Ok(Self::Float),
            3 => Ok(Self::Bool),
            4 => Ok(Self::Type),
            5 => Ok(Self::String),
            6 => Ok(Self::Array),
            7 => Ok(Self::DenseElements),
            8 => Ok(Self::DivBy),
            9 => Ok(Self::SameElements),
            10 => Ok(Self::Dictionary),
            11 => Ok(Self::OptimizationHints),
            12 => Ok(Self::NonNegative),
            _ => Err(()),
        }
    }
}
