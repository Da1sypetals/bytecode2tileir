#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    Ptr,
}

#[derive(Debug, Clone, Copy)]
pub enum Scalar {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F16(f16),
    F32(f32),
    F64(f64),
    Ptr(*mut u8),
}

impl Scalar {
    pub fn elem_type(&self) -> ElemType {
        match self {
            Scalar::Bool(_) => ElemType::Bool,
            Scalar::I8(_) => ElemType::I8,
            Scalar::I16(_) => ElemType::I16,
            Scalar::I32(_) => ElemType::I32,
            Scalar::I64(_) => ElemType::I64,
            Scalar::F16(_) => ElemType::F16,
            Scalar::F32(_) => ElemType::F32,
            Scalar::F64(_) => ElemType::F64,
            Scalar::Ptr(_) => ElemType::Ptr,
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.elem_type().size_bytes()
    }
}

impl ElemType {
    pub fn size_bytes(&self) -> usize {
        match self {
            ElemType::Bool => 1,
            ElemType::I8 => 1,
            ElemType::I16 => 2,
            ElemType::I32 => 4,
            ElemType::I64 => 8,
            ElemType::F16 => 2,
            ElemType::F32 => 4,
            ElemType::F64 => 8,
            ElemType::Ptr => 8,
        }
    }
}
