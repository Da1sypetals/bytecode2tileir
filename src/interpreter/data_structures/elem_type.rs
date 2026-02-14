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

#[derive(Debug, Clone, Copy, PartialEq)]
/// Just a helper.
/// DO NOT use in interpreter code unless you have good reasons to.
/// Use 0-dim Tile instead. Everything in Tile IR model is a Tile,
/// and scalar CANNOT be used interchangeably with 0-dim Tile.
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

impl Eq for Scalar {}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Scalar::Bool(a), Scalar::Bool(b)) => a.partial_cmp(b),
            (Scalar::I8(a), Scalar::I8(b)) => a.partial_cmp(b),
            (Scalar::I16(a), Scalar::I16(b)) => a.partial_cmp(b),
            (Scalar::I32(a), Scalar::I32(b)) => a.partial_cmp(b),
            (Scalar::I64(a), Scalar::I64(b)) => a.partial_cmp(b),
            (Scalar::F16(a), Scalar::F16(b)) => {
                if a.is_nan() && b.is_nan() {
                    Some(std::cmp::Ordering::Equal)
                } else {
                    a.partial_cmp(b)
                }
            }
            (Scalar::F32(a), Scalar::F32(b)) => {
                if a.is_nan() && b.is_nan() {
                    Some(std::cmp::Ordering::Equal)
                } else {
                    a.partial_cmp(b)
                }
            }
            (Scalar::F64(a), Scalar::F64(b)) => {
                if a.is_nan() && b.is_nan() {
                    Some(std::cmp::Ordering::Equal)
                } else {
                    a.partial_cmp(b)
                }
            }
            (Scalar::Ptr(a), Scalar::Ptr(b)) => (*a as usize).partial_cmp(&(*b as usize)),
            _ => None,
        }
    }
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

// UNSAFE: We don't care about thread safety for GPU-like operations
// GPU execution doesn't guarantee thread safety either
unsafe impl Send for Scalar {}
unsafe impl Sync for Scalar {}
