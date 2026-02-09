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
