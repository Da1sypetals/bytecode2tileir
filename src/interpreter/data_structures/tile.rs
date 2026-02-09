use ndarray::{Array, IxDyn};

#[derive(Debug, Clone)]
pub enum Tile {
    I1(Array<bool, IxDyn>),
    I8(Array<i8, IxDyn>),
    I16(Array<i16, IxDyn>),
    I32(Array<i32, IxDyn>),
    I64(Array<i64, IxDyn>),
    F32(Array<f32, IxDyn>),
    F64(Array<f64, IxDyn>),
    Ptr(Array<usize, IxDyn>),
}
