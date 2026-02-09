use ndarray::{Array, ArrayD, Axis, IxDyn};

use crate::interpreter::data_structures::elem_type::{ElemType, Scalar};

#[derive(Debug, Clone)]
pub enum Tile {
    I1(Array<bool, IxDyn>),
    I8(Array<i8, IxDyn>),
    I16(Array<i16, IxDyn>),
    I32(Array<i32, IxDyn>),
    I64(Array<i64, IxDyn>),
    F16(Array<f16, IxDyn>),
    F32(Array<f32, IxDyn>),
    F64(Array<f64, IxDyn>),
    Ptr(Array<*mut u8, IxDyn>),
}

impl Tile {
    /// Create a new Tile with given shape and element type, initialized to zeros
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

    pub fn elem_type(&self) -> ElemType {
        match self {
            Tile::I1(_) => ElemType::Bool,
            Tile::I8(_) => ElemType::I8,
            Tile::I16(_) => ElemType::I16,
            Tile::I32(_) => ElemType::I32,
            Tile::I64(_) => ElemType::I64,
            Tile::F16(_) => ElemType::F16,
            Tile::F32(_) => ElemType::F32,
            Tile::F64(_) => ElemType::F64,
            Tile::Ptr(_) => ElemType::Ptr,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Tile::I1(a) => a.shape().to_vec(),
            Tile::I8(a) => a.shape().to_vec(),
            Tile::I16(a) => a.shape().to_vec(),
            Tile::I32(a) => a.shape().to_vec(),
            Tile::I64(a) => a.shape().to_vec(),
            Tile::F16(a) => a.shape().to_vec(),
            Tile::F32(a) => a.shape().to_vec(),
            Tile::F64(a) => a.shape().to_vec(),
            Tile::Ptr(a) => a.shape().to_vec(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }
}
