use ndarray::{Array, IxDyn};

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

    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Create a scalar (0-dim) tile from a Scalar value
    pub fn from_scalar(value: Scalar, _elem_type: ElemType) -> Self {
        match value {
            Scalar::Bool(v) => Tile::I1(Array::from_elem(IxDyn(&[]), v)),
            Scalar::I8(v) => Tile::I8(Array::from_elem(IxDyn(&[]), v)),
            Scalar::I16(v) => Tile::I16(Array::from_elem(IxDyn(&[]), v)),
            Scalar::I32(v) => Tile::I32(Array::from_elem(IxDyn(&[]), v)),
            Scalar::I64(v) => Tile::I64(Array::from_elem(IxDyn(&[]), v)),
            Scalar::F16(v) => Tile::F16(Array::from_elem(IxDyn(&[]), v)),
            Scalar::F32(v) => Tile::F32(Array::from_elem(IxDyn(&[]), v)),
            Scalar::F64(v) => Tile::F64(Array::from_elem(IxDyn(&[]), v)),
            Scalar::Ptr(v) => Tile::Ptr(Array::from_elem(IxDyn(&[]), v)),
        }
    }
}

impl Tile {
    pub fn get_scalar(&self, indices: &[i64]) -> Scalar {
        let idx_usize: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
        match self {
            Tile::I1(a) => Scalar::Bool(a[IxDyn(&idx_usize)]),
            Tile::I8(a) => Scalar::I8(a[IxDyn(&idx_usize)]),
            Tile::I16(a) => Scalar::I16(a[IxDyn(&idx_usize)]),
            Tile::I32(a) => Scalar::I32(a[IxDyn(&idx_usize)]),
            Tile::I64(a) => Scalar::I64(a[IxDyn(&idx_usize)]),
            Tile::F16(a) => Scalar::F16(a[IxDyn(&idx_usize)]),
            Tile::F32(a) => Scalar::F32(a[IxDyn(&idx_usize)]),
            Tile::F64(a) => Scalar::F64(a[IxDyn(&idx_usize)]),
            Tile::Ptr(a) => Scalar::Ptr(a[IxDyn(&idx_usize)]),
        }
    }

    pub fn set_scalar(&mut self, indices: &[i64], value: Scalar) {
        assert_eq!(
            value.elem_type(),
            self.elem_type(),
            "Scalar type must match tile element type"
        );
        let idx_usize: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
        match (self, value) {
            (Tile::I1(a), Scalar::Bool(v)) => a[IxDyn(&idx_usize)] = v,
            (Tile::I8(a), Scalar::I8(v)) => a[IxDyn(&idx_usize)] = v,
            (Tile::I16(a), Scalar::I16(v)) => a[IxDyn(&idx_usize)] = v,
            (Tile::I32(a), Scalar::I32(v)) => a[IxDyn(&idx_usize)] = v,
            (Tile::I64(a), Scalar::I64(v)) => a[IxDyn(&idx_usize)] = v,
            (Tile::F16(a), Scalar::F16(v)) => a[IxDyn(&idx_usize)] = v,
            (Tile::F32(a), Scalar::F32(v)) => a[IxDyn(&idx_usize)] = v,
            (Tile::F64(a), Scalar::F64(v)) => a[IxDyn(&idx_usize)] = v,
            (Tile::Ptr(a), Scalar::Ptr(v)) => a[IxDyn(&idx_usize)] = v,
            _ => panic!("Type mismatch in set_scalar"),
        }
    }

    pub fn to_c_contiguous(&self) -> Self {
        match self {
            Tile::I1(array_base) => Tile::I1(array_base.as_standard_layout().to_owned()),
            Tile::I8(array_base) => Tile::I8(array_base.as_standard_layout().to_owned()),
            Tile::I16(array_base) => Tile::I16(array_base.as_standard_layout().to_owned()),
            Tile::I32(array_base) => Tile::I32(array_base.as_standard_layout().to_owned()),
            Tile::I64(array_base) => Tile::I64(array_base.as_standard_layout().to_owned()),
            Tile::F16(array_base) => Tile::F16(array_base.as_standard_layout().to_owned()),
            Tile::F32(array_base) => Tile::F32(array_base.as_standard_layout().to_owned()),
            Tile::F64(array_base) => Tile::F64(array_base.as_standard_layout().to_owned()),
            Tile::Ptr(array_base) => Tile::Ptr(array_base.as_standard_layout().to_owned()),
        }
    }
}
