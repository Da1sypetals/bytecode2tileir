use std::collections::HashSet;

use ndarray::{Array, IxDyn, SliceInfo, SliceInfoElem};

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

impl Tile {
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

    pub fn iota(length: usize, elem_type: ElemType) -> Self {
        match elem_type {
            ElemType::Bool => Tile::I1(
                Array::from_iter((0..length as i8).map(|v| v != 0))
                    .into_shape_with_order(IxDyn(&[length]))
                    .unwrap(),
            ),
            ElemType::I8 => Tile::I8(
                Array::from_iter(0..length as i8)
                    .into_shape_with_order(IxDyn(&[length]))
                    .unwrap(),
            ),
            ElemType::I16 => Tile::I16(
                Array::from_iter(0..length as i16)
                    .into_shape_with_order(IxDyn(&[length]))
                    .unwrap(),
            ),
            ElemType::I32 => Tile::I32(
                Array::from_iter(0..length as i32)
                    .into_shape_with_order(IxDyn(&[length]))
                    .unwrap(),
            ),
            ElemType::I64 => Tile::I64(
                Array::from_iter(0..length as i64)
                    .into_shape_with_order(IxDyn(&[length]))
                    .unwrap(),
            ),
            _ => panic!("Iota only supports integer types, got {:?}", elem_type),
        }
    }

    pub fn broadcast(&self, result_shape: &[usize]) -> Self {
        assert!(
            self.rank() <= result_shape.len(),
            "Rank mismatch in broadcast"
        );

        match self {
            Tile::I1(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::I1(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::I8(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::I8(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::I16(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::I16(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::I32(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::I32(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::I64(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::I64(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::F16(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::F16(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::F32(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::F32(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::F64(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::F64(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
            Tile::Ptr(arr) => {
                //
                match arr.broadcast(IxDyn(result_shape)) {
                    Some(b) => Tile::Ptr(b.to_owned()),
                    None => panic!(
                        "Broadcast failed:\nsrc.shape: {:?}\ndst.shape: {:?}",
                        arr.shape(),
                        result_shape
                    ),
                }
            }
        }
    }

    pub fn reshape(&self, result_shape: &[usize]) -> Self {
        assert_eq!(
            self.len(),
            result_shape.iter().product(),
            "Element count mismatch"
        );

        match self {
            Tile::I1(arr) => Tile::I1(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::I8(arr) => Tile::I8(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::I16(arr) => Tile::I16(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::I32(arr) => Tile::I32(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::I64(arr) => Tile::I64(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::F16(arr) => Tile::F16(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::F32(arr) => Tile::F32(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::F64(arr) => Tile::F64(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
            Tile::Ptr(arr) => Tile::Ptr(
                arr.clone()
                    .into_shape_with_order(IxDyn(result_shape))
                    .unwrap()
                    .as_standard_layout()
                    .to_owned(),
            ),
        }
    }

    pub fn permute(&self, permutation: &[usize]) -> Self {
        assert_eq!(
            permutation.len(),
            self.rank(),
            "Permutation does not match tile rank: self.rank = {}, got permutation {:?}",
            self.rank(),
            permutation
        );

        // Verify permutation is valid
        let dedup_perm: HashSet<_> = permutation.iter().cloned().collect();
        assert_eq!(
            dedup_perm.len(),
            self.rank(),
            "Permutation is invalid: {:?}, require a permutation of integers from 0 to rank-1 inclusive",
            permutation
        );
        for dim in 0..self.rank() {
            assert!(
                dedup_perm.contains(&dim),
                "Permutation {:?} does not contain dimension {}",
                permutation,
                dim
            );
        }

        match self {
            Tile::I1(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::I1(result)
            }
            Tile::I8(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::I8(result)
            }
            Tile::I16(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::I16(result)
            }
            Tile::I32(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::I32(result)
            }
            Tile::I64(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::I64(result)
            }
            Tile::F16(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::F16(result)
            }
            Tile::F32(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::F32(result)
            }
            Tile::F64(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::F64(result)
            }
            Tile::Ptr(arr) => {
                let result = arr.clone().permuted_axes(permutation);
                Tile::Ptr(result)
            }
        }
    }

    pub fn cat(&self, other: &Tile, dim: usize) -> Self {
        assert_eq!(self.rank(), other.rank(), "Rank mismatch in cat");
        assert!(dim < self.rank(), "Invalid dimension");

        for i in 0..self.rank() {
            if i != dim {
                // non-concatenated dim must match
                assert_eq!(self.shape()[i], other.shape()[i]);
            } else {
                // the concatenated dim is arbitrary
            }
        }

        match (self, other) {
            (Tile::I1(lhs), Tile::I1(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::I1(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::I8(lhs), Tile::I8(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::I8(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::I16(lhs), Tile::I16(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::I16(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::I32(lhs), Tile::I32(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::I32(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::I64(lhs), Tile::I64(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::I64(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::F16(lhs), Tile::F16(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::F16(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::F32(lhs), Tile::F32(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::F32(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::F64(lhs), Tile::F64(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::F64(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            (Tile::Ptr(lhs), Tile::Ptr(rhs)) => {
                let axis = ndarray::Axis(dim);
                Tile::Ptr(
                    ndarray::concatenate(axis, &[lhs.view(), rhs.view()])
                        .unwrap()
                        .to_owned(),
                )
            }
            _ => panic!("Type mismatch in cat"),
        }
    }

    /// Extract sub-tile from tile.
    /// self is tiled into result_shape;
    /// indices is the index of subtile, not element.
    pub fn extract(&self, indices: &[i64], result_shape: &[usize]) -> Self {
        assert_eq!(indices.len(), self.rank(), "Index rank mismatch");

        let start_idx: Vec<usize> = indices
            .iter()
            .zip(result_shape)
            .map(|(&idx, &shape)| idx as usize * shape)
            .collect();
        let end_idx: Vec<usize> = indices
            .iter()
            .zip(result_shape)
            .map(|(&idx, &shape)| (idx + 1) as usize * shape)
            .collect();

        let slice: SliceInfo<_, IxDyn, IxDyn> = start_idx
            .into_iter()
            .zip(end_idx.into_iter())
            .map(|(start, end)| (start..end).into())
            .collect::<Vec<SliceInfoElem>>()
            .try_into()
            .expect("Failed to convert ranges to slice");

        match self {
            Tile::I1(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::I1(tile)
            }
            Tile::I8(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::I8(tile)
            }
            Tile::I16(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::I16(tile)
            }
            Tile::I32(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::I32(tile)
            }
            Tile::I64(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::I64(tile)
            }
            Tile::F16(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::F16(tile)
            }
            Tile::F32(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::F32(tile)
            }
            Tile::F64(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::F64(tile)
            }
            Tile::Ptr(array_base) => {
                let tile = array_base
                    .slice(&slice)
                    .as_standard_layout()
                    .into_dyn()
                    .to_owned();

                Tile::Ptr(tile)
            }
        }
    }

    pub fn offset(&self, offset: &Tile, pointee_size: usize) -> Self {
        assert_eq!(
            self.shape(),
            offset.shape(),
            "Offset shape mismatch: {:?} vs {:?}",
            self.shape(),
            offset.shape()
        );

        let Tile::Ptr(ptrs) = self else {
            panic!("Offset requires Ptr tile and integer offset tile");
        };

        let offsets_isize: Array<isize, IxDyn> = match offset {
            Tile::I8(arr) => arr.mapv(|v| v as isize),
            Tile::I16(arr) => arr.mapv(|v| v as isize),
            Tile::I32(arr) => arr.mapv(|v| v as isize),
            Tile::I64(arr) => arr.mapv(|v| v as isize),
            _ => panic!("Offset requires Ptr tile and integer offset tile"),
        };

        let result: Vec<*mut u8> = ptrs
            .indexed_iter()
            .zip(offsets_isize.indexed_iter())
            .map(|((_, ptr), (_, off))| {
                // indexed_iter returns nd-index and value
                let addr = unsafe { ptr.offset((off).wrapping_mul(pointee_size as isize)) };
                addr as *mut u8
            })
            .collect();

        Tile::Ptr(Array::from_shape_vec(IxDyn(ptrs.shape()), result).unwrap())
    }

    pub fn select(&self, val_if_true: &Tile, val_if_false: &Tile) -> Self {
        assert_eq!(
            self.shape(),
            val_if_true.shape(),
            "Expect condition and true value to have same shapes"
        );

        assert_eq!(
            self.shape(),
            val_if_false.shape(),
            "Expect condition and false value to have same shapes"
        );

        match (self, val_if_true, val_if_false) {
            (Tile::I1(cond), Tile::I1(true_vals), Tile::I1(false_vals)) => {
                let result: Vec<bool> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c)) // flatten
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond {
                                true_val
                            } else {
                                false_val
                            }
                        },
                    )
                    .collect();
                Tile::I1(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::I8(true_vals), Tile::I8(false_vals)) => {
                let result: Vec<i8> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond {
                                true_val
                            } else {
                                false_val
                            }
                        },
                    )
                    .collect();
                Tile::I8(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::I16(true_vals), Tile::I16(false_vals)) => {
                let result: Vec<i16> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond {
                                true_val
                            } else {
                                false_val
                            }
                        },
                    )
                    .collect();
                Tile::I16(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::I32(true_vals), Tile::I32(false_vals)) => {
                let result: Vec<i32> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond {
                                true_val
                            } else {
                                false_val
                            }
                        },
                    )
                    .collect();
                Tile::I32(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::I64(true_vals), Tile::I64(false_vals)) => {
                let result: Vec<i64> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond {
                                true_val
                            } else {
                                false_val
                            }
                        },
                    )
                    .collect();
                Tile::I64(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::F16(true_vals), Tile::F16(false_vals)) => {
                let result: Vec<f16> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond {
                                true_val
                            } else {
                                false_val
                            }
                        },
                    )
                    .collect();
                Tile::F16(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::F32(true_vals), Tile::F32(false_vals)) => {
                let result: Vec<f32> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond {
                                true_val
                            } else {
                                false_val
                            }
                        },
                    )
                    .collect();
                Tile::F32(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            (Tile::I1(cond), Tile::F64(true_vals), Tile::F64(false_vals)) => {
                let result: Vec<f64> = cond
                    .indexed_iter()
                    .zip(true_vals.indexed_iter())
                    .zip(false_vals.indexed_iter())
                    .map(|((a, b), c)| (a, b, c))
                    .map(
                        |((_, &cond), (_, &true_val), (_, &false_val))| {
                            if cond {
                                true_val
                            } else {
                                false_val
                            }
                        },
                    )
                    .collect();
                Tile::F64(Array::from_shape_vec(IxDyn(cond.shape()), result).unwrap())
            }
            _ => panic!("Type mismatch in select"),
        }
    }
}
