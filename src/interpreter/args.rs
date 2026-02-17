use crate::interpreter::data_structures::{tile::Tile, value::Value};
use ndarray::{Array, IxDyn};

pub struct KernelArgIterator {
    
}

pub trait KernelArg {
    fn to_value(self) -> Value;
}

impl KernelArg for bool {
    fn to_value(self) -> Value {
        Value::Tile(Tile::I1(Array::from_elem(IxDyn(&[]), self)))
    }
}

impl KernelArg for i8 {
    fn to_value(self) -> Value {
        Value::Tile(Tile::I8(Array::from_elem(IxDyn(&[]), self)))
    }
}

impl KernelArg for i16 {
    fn to_value(self) -> Value {
        Value::Tile(Tile::I16(Array::from_elem(IxDyn(&[]), self)))
    }
}

impl KernelArg for i32 {
    fn to_value(self) -> Value {
        Value::Tile(Tile::I32(Array::from_elem(IxDyn(&[]), self)))
    }
}

impl KernelArg for i64 {
    fn to_value(self) -> Value {
        Value::Tile(Tile::I64(Array::from_elem(IxDyn(&[]), self)))
    }
}

impl KernelArg for f16 {
    fn to_value(self) -> Value {
        Value::Tile(Tile::F16(Array::from_elem(IxDyn(&[]), self)))
    }
}

impl KernelArg for f32 {
    fn to_value(self) -> Value {
        Value::Tile(Tile::F32(Array::from_elem(IxDyn(&[]), self)))
    }
}

impl KernelArg for f64 {
    fn to_value(self) -> Value {
        Value::Tile(Tile::F64(Array::from_elem(IxDyn(&[]), self)))
    }
}

impl KernelArg for *mut u8 {
    fn to_value(self) -> Value {
        Value::Tile(Tile::Ptr(Array::from_elem(IxDyn(&[]), self)))
    }
}
