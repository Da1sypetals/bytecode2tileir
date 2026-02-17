use crate::interpreter::data_structures::{tile::Tile, value::Value};
use ndarray::{Array, IxDyn};

pub type KernelArgv = Vec<Value>;

/// To workaround orphan rule
pub trait KernelGenericArgv {
    fn and<T: KernelArg>(self, arg: T) -> Self;
    fn push<T: KernelArg>(&mut self, arg: T);
}

/// To workaround orphan rule
impl KernelGenericArgv for Vec<Value> {
    fn and<T: KernelArg>(self, arg: T) -> Self {
        let mut new = self;
        new.push(arg.to_value());
        new
    }

    /// ROFL: we workaround lack of function overloading in Rust
    fn push<T: KernelArg>(&mut self, arg: T) {
        self.push(arg.to_value());
    }
}

// Definition of single Kernel Argument
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
