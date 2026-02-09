use crate::interpreter::data_structures::elem_type::{ElemType, Scalar};

/// TensorView: A structured pointer to tensor data in memory.
/// Does NOT own data - holds pointer + shape + stride metadata.
/// The underlying memory can be modified through this view.
#[derive(Debug, Clone)]
pub struct TensorView {
    /// Base pointer address
    base_ptr: *mut u8,
    /// Element type
    elem_type: ElemType,
    /// Shape (can have dynamic dimensions, but values are resolved at runtime)
    shape: Vec<i64>,
    /// Strides (in elements, not bytes)
    strides: Vec<i64>,
}

unsafe impl Sync for TensorView {}
unsafe impl Send for TensorView {}

#[derive(Debug, Clone)]
pub struct PartitionView {
    /// The underlying tensor view
    tensor_view: TensorView,
    /// Tile shape (static, each dimension is power of 2)
    tile_shape: Vec<i32>,
    /// Dimension mapping from tile dims to view dims
    dim_map: Vec<i32>,
    /// Whether out-of-bounds accesses are masked
    masked: bool,
    /// Padding value for masked loads
    padding_value: Option<Scalar>,
}

unsafe impl Sync for PartitionView {}
unsafe impl Send for PartitionView {}
