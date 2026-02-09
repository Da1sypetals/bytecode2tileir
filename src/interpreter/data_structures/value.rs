use crate::interpreter::data_structures::{
    tensor_view::{PartitionView, TensorView},
    tile::Tile,
};

pub enum Value {
    /// Scalar or tensor of data
    Tile(Tile),
    /// Tensor view (reference to memory region)
    TensorView(TensorView),
    /// Partition view (tiled view of tensor)
    PartitionView(PartitionView),
    /// Token (for memory ordering, ignored in serial execution)
    Token,
}
