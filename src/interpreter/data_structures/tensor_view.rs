use crate::interpreter::data_structures::elem_type::{ElemType, Scalar};
use crate::interpreter::data_structures::tile::Tile;
use log::debug;
use log::trace;

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
    /// dim_map = [0, 3, 1, 2] means:
    ///     tile dim [0] -> tensor dim [0]
    ///     tile dim [1] -> tensor dim [3]
    ///     tile dim [2] -> tensor dim [1]
    ///     tile dim [3] -> tensor dim [2]
    dim_map: Vec<i32>,
    /// Whether out-of-bounds accesses are masked
    masked: bool,
    /// Padding value for masked loads
    padding_value: Option<Scalar>,
}

unsafe impl Sync for PartitionView {}
unsafe impl Send for PartitionView {}

impl TensorView {
    /// Create a new TensorView.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `base_ptr` points to valid memory of sufficient size
    /// - The memory region is valid for the lifetime of this TensorView
    /// - Shape and strides correctly describe the tensor layout
    pub fn new(base_ptr: *mut u8, elem_type: ElemType, shape: Vec<i64>, strides: Vec<i64>) -> Self {
        assert_eq!(
            shape.len(),
            strides.len(),
            "Shape and strides must have the same length, but got shape.len() = {}, strides.len() = {}",
            shape.len(),
            strides.len()
        );

        TensorView {
            base_ptr,
            elem_type,
            shape,
            strides,
        }
    }

    /// Get the base pointer address
    pub fn base_ptr(&self) -> *mut u8 {
        self.base_ptr
    }

    /// Get the element type
    pub fn elem_type(&self) -> ElemType {
        self.elem_type
    }

    /// Get the shape
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Get the strides (in elements, not bytes)
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    /// Get the rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

impl PartitionView {
    /// Create a new PartitionView.
    ///
    /// # Arguments
    /// - `tensor_view`: The underlying tensor view to partition
    /// - `tile_shape`: The shape of each tile (power-of-2 dimensions)
    /// - `dim_map`: Permutation mapping tile dimensions to view dimensions
    /// - `masked`: Whether out-of-bounds accesses are masked
    /// - `padding_value`: Value to use for masked out-of-bounds loads
    pub fn new(
        tensor_view: TensorView,
        tile_shape: Vec<i32>,
        dim_map: Vec<i32>,
        masked: bool,
        padding_value: Option<Scalar>,
    ) -> Self {
        assert_eq!(
            tile_shape.len(),
            dim_map.len(),
            "tile_shape and dim_map must have the same length"
        );
        assert_eq!(
            tensor_view.rank(),
            dim_map.len(),
            "tensor_view rank must match dim_map length"
        );

        // Verify dim_map is a valid permutation
        let mut sorted_map = dim_map.clone();
        sorted_map.sort();
        for (i, &v) in sorted_map.iter().enumerate() {
            assert_eq!(v, i as i32, "dim_map must be a permutation of 0..n-1");
        }

        PartitionView {
            tensor_view,
            tile_shape,
            dim_map,
            masked,
            padding_value,
        }
    }

    /// Get the underlying tensor view
    pub fn tensor_view(&self) -> &TensorView {
        &self.tensor_view
    }

    /// Get the tile shape
    pub fn tile_shape(&self) -> &[i32] {
        &self.tile_shape
    }

    /// Get the dimension mapping
    pub fn dim_map(&self) -> &[i32] {
        &self.dim_map
    }

    /// Check if masked
    pub fn is_masked(&self) -> bool {
        self.masked
    }

    /// Get the padding value
    pub fn padding_value(&self) -> Option<Scalar> {
        self.padding_value
    }
}

impl PartitionView {
    /// Calculate the index space shape (number of tiles in each dimension).
    /// For dimension i, this is ceil(tensor_shape[i] / tile_shape[i]).
    pub fn index_space_shape(&self) -> Vec<i64> {
        let view_shape = self.tensor_view.shape();
        self.tile_shape
            .iter()
            .zip(self.dim_map.iter())
            .map(|(&tile_dim, &view_dim_idx)| {
                let view_dim = view_shape[view_dim_idx as usize];
                (view_dim + tile_dim as i64 - 1) / tile_dim as i64
            })
            .collect()
    }

    /// Load a tile at the specified grid indices.
    ///
    /// # Algorithm
    /// For a partition view with:
    /// - Tensor view: shape [S_0, ..., S_n], strides [st_0, ..., st_n]
    /// - Tile size: [T_0, ..., T_n]
    /// - Grid index: [I_0, ..., I_n]
    ///
    /// The location of element [i_0, ..., i_n] within the loaded tile is:
    /// ```text
    /// byte_offset = sum(m=0 to n) { (I_m * T_m + i_m) * st_m } * elem_size_bytes
    /// address = base_ptr + byte_offset
    /// ```
    ///
    /// Where:
    /// - I_m is the grid position (input to load_tile)
    /// - T_m is the tile dimension size
    /// - i_m is the position within the tile (iterating 0 to T_m-1)
    /// - st_m is the stride in elements (from tensor view)
    /// - elem_size_bytes is the byte size of the element type
    ///
    /// If masked and a position is out of bounds, uses padding_value.
    pub fn load_tile(&self, grid_indices: &[i64]) -> Tile {
        debug!("PartitionView::load_tile: grid_indices = {:?}", grid_indices);
        debug!("PartitionView::load_tile: tile_shape = {:?}", self.tile_shape);
        assert_eq!(
            grid_indices.len(),
            self.tile_shape.len(),
            "grid_indices must match tile rank"
        );

        let elem_type = self.tensor_view.elem_type();
        let elem_size = elem_type.size_bytes();
        let base_ptr = self.tensor_view.base_ptr();
        let view_shape = self.tensor_view.shape();
        let view_strides = self.tensor_view.strides();

        // Create output tile with tile_shape dimensions
        let tile_shape_usize: Vec<usize> = self.tile_shape.iter().map(|&x| x as usize).collect();
        let mut tile = Tile::zeros(&tile_shape_usize, elem_type);

        // Iterate over all positions in the tile
        self.iterate_tile_indices(&tile_shape_usize, |tile_indices| {
            // Compute element position in view: view_pos[m] = I_m * T_m + i_m
            // where m is the view dimension, which is dim_map[tile_dim]
            let mut view_pos = vec![0i64; view_shape.len()];
            for (tile_dim, &tile_idx) in tile_indices.iter().enumerate() {
                let view_dim = self.dim_map[tile_dim] as usize;
                view_pos[view_dim] =
                    grid_indices[tile_dim] * self.tile_shape[tile_dim] as i64 + tile_idx;
            }

            // Debug: print view_pos on first iteration
            if tile_indices.iter().all(|&x| x == 0) {
                trace!("PartitionView::load_tile: tile_indices = {:?}, view_pos = {:?}", tile_indices, view_pos);
            }

            // Check if masked and out of bounds
            let out_of_bounds = view_pos
                .iter()
                .zip(view_shape.iter())
                .any(|(&pos, &shape)| pos < 0 || pos >= shape);

            let value = if out_of_bounds {
                if self.masked {
                    // Use padding value
                    self.padding_value.unwrap_or_else(|| {
                        panic!("Out-of-bounds access in masked load without padding value")
                    })
                } else {
                    panic!("Out-of-bounds access in unmasked load")
                }
            } else {
                // Compute byte offset: sum(view_pos[m] * st_m) * elem_size
                let elem_offset: i64 = view_pos
                    .iter()
                    .zip(view_strides.iter())
                    .map(|(&pos, &stride)| pos * stride)
                    .sum();
                let byte_offset = elem_offset * elem_size as i64;

                // Load from memory
                unsafe {
                    let ptr = base_ptr.offset(byte_offset as isize);
                    self.load_scalar_from_ptr(ptr, elem_type)
                }
            };

            // Store into tile
            tile.set_scalar(tile_indices, value);
        });

        tile
    }

    /// Store a tile at the specified grid indices.
    ///
    /// If masked and a position is out of bounds, the store is skipped for that element.
    pub fn store_tile(&self, grid_indices: &[i64], tile: &Tile) {
        assert_eq!(
            grid_indices.len(),
            self.tile_shape.len(),
            "grid_indices must match tile rank"
        );
        assert_eq!(
            tile.elem_type(),
            self.tensor_view.elem_type(),
            "Tile element type must match tensor view element type"
        );
        assert_eq!(
            tile.shape(),
            self.tile_shape
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<_>>(),
            "Tile shape must match partition tile shape"
        );

        let elem_type = self.tensor_view.elem_type();
        let elem_size = elem_type.size_bytes();
        let base_ptr = self.tensor_view.base_ptr();
        let view_shape = self.tensor_view.shape();
        let view_strides = self.tensor_view.strides();

        let tile_shape_usize: Vec<usize> = self.tile_shape.iter().map(|&x| x as usize).collect();

        // Iterate over all positions in the tile
        self.iterate_tile_indices(&tile_shape_usize, |tile_indices| {
            // Compute element position in view
            let mut view_pos = vec![0i64; view_shape.len()];
            for (tile_dim, &tile_idx) in tile_indices.iter().enumerate() {
                let view_dim = self.dim_map[tile_dim] as usize;
                view_pos[view_dim] =
                    grid_indices[tile_dim] * self.tile_shape[tile_dim] as i64 + tile_idx;
            }

            // Check if out of bounds
            let out_of_bounds = view_pos
                .iter()
                .zip(view_shape.iter())
                .any(|(&pos, &shape)| pos < 0 || pos >= shape);

            if out_of_bounds {
                if self.masked {
                    // Skip this element
                    return;
                } else {
                    panic!("Out-of-bounds access in unmasked store");
                }
            }

            // Compute byte offset
            let elem_offset: i64 = view_pos
                .iter()
                .zip(view_strides.iter())
                .map(|(&pos, &stride)| pos * stride)
                .sum();
            let byte_offset = elem_offset * elem_size as i64;

            // Get value from tile
            let value = tile.get_scalar(tile_indices);

            // Store to memory
            unsafe {
                let ptr = base_ptr.offset(byte_offset as isize);
                self.store_scalar_to_ptr(ptr, value);
            }
        });
    }

    /// Helper to iterate over all indices in a tile
    fn iterate_tile_indices<F>(&self, shape: &[usize], mut f: F)
    where
        F: FnMut(&[i64]),
    {
        let rank = shape.len();
        if rank == 0 {
            f(&[]);
            return;
        }

        let mut indices = vec![0i64; rank];
        loop {
            f(&indices);

            // Increment indices
            let mut dim = rank - 1;
            loop {
                indices[dim] += 1;
                if indices[dim] < shape[dim] as i64 {
                    break;
                }
                indices[dim] = 0;
                if dim == 0 {
                    return;
                }
                dim -= 1;
            }
        }
    }

    /// Helper to load a scalar from a pointer. NOT public API. should not be used in interpreter
    ///
    /// # Safety
    /// The pointer must be valid and aligned for the element type
    unsafe fn load_scalar_from_ptr(&self, ptr: *const u8, elem_type: ElemType) -> Scalar {
        match elem_type {
            ElemType::Bool => Scalar::Bool(*(ptr as *const u8) != 0),
            ElemType::I8 => Scalar::I8(*(ptr as *const i8)),
            ElemType::I16 => Scalar::I16(*(ptr as *const i16)),
            ElemType::I32 => Scalar::I32(*(ptr as *const i32)),
            ElemType::I64 => Scalar::I64(*(ptr as *const i64)),
            ElemType::F16 => Scalar::F16(*(ptr as *const f16)),
            ElemType::F32 => Scalar::F32(*(ptr as *const f32)),
            ElemType::F64 => Scalar::F64(*(ptr as *const f64)),
            ElemType::Ptr => Scalar::Ptr(*(ptr as *const *mut u8)),
        }
    }

    /// Helper to store a scalar to a pointer. NOT public API. should not be used in interpreter
    ///
    /// # Safety
    /// The pointer must be valid and aligned for the element type
    unsafe fn store_scalar_to_ptr(&self, ptr: *mut u8, value: Scalar) {
        match value {
            Scalar::Bool(v) => *(ptr as *mut u8) = if v { 1 } else { 0 },
            Scalar::I8(v) => *(ptr as *mut i8) = v,
            Scalar::I16(v) => *(ptr as *mut i16) = v,
            Scalar::I32(v) => *(ptr as *mut i32) = v,
            Scalar::I64(v) => *(ptr as *mut i64) = v,
            Scalar::F16(v) => *(ptr as *mut f16) = v,
            Scalar::F32(v) => *(ptr as *mut f32) = v,
            Scalar::F64(v) => *(ptr as *mut f64) = v,
            Scalar::Ptr(v) => *(ptr as *mut *mut u8) = v,
        }
    }
}
