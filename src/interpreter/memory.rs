use crate::cuda_tile_ir::OpAttrKey;
use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::types::{Dim, Type};
use crate::interpreter::data_structures::elem_type::ElemType;
use crate::interpreter::data_structures::interpreter::ExecutionContext;
use crate::interpreter::data_structures::tile::Tile;
use crate::interpreter::data_structures::value::Value;
use crate::interpreter::type_conversion;
use ndarray::{ArrayView, Dimension, IxDyn};

impl ExecutionContext<'_> {
    /// Load and gather data from global memory using a pointer tile.
    ///
    /// ```mlir
    /// %result, %result_token = cuda_tile.load_ptr_tko weak, tl_blk
    ///     %source, %mask, %paddingValue token=%token
    /// ```
    ///
    /// https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-load-ptr-tko
    /// The paddingValue must have the same shape of the source tile. If it is not present, the value of masked elements are undefined.
    pub fn execute_load_ptr_tko(&mut self, op: &Operation) {
        // Extract attributes
        // FIXME: Validate memory ordering semantics and scope
        // FIXME: Token is ignored
        let _ordering_attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::MemoryOrderingSemantics)
            .map(|(_, id)| id)
            .expect("LoadPtrTko operation missing MemoryOrderingSemantics attribute");
        let _scope_attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::MemoryScope)
            .map(|(_, id)| id)
            .expect("LoadPtrTko operation missing MemoryScope attribute");

        // Extract operands
        let ptr_tile = match self.get_value(op.operands[0]) {
            Value::Tile(tile) => tile,
            other => panic!(
                "LoadPtrTko source operand must be pointer tile, got {:?}",
                other
            ),
        };
        // Get mask (or unmasked)
        let (mask_tile, padding_tile) = if op.operands.len() > 1 {
            assert!(
                op.operands.len() == 3,
                "Masked LoadPtrTko must have 3 operands, got {}",
                op.operands.len()
            );
            let mask_value = self.get_value(op.operands[1]);
            let padding_value = self.get_value(op.operands[2]);

            match (mask_value, padding_value) {
                (Value::Tile(mask_tile), Value::Tile(padding_tile)) => {
                    (Some(mask_tile), Some(padding_tile))
                }
                _ => panic!(
                    "Mask and padding operands must be tile values, got\nmask: {:?}\npadding: {:?}",
                    mask_value, padding_value
                ),
            }
        } else {
            (None, None)
        };

        // Get result type to determine shape and element type
        let result_value_id = op.results[0];
        let result_value_data = self.arena.value_(result_value_id);
        let result_ty = self.arena.type_(result_value_data.ty());

        // For now:
        // Ignore the pointee-type on the pointer
        // Infer the pointee data type with the result data type.
        // This could be a problem when there is a mismatch.
        let (result_shape, elem_type) = match result_ty {
            Type::Tile { element, shape } => {
                let elem_ty = self.arena.type_(*element);
                let elem_type = type_conversion::type_to_elem_type(elem_ty, self.arena);
                (
                    shape
                        .0
                        .iter()
                        .map(|d| match d {
                            Dim::Static(v) => *v as usize,
                            Dim::Dynamic => {
                                panic!("Cannot create result tile with dynamic dimension")
                            }
                        })
                        .collect::<Vec<_>>(),
                    elem_type,
                )
            }
            _ => panic!("LoadPtrTko result type must be Tile, got {:?}", result_ty),
        };

        // Use Tile method to gather data from pointers
        let result_tile = ptr_tile.load_from_ptrs(
            mask_tile.as_deref(),
            padding_tile.as_deref(),
            &result_shape,
            elem_type,
        );

        // Store result tile and token
        self.set_value(op.results[0], Value::Tile(result_tile));
        self.set_value(op.results[1], Value::Token);
    }

    /// Store and scatter data from a tile to global memory using pointer tile.
    ///
    /// ```mlir
    /// %result_token = cuda_tile.store_ptr_tko weak, tl_blk
    ///     %destination, %value, %mask token=%token
    /// ```
    pub fn execute_store_ptr_tko(&mut self, op: &Operation) {
        // Extract attributes
        // FIXME: Validate memory ordering semantics and scope
        let _ordering_attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::MemoryOrderingSemantics)
            .map(|(_, id)| id)
            .expect("StorePtrTko operation missing MemoryOrderingSemantics attribute");

        let _scope_attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::MemoryScope)
            .map(|(_, id)| id)
            .expect("StorePtrTko operation missing MemoryScope attribute");

        // Extract operands
        let dest_tile = match self.get_value(op.operands[0]) {
            Value::Tile(tile) => tile,
            other => panic!(
                "StorePtrTko destination operand must be pointer tile, got {:?}",
                other
            ),
        };
        let value_tile = match self.get_value(op.operands[1]) {
            Value::Tile(tile) => tile,
            other => panic!("StorePtrTko value operand must be tile, got {:?}", other),
        };
        let mask_tile = if op.operands.len() > 2 {
            Some(match self.get_value(op.operands[2]) {
                Value::Tile(tile) => tile,
                other => panic!("StorePtrTko mask operand must be tile, got {:?}", other),
            })
        } else {
            None
        };
        // FIXME: Token is ignored

        // Use Tile method to scatter data to pointers
        value_tile.store_to_ptrs(&dest_tile, mask_tile.as_deref());

        // Store result token
        self.set_value(op.results[0], Value::Token);
    }
}

impl Tile {
    /// Gather data from memory locations pointed to by this pointer tile.
    ///
    /// # Arguments
    /// * `mask` - Optional mask tile (I1) to control which elements are loaded
    /// * `padding` - Optional padding tile for masked-out elements
    /// * `result_shape` - Shape of the result tile
    /// * `elem_type` - Element type of the result tile
    ///
    /// # Returns
    /// A new Tile containing gathered data
    ///
    /// # Panics
    /// * If self is not Tile::Ptr
    /// * If mask tile exists and is not Tile::I1
    pub fn load_from_ptrs(
        &self,
        mask: Option<&Tile>,
        padding: Option<&Tile>,
        result_shape: &[usize],
        elem_type: ElemType,
    ) -> Self {
        // Extract pointer array from self
        let ptr_arr = match self {
            Tile::Ptr(arr) => arr,
            _ => panic!("load_from_ptrs can only be called on pointer tiles"),
        };

        // Create result tile
        let mut result_tile = Self::zeros(result_shape, elem_type);

        // Convert mask tile to ArrayView for efficient access
        let mask_view = mask.and_then(|t| match t {
            Tile::I1(arr) => Some(arr.view()),
            _ => panic!("Mask tile must be I1 (boolean) type"),
        });

        // Gather: iterate over all indices using indexed_iter
        match &mut result_tile {
            Tile::I1(result_arr) => {
                for (idx, result_elem) in result_arr.indexed_iter_mut() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        *result_elem = unsafe { ptr.read() != 0 };
                    } else {
                        *result_elem = match padding {
                            Some(Tile::I1(arr)) => {
                                let idx_usize: Vec<usize> =
                                    idx_i64.iter().map(|&i| i as usize).collect();
                                arr[IxDyn(&idx_usize)]
                            }
                            _ => panic!("Padding tile must match result type bool"),
                        };
                    }
                }
            }
            Tile::I8(result_arr) => {
                for (idx, result_elem) in result_arr.indexed_iter_mut() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        *result_elem = unsafe { ptr.cast::<i8>().read() };
                    } else {
                        *result_elem = match padding {
                            Some(Tile::I8(arr)) => {
                                let idx_usize: Vec<usize> =
                                    idx_i64.iter().map(|&i| i as usize).collect();
                                arr[IxDyn(&idx_usize)]
                            }
                            _ => panic!("Padding tile must match result type i8"),
                        };
                    }
                }
            }
            Tile::I16(result_arr) => {
                for (idx, result_elem) in result_arr.indexed_iter_mut() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        *result_elem = unsafe { ptr.cast::<i16>().read() };
                    } else {
                        *result_elem = match padding {
                            Some(Tile::I16(arr)) => {
                                let idx_usize: Vec<usize> =
                                    idx_i64.iter().map(|&i| i as usize).collect();
                                arr[IxDyn(&idx_usize)]
                            }
                            _ => panic!("Padding tile must match result type i16"),
                        };
                    }
                }
            }
            Tile::I32(result_arr) => {
                for (idx, result_elem) in result_arr.indexed_iter_mut() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        *result_elem = unsafe { ptr.cast::<i32>().read() };
                    } else {
                        *result_elem = match padding {
                            Some(Tile::I32(arr)) => {
                                let idx_usize: Vec<usize> =
                                    idx_i64.iter().map(|&i| i as usize).collect();
                                arr[IxDyn(&idx_usize)]
                            }
                            _ => panic!("Padding tile must match result type i32"),
                        };
                    }
                }
            }
            Tile::I64(result_arr) => {
                for (idx, result_elem) in result_arr.indexed_iter_mut() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        *result_elem = unsafe { ptr.cast::<i64>().read() };
                    } else {
                        *result_elem = match padding {
                            Some(Tile::I64(arr)) => {
                                let idx_usize: Vec<usize> =
                                    idx_i64.iter().map(|&i| i as usize).collect();
                                arr[IxDyn(&idx_usize)]
                            }
                            _ => panic!("Padding tile must match result type i64"),
                        };
                    }
                }
            }
            Tile::F16(result_arr) => {
                for (idx, result_elem) in result_arr.indexed_iter_mut() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        *result_elem = unsafe { f16::from_bits(ptr.cast::<u16>().read()) };
                    } else {
                        *result_elem = match padding {
                            Some(Tile::F16(arr)) => {
                                let idx_usize: Vec<usize> =
                                    idx_i64.iter().map(|&i| i as usize).collect();
                                arr[IxDyn(&idx_usize)]
                            }
                            _ => panic!("Padding tile must match result type f16"),
                        };
                    }
                }
            }
            Tile::F32(result_arr) => {
                for (idx, result_elem) in result_arr.indexed_iter_mut() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        *result_elem = unsafe { ptr.cast::<f32>().read() };
                    } else {
                        *result_elem = match padding {
                            Some(Tile::F32(arr)) => {
                                let idx_usize: Vec<usize> =
                                    idx_i64.iter().map(|&i| i as usize).collect();
                                arr[IxDyn(&idx_usize)]
                            }
                            _ => panic!("Padding tile must match result type f32"),
                        };
                    }
                }
            }
            Tile::F64(result_arr) => {
                for (idx, result_elem) in result_arr.indexed_iter_mut() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        *result_elem = unsafe { ptr.cast::<f64>().read() };
                    } else {
                        *result_elem = match padding {
                            Some(Tile::F64(arr)) => {
                                let idx_usize: Vec<usize> =
                                    idx_i64.iter().map(|&i| i as usize).collect();
                                arr[IxDyn(&idx_usize)]
                            }
                            _ => panic!("Padding tile must match result type f64"),
                        };
                    }
                }
            }
            Tile::Ptr(_) => panic!("Cannot load into pointer tiles"),
        }

        result_tile
    }

    /// Scatter this tile's data to memory locations pointed to by a pointer tile.
    ///
    /// # Arguments
    /// * `ptr_tile` - Pointer tile containing destination memory addresses
    /// * `mask` - Optional mask tile (I1) to control which elements are stored
    ///
    /// # Panics
    /// * If ptr_tile is not Tile::Ptr
    /// * If mask tile exists and is not Tile::I1
    pub fn store_to_ptrs(&self, ptr_tile: &Tile, mask: Option<&Tile>) {
        // Extract pointer array from ptr_tile
        let ptr_arr = match ptr_tile {
            Tile::Ptr(arr) => arr,
            _ => panic!("ptr_tile must be a pointer tile"),
        };

        // Convert mask tile to ArrayView for efficient access
        let mask_view = mask.and_then(|t| match t {
            Tile::I1(arr) => Some(arr.view()),
            _ => panic!("Mask tile must be I1 (boolean) type"),
        });

        // Scatter: iterate using indexed_iter
        match self {
            Tile::I1(val_arr) => {
                for (idx, &val) in val_arr.indexed_iter() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        unsafe { ptr.cast::<bool>().write(val) };
                    }
                }
            }
            Tile::I8(val_arr) => {
                for (idx, &val) in val_arr.indexed_iter() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        unsafe { ptr.cast::<i8>().write(val) };
                    }
                }
            }
            Tile::I16(val_arr) => {
                for (idx, &val) in val_arr.indexed_iter() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        unsafe { ptr.cast::<i16>().write(val) };
                    }
                }
            }
            Tile::I32(val_arr) => {
                for (idx, &val) in val_arr.indexed_iter() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        unsafe { ptr.cast::<i32>().write(val) };
                    }
                }
            }
            Tile::I64(val_arr) => {
                for (idx, &val) in val_arr.indexed_iter() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        unsafe { ptr.cast::<i64>().write(val) };
                    }
                }
            }
            Tile::F16(val_arr) => {
                for (idx, &val) in val_arr.indexed_iter() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        unsafe { ptr.cast::<u16>().write(val.to_bits()) };
                    }
                }
            }
            Tile::F32(val_arr) => {
                for (idx, &val) in val_arr.indexed_iter() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        unsafe { ptr.cast::<f32>().write(val) };
                    }
                }
            }
            Tile::F64(val_arr) => {
                for (idx, &val) in val_arr.indexed_iter() {
                    let idx_i64: Vec<i64> = idx.slice().iter().map(|&i| i as i64).collect();
                    if check_mask_tile(&mask_view, &idx_i64) {
                        let ptr = ptr_arr[idx];
                        unsafe { ptr.cast::<f64>().write(val) };
                    }
                }
            }
            Tile::Ptr(_) => panic!("Cannot store pointer tiles"),
        }
    }
}

/// Check if mask allows access at given indices.
fn check_mask_tile(mask_tile: &Option<ArrayView<bool, IxDyn>>, idx: &[i64]) -> bool {
    match mask_tile {
        Some(mask_arr) => {
            let idx_usize: Vec<usize> = idx.iter().map(|&i| i as usize).collect();
            mask_arr[IxDyn(&idx_usize)]
        }
        None => true,
    }
}
