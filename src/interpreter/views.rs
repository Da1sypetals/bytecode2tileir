// View operations for TileIR interpreter (Section 8.11)
//
// Implements: get_index_space_shape, get_tensor_shape, load_view_tko,
//             make_partition_view, make_tensor_view, store_view_tko

use crate::cuda_tile_ir::OpAttrKey;
use crate::cuda_tile_ir::attrs::Attr;
use crate::cuda_tile_ir::ir::Operation;
use crate::cuda_tile_ir::types::{Dim, Type};
use crate::interpreter::data_structures::elem_type::{ElemType, Scalar};
use crate::interpreter::data_structures::interpreter::ExecutionContext;
use crate::interpreter::data_structures::tensor_view::{PartitionView, TensorView};
use crate::interpreter::data_structures::tile::Tile;
use crate::interpreter::data_structures::value::Value;
use crate::interpreter::type_conversion;
use log::debug;

impl ExecutionContext<'_> {
    // =========================================================================
    // Helper Functions (private)
    // =========================================================================

    fn extract_memory_ordering(
        &self,
        op: &Operation,
    ) -> crate::cuda_tile_ir::enums::MemoryOrdering {
        let attr_id = op
            .attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::MemoryOrderingSemantics)
            .map(|(_, id)| id)
            .expect("Memory operation missing MemoryOrderingSemantics attribute");
        match self.arena.attr_(*attr_id) {
            Attr::MemoryOrdering(ord) => *ord,
            _ => panic!("MemoryOrderingSemantics attribute must be MemoryOrdering enum"),
        }
    }

    fn extract_memory_scope(
        &self,
        op: &Operation,
    ) -> Option<crate::cuda_tile_ir::enums::MemoryScope> {
        op.attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::MemoryScope)
            .map(|(_, id)| match self.arena.attr_(*id) {
                Attr::MemoryScope(scope) => *scope,
                _ => panic!("MemoryScope attribute must be MemoryScope enum"),
            })
    }

    fn extract_operand_segment_sizes(&self, op: &Operation) -> Vec<i32> {
        op.attrs
            .iter()
            .find(|(key, _)| *key == OpAttrKey::OperandSegmentSizes)
            .map(|(_, id)| match self.arena.attr_(*id) {
                Attr::DenseI32Array(sizes) => sizes.clone(),
                _ => panic!("OperandSegmentSizes must be DenseI32Array"),
            })
            .unwrap_or_default()
    }

    // =========================================================================
    // Shape Query Operations
    // =========================================================================

    /// 8.11.1. cuda_tile.get_index_space_shape - Return index space dimension size
    pub fn execute_get_index_space_shape(&mut self, op: &Operation) {
        let src_value = self.get_value(op.operands[0]);
        let shape = match src_value {
            Value::PartitionView(pv) => pv.index_space_shape(),
            other => panic!(
                "GetIndexSpaceShape requires PartitionView operand, got {:?}",
                other
            ),
        };

        // Create result tiles for each dimension
        for (i, &result_id) in op.results.iter().enumerate() {
            let value_data = self.arena.value_(result_id);
            let result_ty = self.arena.type_(value_data.ty());

            let index_ty = match result_ty {
                Type::Int { width } => *width,
                Type::Tile { element, .. } => {
                    let elem_ty = self.arena.type_(*element);
                    match elem_ty {
                        Type::Int { width } => *width,
                        _ => panic!("GetIndexSpaceShape result tile element must be integer type"),
                    }
                }
                _ => panic!("GetIndexSpaceShape result must be integer or tile type"),
            };

            let elem_type = type_conversion::int_width_to_elem_type(index_ty);
            let tile = Tile::from_scalar(Scalar::from_i64(shape[i], index_ty), elem_type);
            self.set_value(result_id, Value::Tile(tile));
        }
    }

    /// 8.11.2. cuda_tile.get_tensor_shape - Returns the shape of a tensor view
    pub fn execute_get_tensor_shape(&mut self, op: &Operation) {
        let src_value = self.get_value(op.operands[0]);
        let shape = match src_value {
            Value::TensorView(tv) => tv.shape().to_vec(),
            other => panic!(
                "GetTensorShape requires TensorView operand, got {:?}",
                other
            ),
        };

        // Create result tiles for each dimension
        for (i, &result_id) in op.results.iter().enumerate() {
            let value_data = self.arena.value_(result_id);
            let result_ty = self.arena.type_(value_data.ty());

            let index_ty = match result_ty {
                Type::Int { width } => *width,
                Type::Tile { element, .. } => {
                    let elem_ty = self.arena.type_(*element);
                    match elem_ty {
                        Type::Int { width } => *width,
                        _ => panic!("GetTensorShape result tile element must be integer type"),
                    }
                }
                _ => panic!("GetTensorShape result must be integer or tile type"),
            };

            let elem_type = type_conversion::int_width_to_elem_type(index_ty);
            let tile = Tile::from_scalar(Scalar::from_i64(shape[i], index_ty), elem_type);
            self.set_value(result_id, Value::Tile(tile));
        }
    }

    // =========================================================================
    // View Creation Operations
    // =========================================================================

    /// 8.11.5. cuda_tile.make_tensor_view - Create tensor_view from a pointer
    pub fn execute_make_tensor_view(&mut self, op: &Operation) {
        let segment_sizes = self.extract_operand_segment_sizes(op);
        let base_value = self.get_value(op.operands[0]);

        // Get the pointer from the tile value
        let base_ptr = match base_value {
            Value::Tile(tile) => match tile {
                Tile::Ptr(arr) => {
                    if arr.len() != 1 {
                        panic!("MakeTensorView base must be a scalar pointer");
                    }
                    *arr.first().unwrap()
                }
                _ => panic!("MakeTensorView base must be pointer tile"),
            },
            _ => panic!("MakeTensorView base must be tile value"),
        };

        // Get result type to determine element type, shape, strides
        let result_value_id = op.results[0];
        let result_value_data = self.arena.value_(result_value_id);
        let result_ty = self.arena.type_(result_value_data.ty());

        let (elem_type, static_shape, static_strides) = match result_ty {
            Type::TensorView {
                element,
                shape,
                strides,
                index: _,
            } => {
                let elem_ty = self.arena.type_(*element);
                let elem_type = type_conversion::type_to_elem_type(elem_ty, self.arena);
                (elem_type, shape, strides)
            }
            _ => panic!("MakeTensorView result must be TensorView type"),
        };

        // Extract dynamic shape values (if any)
        let mut shape = Vec::new();
        let num_shape_dims = segment_sizes.get(1).map(|&x| x as usize).unwrap_or(0);
        if num_shape_dims > 0 {
            let shape_start = 1;
            for i in 0..num_shape_dims {
                let shape_value = self.get_value(op.operands[shape_start + i]);
                match shape_value {
                    Value::Tile(tile) => {
                        let dim = tile.get_i64_scalar();
                        shape.push(dim);
                    }
                    _ => panic!("Dynamic shape must be tile values"),
                }
            }
        } else {
            // Use static shape from type
            for dim in static_shape.0.iter() {
                match dim {
                    Dim::Static(v) => shape.push(*v),
                    Dim::Dynamic => panic!("Dynamic dimension requires operand"),
                }
            }
        }

        // Extract stride values (merge dynamic and static)
        // Negative values in static_strides indicate dynamic strides (use operand)
        // Non-negative values are static strides
        let mut strides = Vec::new();
        let num_stride_dims = segment_sizes.get(2).map(|&x| x as usize).unwrap_or(0);
        if num_stride_dims > 0 {
            let stride_start = 1 + num_shape_dims;
            let mut dynamic_stride_idx = 0;
            for &static_stride in static_strides.iter() {
                if static_stride < 0 {
                    // Dynamic stride - get from operand
                    let stride_value = self.get_value(op.operands[stride_start + dynamic_stride_idx]);
                    dynamic_stride_idx += 1;
                    match stride_value {
                        Value::Tile(tile) => {
                            let stride = tile.get_i64_scalar();
                            strides.push(stride);
                        }
                        _ => panic!("Dynamic stride must be tile values"),
                    }
                } else {
                    // Static stride - use value from type
                    strides.push(static_stride);
                }
            }
        } else {
            // Use static strides from type
            strides = static_strides.clone();
        }

        let tensor_view = TensorView::new(base_ptr, elem_type, shape, strides);
        self.set_value(result_value_id, Value::TensorView(tensor_view));
    }

    /// 8.11.4. cuda_tile.make_partition_view - Create a partition view from a tensor view
    pub fn execute_make_partition_view(&mut self, op: &Operation) {
        let tensor_view_value = self.get_value(op.operands[0]);
        let tensor_view = match tensor_view_value {
            Value::TensorView(tv) => tv,
            _ => panic!("MakePartitionView requires TensorView operand"),
        };

        // Get result type to determine tile_shape, dim_map, masked, padding_value
        let result_value_id = op.results[0];
        let result_value_data = self.arena.value_(result_value_id);
        let result_ty = self.arena.type_(result_value_data.ty());

        let (tile_shape, dim_map, masked, padding_value) = match result_ty {
            Type::PartitionView {
                tile_shape,
                view: _,
                dim_map,
                masked,
                padding_value,
            } => (tile_shape.clone(), dim_map.clone(), *masked, *padding_value),
            _ => panic!("MakePartitionView result must be PartitionView type"),
        };

        // Convert padding_value from Attr to Scalar
        let padding_scalar = match padding_value {
            Some(crate::cuda_tile_ir::enums::PaddingValue::Zero) => Some(Scalar::I32(0)),
            Some(crate::cuda_tile_ir::enums::PaddingValue::NegZero) => Some(Scalar::F32(-0.0)),
            Some(crate::cuda_tile_ir::enums::PaddingValue::Nan) => {
                // Use NAN based on tensor view element type
                match tensor_view.elem_type() {
                    ElemType::F16 => Some(Scalar::F16(f16::NAN)),
                    ElemType::F32 => Some(Scalar::F32(f32::NAN)),
                    ElemType::F64 => Some(Scalar::F64(f64::NAN)),
                    _ => Some(Scalar::I32(0)),
                }
            }
            Some(crate::cuda_tile_ir::enums::PaddingValue::PosInf) => {
                match tensor_view.elem_type() {
                    ElemType::F16 => Some(Scalar::F16(f16::INFINITY)),
                    ElemType::F32 => Some(Scalar::F32(f32::INFINITY)),
                    ElemType::F64 => Some(Scalar::F64(f64::INFINITY)),
                    _ => Some(Scalar::I32(i32::MAX)),
                }
            }
            Some(crate::cuda_tile_ir::enums::PaddingValue::NegInf) => {
                match tensor_view.elem_type() {
                    ElemType::F16 => Some(Scalar::F16(f16::NEG_INFINITY)),
                    ElemType::F32 => Some(Scalar::F32(f32::NEG_INFINITY)),
                    ElemType::F64 => Some(Scalar::F64(f64::NEG_INFINITY)),
                    _ => Some(Scalar::I32(i32::MIN)),
                }
            }
            None => None,
        };

        let partition_view = PartitionView::new(
            tensor_view.clone(),
            tile_shape,
            dim_map,
            masked,
            padding_scalar,
        );
        self.set_value(result_value_id, Value::PartitionView(partition_view));
    }

    // =========================================================================
    // Load/Store Operations
    // =========================================================================

    /// 8.11.3. cuda_tile.load_view_tko - Load a tile from a tile view
    pub fn execute_load_view_tko(&mut self, op: &Operation) {
        // Extract attributes (currently ignored for serial execution)
        let _memory_ordering = self.extract_memory_ordering(op);
        let _memory_scope = self.extract_memory_scope(op);

        let segment_sizes = self.extract_operand_segment_sizes(op);

        // Operands: [view, index..., token?]
        // Segment sizes: [1 (view), n (indices), 0/1 (token)]
        let view_value = self.get_value(op.operands[0]);
        let num_indices = segment_sizes.get(1).copied().unwrap_or(1) as usize;

        let mut indices = Vec::with_capacity(num_indices);
        for i in 0..num_indices {
            let index_value = self.get_value(op.operands[1 + i]);
            match index_value {
                Value::Tile(tile) => {
                    let idx = tile.get_i64_scalar();
                    indices.push(idx);
                }
                _ => panic!("LoadViewTko indices must be tile values"),
            }

        debug!("LoadViewTko: index[0] (X)={}, index[1] (Y)={}", indices[0], if indices.len() > 1 { indices[1] } else { -1 });
        }

        // Load tile based on view type
        let tile = match view_value {
            Value::PartitionView(pv) => pv.load_tile(&indices),
            _ => panic!("LoadViewTko view must be PartitionView"),
        };

        self.set_value(op.results[0], Value::Tile(tile));

        // Set token result
        if op.results.len() > 1 {
            self.set_value(op.results[1], Value::Token);
        }
    }

    /// 8.11.6. cuda_tile.store_view_tko - Store a tile into a tile view
    pub fn execute_store_view_tko(&mut self, op: &Operation) {
        // Extract attributes (currently ignored for serial execution)
        let _memory_ordering = self.extract_memory_ordering(op);
        let _memory_scope = self.extract_memory_scope(op);

        let segment_sizes = self.extract_operand_segment_sizes(op);

        // Operands: [tile, view, index..., token?]
        // Segment sizes: [1 (tile), 1 (view), n (indices), 0/1 (token)]
        let tile_value = self.get_value(op.operands[0]);
        let view_value = self.get_value(op.operands[1]);
        let num_indices = segment_sizes.get(2).copied().unwrap_or(1) as usize;

        let mut indices = Vec::with_capacity(num_indices);
        for i in 0..num_indices {
            let index_value = self.get_value(op.operands[2 + i]);
            match index_value {
                Value::Tile(tile) => {
                    let idx = tile.get_i64_scalar();
                    indices.push(idx);
                }
                _ => panic!("StoreViewTko indices must be tile values"),
            }
        }

        let tile = match tile_value {
            Value::Tile(t) => t,
            _ => panic!("StoreViewTko tile must be Tile value"),
        };

        match view_value {
            Value::PartitionView(pv) => {
                pv.store_tile(&indices, &tile);
        debug!("StoreViewTko: indices = {:?}", indices);
        debug!("StoreViewTko: index[0] (X)={}, index[1] (Y)={}", indices[0], if indices.len() > 1 { indices[1] } else { -1 });
            }
            _ => panic!("StoreViewTko view must be PartitionView"),
        }

        // Set token result
        self.set_value(op.results[0], Value::Token);
    }
}

// ============================================================================
// Tile Helper Methods
// ============================================================================

impl Tile {
    fn get_i64_scalar(&self) -> i64 {
        match self {
            Tile::I8(arr) => {
                assert_eq!(arr.len(), 1, "Expected scalar tile");
                arr[ndarray::IxDyn(&[])] as i64
            }
            Tile::I16(arr) => {
                assert_eq!(arr.len(), 1, "Expected scalar tile");
                arr[ndarray::IxDyn(&[])] as i64
            }
            Tile::I32(arr) => {
                assert_eq!(arr.len(), 1, "Expected scalar tile");
                arr[ndarray::IxDyn(&[])] as i64
            }
            Tile::I64(arr) => {
                assert_eq!(arr.len(), 1, "Expected scalar tile");
                arr[ndarray::IxDyn(&[])]
            }
            _ => panic!("Cannot get i64 scalar from non-integer tile"),
        }
    }
}
