# 8.11. Views

Views are a structured way to interact with tensors in memory. They are described in both the types section Tensor View and the semantics section Views. Views are the primary way to interact with global memory in Tile IR. A common pattern is to construct a Tensor View from a pointer with `cuda_tile.make_tensor_view` and then use the `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko` operations to read and write to them. For larger tensors, loading the entire tensor is not efficient and therefore we have a sub-view Partition View which allows a user to tile a tensor_view.

## 8.11.1. cuda_tile.get_index_space_shape

Return index space dimension size

```
cuda_tile.get_index_space_shape %src
```

### Parameters

- **src** (`view_type`) - The source view type.

### Results

- **result** (`Variadic<tile<any>>`) - The shape of the index space, each value representing the size of the corresponding dimension.

### Description

The `get_index_space_shape` operation returns the shape of the index space of `src`. The result types must be the same as the view’s index type, and the number of results must be the same as the view’s index rank. If the index space shape sizes do not fit within the provided type, behavior is undefined.

### Constraints

The operation is pure and does not perform any memory side effects.

## 8.11.2. cuda_tile.get_tensor_shape

Returns the shape of a tensor view

```
cuda_tile.get_tensor_shape %src
```

### Parameters

- **src** (`tensor_view`) - The source tensor view.

### Results

- **result** (`Variadic<tile<any>>`) - The shape of the tensor, each value representing the size of the corresponding dimension.

### Description

The `get_tensor_shape` operation returns the shape of the tensor backing the provided `tensor_view`. If the tensor shape sizes do not fit within the provided type, behavior is undefined.

### Constraints

The operation is pure and does not perform any memory side effects.

## 8.11.3. cuda_tile.load_view_tko

Load a tile from a tile view

```
cuda_tile.load_view_tko %memory_ordering_semantics %memory_scope %view %index %token %optimization_hints
```

### Parameters

- **memory_ordering_semantics** (`MemoryOrderingSemantics`) - The memory ordering semantics for the load operation.
- **memory_scope** (`MemoryScope`) - The memory scope for the atomic operation.
- **view** (`view_type`) - The view from which the tile will be loaded.
- **index** (`Variadic<tile<any>>`) - The n-dimensional index of the desired element to load from the view.
- **token** (`token`) - The optional token for the load operation.
- **optimization_hints** (`OptimizationHints`) - Optimization hints for operation.

### Results

- **tile** (`tile`) - The loaded tile.
- **result_token** (`token`) - The result token.

### Description

The `load_view_tko` operation loads a tile from a tile view. A view is mapping from view-space indices to a particular element in the view, each view type has a defined mapping from view-space indices to tiles produced from elements of the view. For example, the Partition View partitions a Tensor View into a grid of equally sized tiles. The view indices one of the partitioned tiles in the grid.

For a given view the rank of the indices must match the rank of the view’s index space. The space of valid indices depends on which view is passed to the operation. For example the index space of a Partition View is equal to the rank of the partitioned tiles. Out of bounds accesses are handling according to the semantics of the tile view.

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required. For example, `weak` ordering allows the compiler to assume that there are no concurrent accesses to any accessed location. For more information, refer to the memory model section of the specification.

- **weak** - No concurrent accesses to the source/destination location.
- **relaxed** - There may be concurrent access to the location, but this access does not establish a happens-before relationship.
- **acquire** - There may be concurrent accesses to the location. If this acquire observes a release operation, then happens before is established.
  - *Note: The following variants are not supported by this operation: release, acq_rel.*

The `memory_scope` attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

- **tl_blk** - There may be concurrent accesses from within the same tile block.
- **device** - There may be concurrent accesses from within the same device (i.e., GPU).
- **sys** - There may be concurrent accesses from anywhere within the system (i.e., all devices).

The `optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries. The hints are specified for each architecture (e.g., `sm_100`, `sm_120`) and for each architecture the user can specify specific hints for each operation.

- **num_cta_in_cga** - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for `cuda_tile.entry`.
- **allow_tma** - suggest whether to use TMA for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.
- **latency** - latency hint for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.

For example they can be annotated as:

```
optimization_hints =< sm_100 = { num_cta_in_cga = 8 }, sm_120 = { num_cta_in_cga = 16 } >
```

### Constraints

The operation must encode variadic operand segment sizes in attributes.

### Examples

```
%tensor_view = make_tensor_view %ptr, shape = [8192, 128], strides = [128, 1] : tensor_view<8192x128xf32, strides = [128, 1]>

// This example uses the PartitionView on a 8192x128xf32 tensor_view,
// dividing the tensor_view in tiles of 64x64.
%view = make_partition_view %tensor_view : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>>

%c0 = constant dense<0> : tile<i32>
%c1 = constant dense<1> : tile<i32>

// Load a tile at index (0, 0) in the view's index space.
// For this PartitionView, this is the rectangular tile such that
// X=[0,64) and Y=[0,64), in the coordinates of tiles.
%tile0, %res_token0 = load_view_tko weak %view [%c0, %c0] : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> tile<64x64xf32>, token

// Load a tile at index (0, 1) in the view's index space.
// For this PartitionView, this is the rectangular tile such that
// X=[0,64) and Y=[64,128), in the coordinates of tiles.
%tile1, %res_token1 = load_view_tko weak %view [%c0, %c1] : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> tile<64x64xf32>, token

// Same example as above but with memory token as input.
%token = make_token : token
%tile2, %res_token2 = load_view_tko weak %view [%c0, %c1] token = %token : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> tile<64x64xf32>, token

// Loads a tile at the dynamic index (%index, %index) in the view's index space.
%tile3, %res_token3 = load_view_tko weak %view [%index, %index] : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> tile<64x64xf32>, token
```

See `cuda_tile.load_view_tko_0` for the full example listing.

## 8.11.4. cuda_tile.make_partition_view

Create a partition view from a tensor view

```
cuda_tile.make_partition_view %tensor_view
```

### Parameters

- **tensor_view** (`tensor_view`) - The source tensor view to create a partition view from.

### Results

- **result** (`partition_view`) - The created partition view.

### Description

The `make_partition_view` operation creates a `partition_view` from a `tensor_view`. For more details about partition views see Partition View. The operation uses the type constraints of the input tensor view and the annotated return type to perform the partitioning.

The tensor view’s type contains its physical layout in the form of shapes and strides and the partition view containts the logical size of a single tile. The resulting partition view can be loaded from using `cuda_tile.load_view_tko` and stored to using `cuda_tile.store_view_tko`. The view memory options act on the computed index space of the partition view see Tensor View and Partition View for detailed semantics.

### Constraints

The operation is conditionally speculatable based on the specific operands and attributes. The operation may be speculatively executed without side effects. The operation is pure and does not perform any memory side effects.

### Examples

```
%tensor_view0 = make_tensor_view %ptr, shape = [8192, 8192, 64], strides = [524288, 64, 1] : tensor_view<8192x8192x64xf32, strides = [524288, 64, 1]>

// Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
// the provided tensor_view.
make_partition_view %tensor_view0 : partition_view<tile = (1024x1x32), tensor_view<8192x8192x64xf32, strides = [524288, 64, 1]>>

%s0 = constant dense<8192> : tile<i32>
%str0 = constant dense<524288> : tile<i32>
%tensor_view1 = make_tensor_view %ptr, shape = [%s0, 8192, 64], strides = [%str0, 64, 1] : tensor_view<?x8192x64xf32, strides = [?, 64, 1]>

// Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
// the provided tensor_view, with masking. The provided tensor_view has a
// dynamically-sized dimension.
make_partition_view %tensor_view1 : partition_view<tensor_view<?x8192x64xf32, strides = [?, 64, 1]>>
```

See `cuda_tile.make_partition_view_0` for the full example listing.

## 8.11.5. cuda_tile.make_tensor_view

Create :code:`tensor_view` from a pointer to global memory

```
cuda_tile.make_tensor_view %base %dynamicShape %dynamicStrides
```

### Parameters

- **base** (`tile<ptr>`) - The scalar base pointer to a portion of global memory.
- **dynamicShape** (`Variadic<tile<any>>`) - The array of values representing the shape of the view, may be fully dynamic.
- **dynamicStrides** (`Variadic<tile<any>>`) - The array of values representing the strides of the view, may be fully dynamic.

### Results

- **result** (`tensor_view`) - The constructed tensor_view.

### Description

The `make_tensor_view` operation constructs a `tensor_view` from a global memory pointer, a dynamic shape and dynamic strides. See Tensor View for more details. The constructor supports taking dynamic arrays for shapes and strides as part of the constructor enabling workloads to take global memory tensors of dynamic shape and strides.

If these arguments are static they will be statically reflected in the type of the resulting `tensor_view`, if they are dynamic they will appear as `?` in the type. See below for concrete examples.

If shapes or strides are larger than the indexBitwidth of the `tensor_view`, behavior is undefined on the creation of the `tensor_view`.

### Constraints

The operation must encode variadic operand segment sizes in attributes. The operation is pure and does not perform any memory side effects.

### Examples

```
// tensor_view to a scalar tile of f32
%a0 = make_tensor_view %base, shape = [], strides = [] : tensor_view<f32>

// tensor_view to a tile of static shape and strides
%a1 = make_tensor_view %base, shape = [32, 32], strides = [32, 1] : tensor_view<32x32xf32, strides = [32, 1]>

%sh0 = constant dense<32> : tile<i32>
%sh1 = constant dense<32> : tile<i32>
%st0 = constant dense<32> : tile<i32>
%st1 = constant dense<1> : tile<i32>

// tensor_view to a tile with partially dynamic shape and strides
// all dynamic values must be of the same type, here tile<i32>
%a2 = make_tensor_view %base, shape = [%sh0, %sh1], strides = [%st0, %st1] : tile<i32> -> tensor_view<?x?xf32, strides = [?, ?]>
```

See `cuda_tile.make_tensor_view_0` for the full example listing.

## 8.11.6. cuda_tile.store_view_tko

Stores a tile into a tile view

```
cuda_tile.store_view_tko %memory_ordering_semantics %memory_scope %tile %view %index %token %optimization_hints
```

### Parameters

- **memory_ordering_semantics** (`MemoryOrderingSemantics`) - The memory scope for the store operation.
- **memory_scope** (`MemoryScope`) - The memory scope for the store operation.
- **tile** (`tile`) - The tile to store.
- **view** (`view_type`) - The view to store the tile to.
- **index** (`Variadic<tile<any>>`) - The indices of the desired target tile within the view.
- **token** (`token`) - The optional token for operation ordering.
- **optimization_hints** (`OptimizationHints`) - Optimization hints for operation.

### Results

- **result_token** (`token`) - The result token for synchronization.

### Description

The `store_view_tko` operation stores a tile to a view indexing into a tile view. A view is mapping from view-space indices to a particular element in the view, each view type has a defined mapping from view-space indices to tiles produced from elements of the view. For example, the Partition View partitions a Tensor View into a grid of equally sized tiles. The view indices one of the partitioned tiles in the grid.

For a given view the rank of the indices must match the rank of the view’s index space. The space of valid indices depends on which view is passed to the operation. For example the index space of a Partition View is equal to the rank of the partitioned tiles. The index space of the view is computed a function of the requested tile size and the shape of the view.

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required. For example, `weak` ordering allows the compiler to assume that there are no concurrent accesses to any accessed location. For more information, refer to the memory model section of the specification.

- **weak** - No concurrent accesses to the source/destination location.
- **relaxed** - There may be concurrent access to the location, but this access does not establish a happens-before relationship.
- **release** - There may be concurrent access to the location. If this release is observed with an acquire operation, then happens before is established.
  - *Note: The following variants are not supported by this operation: acquire, acq_rel.*

The `memory_scope` attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

- **tl_blk** - There may be concurrent accesses from within the same tile block.
- **device** - There may be concurrent accesses from within the same device (i.e., GPU).
- **sys** - There may be concurrent accesses from anywhere within the system (i.e., all devices).

The `optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries. The hints are specified for each architecture (e.g., `sm_100`, `sm_120`) and for each architecture the user can specify specific hints for each operation.

- **num_cta_in_cga** - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for `cuda_tile.entry`.
- **allow_tma** - suggest whether to use TMA for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.
- **latency** - latency hint for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`.

For example they can be annotated as:

```
optimization_hints =< sm_100 = { num_cta_in_cga = 8 }, sm_120 = { num_cta_in_cga = 16 } >
```

### Constraints

The operation must encode variadic operand segment sizes in attributes. The operation’s result type may be inferred from its operands and attributes.

### Examples

```
%tensor_view = make_tensor_view %ptr, shape = [8192, 128], strides = [128, 1] : tensor_view<8192x128xf32, strides = [128, 1]>

// This example uses the PartitionView on a 8192x128xf32 tensor_view,
// dividing the tensor_view in tiles of 64x64.
%view = make_partition_view %tensor_view : partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>>

%c0 = constant dense<0> : tile<i32>
%c1 = constant dense<1> : tile<i32>
%tile = constant dense<0.0> : tile<64x64xf32>

// Store a tile at index (0, 0) in the view's index space.
// For this TilePartitionView, this is the rectangular tile such that
// X=[0,64) and Y=[0,64), in the coordinates of tiles.
%res_token0 = store_view_tko weak %tile, %view [%c0, %c0] : tile<64x64xf32>, partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> token

// Store a tile at index (0, 1) in the view's index space.
// For this PartitionView, this is the rectangular tile such that
// X=[0,64) and Y=[64,128), in the coordinates of tiles.
%res_token1 = store_view_tko weak %tile, %view [%c0, %c1] : tile<64x64xf32>, partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> token

// Same example as above but with input token.
%token = make_token : token
%res_token2 = store_view_tko weak %tile, %view [%c0, %c1] token = %token : tile<64x64xf32>, partition_view<tile = (64x64), tensor_view<8192x128xf32, strides = [128, 1]>> -> token
```

See `cuda_tile.store_view_tko_0` for the full example listing.