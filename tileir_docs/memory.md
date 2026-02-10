## 8.6. Memory

Tile IR contains a set of memory operations which enable loading, storing, and manipulating memory. There are a few families of memory operations in Tile IR:

- Tile of pointer based memory operations such as `cuda_tile.load_ptr_tko` and `cuda_tile.store_ptr_tko` which load and store tiles from and to global memory.
- View based memory operations such as `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko` which load and store tiles from and to views.
- Atomic memory operations such as `cuda_tile.atomic_rmw_tko` and `cuda_tile.atomic_cas_tko` which perform atomic operations on global memory.

Currently all memory operations are token-ordered; the ordering between any pair of memory operations is undefined unless connected by tokens. For more discussion on token-ordered operations see section-memory_model.

**Warning** Reading or writing of bound of any allocation is undefined behavior. Examples of out of bounds access are:

* Pointer memory operations to tiles containing elements outside the allocation, for example offseting passed the end of the allocation.
* Associating an invalid layout with a base pointer, that describes a striding or shape that over runs the allocation and then indexing into the view.
* Indexing into a view with indices that are out of bounds.

**Note** The rules of what consititues out of bounds is modified when using padded views or masking, see Type System for more details on specific types.

### 8.6.1. cuda_tile.join_tokens

Product a new token which depends on the input tokens `cuda_tile.join_tokens % tokens`

#### Parameters
- **tokens** (Variadic<token>) - The input tokens to join. 13.1

#### Results
- **result** (token) - The joined token. 13.1

#### Description
The join_tokens operation produces a fresh token which depends on all input tokens. Token-ordered operations which consume the new token will then be ordered with respect to all joined tokens.

#### Constraints
The operation is conditionally speculatablebased on the specific operands and attributes. The operation may be speculatively executed without side effects. The operation is pure and does not perform any memory side effects. The operation’s result type may be inferred from its operands and attributes.

### 8.6.2. cuda_tile.load_ptr_tko

Load and gather data from global memory using a pointer tile without ordering guarantees `cuda_tile.load_ptr_tko % memory_ordering_semantics % memory_scope % source % mask % paddingValue % token % optimization_hints`

#### Parameters
- **memory_ordering_semantics** (MemoryOrderingSemantics) - The memory ordering semantics for the load operation. 13.1
- **memory_scope** (MemoryScope) - The memory scope for the atomic operation. 13.1
- **source** (ptr) - The source tile of pointers. 13.1
- **mask** (tile<i1>) - The mask for the load operation. 13.1
- **paddingValue** (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The padding value for the load operation. 13.1
- **token** (token) - The token for the load operation. 13.1
- **optimization_hints** (OptimizationHints) - Optimization hints for operation 13.1

#### Results
- **result** (tile) - The result of the load operation. 13.1
- **result_token** (token) - The result token of the load operation. 13.1

#### Description
This load OP performs a gather operation by loading a tile of data from global memory into a result tile based on a tile of pointers provided by the source operand. The source operand is a tile of pointers, which specifies the memory locations from which the data is gathered. The operation loads this data and returns it as the result tile. When loading i1 values, each value is loaded from a full byte in memory. Any nonzero byte is canonicalized to 0x01, and zero bytes become 0x00. Optionally, a mask operand can be provided to control the gathering of elements. If present, only the elements specified by the mask are loaded. The shape of the mask must match the shape of the result. When mask is present one paddingValue can be optionally present as well. The paddingValue must have the same shape of the source tile. If it is not present, the value of masked elements are undefined. Token-ordered operations are not constrained by program order. The compiler may reorder them (i.e. place them earlier or later in program order) unless further constrained by tokens. The memory_ordering_semantics attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required. For example, weak ordering allows the compiler to assume that there are no concurrent accesses to any accessed location. For more information, refer to the memory model section of the specification.

weak - No concurrent accesses to the source/destination location.
relaxed - There may be concurrent access to the location, but this access does not establish a happens-before relationship.
acquire - There may be concurrent accesses to the location. If this acquire observes a release operation, then happens before is established.
Note: The following variants are not supported by this operation: release, acq_rel.

The memory_scope attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

tl_blk - There may be concurrent accesses from within the same tile block.
device - There may be concurrent accesses from within the same device (i.e., GPU).
sys - There may be concurrent accesses from anywhere within the system (i.e., all devices).

The optimization_hints attribute provides architecture-specific compiler hints in the form of nested dictionaries. The hints are specified for each architecture (e.g., sm_100, sm_120) and for each architecture the user can specify specific hints for each operation.

num_cta_in_cga - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for cuda_tile.entry.
allow_tma - suggest whether to use TMA for cuda_tile.load_view_tko and cuda_tile.store_view_tko.
latency - latency hint for cuda_tile.load_view_tko and cuda_tile.store_view_tko.
For example they can be annotated as: `optimization_hints =< sm_100 = { num_cta_in_cga = 8 }, sm_120 = { num_cta_in_cga = 16 } >`

#### Constraints
The operation must encode variadic operand segment sizes in attributes. source type is expected a pointer type of result type shape of 'mask' must match the shape of 'source' type of 'paddingValue' must match the type of 'result'

#### Examples
```
%mask = constant dense < 1 > : tile < i1 > %padding = constant dense < 0.0 > : tile < f32 > // Load without token. %result0 , %res_token0 = load_ptr_tko weak %ptr , %mask , %padding : tile < ptr < f32 >> , tile < i1 > , tile < f32 > -> tile < f32 > , token // Load with token. %token0 = make_token : token %result1 , %res_token1 = load_ptr_tko weak %ptr , %mask , %padding token =% token0 : tile < ptr < f32 >> , tile < i1 > , tile < f32 > -> tile < f32 > , token return See cuda_tile.load_ptr_tko_0 for the full example listing.
```

### 8.6.3. cuda_tile.make_token

Create a fresh token with no prior dependencies

#### Parameters
No parameters.

#### Results
- **result** (token) - A fresh token with no prior dependencies. 13.1

#### Description
The make_token operation creates a fresh token with no prior dependencies.

#### Constraints
The operation is conditionally speculatablebased on the specific operands and attributes. The operation may be speculatively executed without side effects. The operation is pure and does not perform any memory side effects. The operation’s result type may be inferred from its operands and attributes.

### 8.6.4. cuda_tile.store_ptr_tko

Store and scatter data from pointer of tile to global memory without ordering guarantees `cuda_tile.store_ptr_tko % memory_ordering_semantics % memory_scope % destination % value % mask % token % optimization_hints`

#### Parameters
- **memory_ordering_semantics** (MemoryOrderingSemantics) - The memory ordering semantics. 13.1
- **memory_scope** (MemoryScope) - The optional memory scope. 13.1
- **destination** (ptr) - The destination pointer tile. 13.1
- **value** (tile) - The value tile to store. 13.1
- **mask** (tile<i1>) - The optional mask for selective storage. 13.1
- **token** (token) - The optional token for operation ordering. 13.1
- **optimization_hints** (OptimizationHints) - Optimization hints for operation 13.1

#### Results
- **result_token** (token) - The result token for synchronization. 13.1

#### Description
The store operation performs a scatter by storing a tile of data from a tile into global memory. The destination operand is a tile of pointers indicating the global memory locations where data from the value tile will be stored. When storing i1 values, each value occupies a full byte in memory. Any nonzero byte is canonicalized to 0x01, and zero bytes become 0x00. Additionally, the operation supports an optional mask operand, which allows selective scattering of elements. If provided, only the elements specified by the mask are stored. The shape of the mask must align with the shape of the value tile. The memory_ordering_semantics attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required. For example, weak ordering allows the compiler to assume that there are no concurrent accesses to any accessed location. For more information, refer to the memory model section of the specification.

weak - No concurrent accesses to the source/destination location.
relaxed - There may be concurrent access to the location, but this access does not establish a happens-before relationship.
release - There may be concurrent access to the location. If this release is observed with an acquire operation, then happens before is established.
Note: The following variants are not supported by this operation: acquire, acq_rel.

The memory_scope attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

tl_blk - There may be concurrent accesses from within the same tile block.
device - There may be concurrent accesses from within the same device (i.e., GPU).
sys - There may be concurrent accesses from anywhere within the system (i.e., all devices).

The optimization_hints attribute provides architecture-specific compiler hints in the form of nested dictionaries. The hints are specified for each architecture (e.g., sm_100, sm_120) and for each architecture the user can specify specific hints for each operation.

num_cta_in_cga - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for cuda_tile.entry.
allow_tma - suggest whether to use TMA for cuda_tile.load_view_tko and cuda_tile.store_view_tko.
latency - latency hint for cuda_tile.load_view_tko and cuda_tile.store_view_tko.
For example they can be annotated as: `optimization_hints =< sm_100 = { num_cta_in_cga = 8 }, sm_120 = { num_cta_in_cga = 16 } >`

#### Constraints
The operation must encode variadic operand segment sizes in attributes. destination type is expected a pointer type of value type shape of 'destination' must match the shape of 'mask' The operation’s result type may be inferred from its operands and attributes.