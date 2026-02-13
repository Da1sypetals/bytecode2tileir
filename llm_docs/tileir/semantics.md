# 6. Semantics — Tile IR

This section provides a written English presentation of the operational semantics of Tile IR. These semantics are intended to provide an understanding of Tile IR for: 1) those interested in generating Tile IR as a code generation target, or 2) those interested in reading Tile IR programs produced by others. It does not attempt to formalize every possible behavior that might be admitted by an axiomatic formulation or a small step operational semantics. For understanding an even more informal presentation of the language and its core concepts see the Programming Model section. We first introduce the abstract machine state and language definitions before describing semantics of individual kernels and programs. We then discuss the semantics of broad classes of operations as well, for more detailed descriptions of individual operations and their behavior see Operations for a complete listing.

## 6.1. The Abstract Machine

The Tile IR abstract machine state \(\mathcal{S}\) is a tuple consisting of the following components, each explained below:

- A well-formed module \(\mathcal{Mod}\) which stores one of more items, discussed below in Modules.
- A grid of tile blocks \(\mathcal{TB}\) (or logical tile threads) each representing a single tile-kernel instance.
- A per-tile-block infinite register file, \(R\), that maps named ”registers” to values.
- A global memory store, \(M\), that maps addresses to scalar values.
- A set of pending memory accesses, \(P\), that make progress asynchronously to the execution of Tile IR operations.

## 6.2. Modules

A program in Tile IR is represented as a module. A module is a single translation unit which contains zero or more items. An item may either be a:

- a global variable definition
- a tile kernel
- a tile function

For those familiar with CUDA, a tile kernel is the global entry point for a tile program, much like a kernel is in CUDA C++ or PTX. Tile functions represent device side functions that can be called from the tile kernel currently with some restrictions.

### 6.2.1. Global Variable

A global is a named global variable that is stored in global device memory and accessible to all tile blocks. Global variables are declared using the `cuda_tile.global` operation. A global variable must be initialized upon declaration and will be initialized exactly once. A global variable must contain a value of Tile Type. A global variable can be modified by using the `cuda_tile.get_global` operation to obtain a pointer which can be used to read and write to the global variable.

### 6.2.2. Tile Kernel

```
entry @tile_func ( %A0: T0 , ..., %AN: TN ) {
  %0 = op %P0 , %P1 , ... %PN -> R0 ...
  return
}
```

The basic unit of execution in Tile IR is the tile kernel. A tile kernel is a tile function that acts as the entry point of a tile program. A tile kernel represents a function parameterized by a set of grid coordinates. At kernel runtime, each unique grid coordinate is available to each kernel instance (tile block). A tile kernel can query its grid coordinates via `cuda_tile.get_tile_block_id` and the coordinates can be one-, two-, or three-dimensional depending on the grid the kernel is launched with. A kernel may also query the total the total number of tile blocks along each dimension via `cuda_tile.get_num_tile_blocks`.

A tile kernel is a tile function with additional restrictions:

- can only have parameters with scalar (i.e., 0-rank) tensor types
- requires all input tensors to be provided as scalar pointers (i.e `tile<ptr<E>>`)
- produces no return value
- the kernel is only executed for its effect on global device memory

A tile kernel is otherwise a tile function and all properties of tile functions also apply to tile kernels.

### 6.2.3. Tile Function

A tile function consists of a name, a list of formal parameters, a return type, and a body. A tile function’s body contains a single threaded tile program (referred to as tile block) parameterized by formal parameters. A tile function has \(N\) formal parameters and produces M return values. The type of the parameters can be one of the valid types described in Type System.

> **Note**
>
> Currently defining non-kernel tile functions is disabled with support planned for a future release.

### 6.2.4. Function Bodies

A function body consists of a sequence of statements that are in static-single-assignment (SSA) form. Each statement assigns the result of a single operation to a set of unique result variables. All operations in Tile IR are represented uniformly in this way, including control flow and memory operations.

### 6.2.5. Well-Formedness

A well-formed module is a module that satisfies the following properties:

- The module contains at least one item.
- Each item is uniquely named within the module.
- Each Tile Kernel and Tile Function has a body that is a sequence of statements in valid static-single-assignment (SSA) form.
- The program type checks according to the rules specified in Type System and the operator type signatures are specified in Operations.

Program well-formedness is required as a pre-condition and post-condition of both optimizations and operational semantic rules.

## 6.3. Values

Tile IR has a small set of types as described in Type System but we only have two types of values.

- Pointers, which represent a memory address.
- Tiles, or an N-dimensional array of scalars.
- Views, which represent a structured view of memory.

### 6.3.1. Pointers

A pointer is a 64-bit integer memory address that references a location in global device memory. Pointers are typed as `ptr<E>` where E is the type of the memory location it references. Pointers are required to be aligned to the size of the underlying datatype they point to see Element Type Encoding for specific encodings and information about allocation layout.

A pointer is a memory address that points to a location in global memory.

#### Data Layout

Allocations pointed to by input pointer values, and by extension views (see below), must conform to the specified data layout. We expect that the allocation pointed to by `ptr<E>` is a sized contiguous allocation of scalar values of element type E. There is no padding between elements of the allocation, and we expect that for an allocation of size N will be equivalent to N * sizeof(E) bytes.

The size and encoding of a element is determined by its type E and is defined in the table below. As an aside the datatype encoding is compatible with DLPack a standard adopted by most deep learning frameworks and array libraries. We provide the equivalent PyTorch and NumPy encodings for each datatype. For NumPy low-precision types we provide the equivalent in terms of the ml_dtypes library a standard collection of low-precision NumPy data types.

> **Warning**
>
> Tile IR layouts are currently restricted to be contiguous for sub-byte types.

> **Note**
>
> Allocations are the only values where memory layout is specified in Tile IR.

### 6.3.2. Tiles

A tile is a immutable N-dimensional array of scalars characterized by:

- Its rank (number of dimensions)
- Its shape (extent along each dimension)
- Its primitive element type

These properties are part of both the tile’s type and the size and shape of the value.

A tile may have any any non-negative rank, where:

- Rank-0 tiles represent scalar values.
- Rank-1 tiles represent vectors.
- Rank-2 tiles represent matrices.
- And Rank-N tiles represent higher-order N-d arrays.

For a given tile of type `tile<NxKxE>`, where the tile has shape (N, K) with elements of type E results in a value of size \(N \times K \times \text{size}(E)\), containing \(N * K\) individual elements. A tile value of this type is abstractly represented as a tuple of an array with \(N * K\) of elements of type \(E\), and an opaque layout which provides a mapping from the index space of elements to a linear index.

> **Note**
>
> The physical layout and memory representation of a given tile is not visible to the program and is not specified by the language semantics. The compiler will choose to represent a tile in memory in a way that is most efficient for the target architecture, and specific program and tiles sizes.

### 6.3.3. Views

Tile IR provides a set of view types as described in Tensor View. Views provide a structured views of memory by enriching a pointer with additional data. Views have their own set of memory operations `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko` which vary in behavior based on the view type and make use of the additional metadata. Due to the face that views are not a singular type, but a family of types each concrete view type has its own value representation.

Our first-order view type is Tensor View. A tensor view value is logically a tuple of \((ptr, \vec{shape}, \vec{strides}, \vec{dimgroups})\), and load and store operate on the entire tensor view. The index space of this view is rank-0 (i.e., there is only a single element in the index space).

Our primary second-order (or “sub-view”) view type is Partition View. The partition view type Partition View is a subview type which represents a tensor view partitioned into tiles. A partition view value is logically a tuple of \((tensor\_view, \vec{tile\_size})\), where load and store operate over tile values of the given size. The index space of this view is:

\[
indexSpace[i] = \text{tensorViewShape}[\text{dimGroups}[i]] / \text{tileSize}[i]
\]

For example a tensor view with shape \((1024, 1024)\) may be partitioned into a grid of \(64 \times 32\) tiles, resulting in a partition view with shape \((16, 32)\).

See Type System for more details on the different types of views.

> **Note**
>
> As with tiles, the physical layout and memory representation of a view is not visible to the program and is not specified by the language semantics.

> **Warning**
>
> Reading or writing of bounds of any allocation is undefined behavior and bounds checking must be performed by the programmer if desired.

## 6.4. Tile Grid

During execution, the abstract machine instantiates a grid of tile blocks. A tile grid is grid of tile blocks arranged in a 1-, 2-, or 3-dimensional array. Each position of the grid corresponds to a single independent tile kernel instance. The abstract machine stores a sequence of tile blocks, \(TB\), that are indexed by a grid of coordinates, \(\vec{g}\). Each tile block is assigned unique tile block id based on the grid size and the tile block’s position within the grid. The bijective mapping between 1-, 2-, or 3-dimensional grid coordinates and flattened tile block ids is computed as follows:

\[
\begin{split}
\begin{aligned}
\text{tile\_grid}(\vec{i}) &= \begin{cases}
i_0 & \text{where grid} = (x) \\
i_0 \cdot x + i_1 & \text{where grid} = (x,y) \\
i_0 \cdot x + i_1 \cdot y + i_2 & \text{where grid} = (x,y,z)
\end{cases} \\
&\text{where } \vec{i} = (i_0, \ldots, i_n) \\
&\text{and } 0 \leq i_k < \text{grid}_k \text{ for all } k \leq 3
\end{aligned}
\end{split}
\]

## 6.5. Register File

The register file, \(R\), maps named registers to values. Each assignment (i.e., SSA variable) in the tile function’s body is assigned to a unique register which eventually holds the value of the operation’s result. Registers are local to a tile block and are not visible to other tile blocks. As stated previously, the memory representation of values in registers are not visible to the program and is not specified by the language semantics. Values will only be fetched or persisted to global memory by memory operations (see Memory).

The register file is indexed by the tile block’s coordinates, \(\vec{g}\), and the register’s name, \(r\) and produces a value \(v\).

\[
S.R(\vec{g}, r) = v
\]

## 6.6. Global Memory

The global memory, \(M\), is a mapping from addresses to scalar values. The global memory is used to store the values of the tile block’s global variables. The heap is abstractly modeled as map from addresses to scalar values, not tile values. This distinction is essential to describe the memory effect of tile operations as a sequence of individual scalar memory operations. A fine-grained model enables both reasoning about aggregate operations granularly as well as a straight forward denotation into the existing PTX memory model.

> **Note**
>
> Global memory is the same global device memory that is used by CUDA programs and described in the PTX Specification. The specification elaborates the intricacies of the Tile IR memory model in section-memory_model.

## 6.7. Tile Block

A tile block is a single thread of execution that is assigned a unique coordinate in the tile grid. Abstractly its state consists of:

- The tile kernel under execution.
- A register file, \(R\), that maps named registers to values.
- A statement under evaluation representing by an integer index into the sequence of SSA statements in the tile function’s body.

## 6.8. Execution Semantics

Tile IR program execution starts with a kernel launch. The kernel launch API is uniform for all CUDA kernels and is specified in the CUDA Runtime API.

### 6.8.1. Initialization

A launch of tile kernel initializes the abstract machine with:

- The module representing the complete program.
- A grid of tile blocks where each tile block is instantiated using the same tile kernel, begins at statement 0, assigned a unique grid coordinate, and assigned a unique empty register file.
- A reference to global memory, where its state is the state of global memory prior to the kernel launch.
- The set of pending memory operations is initialize to be empty.

### 6.8.2. Forward Progress

Execution proceeds with unspecified scheduling of tile blocks. Each tile block will be executed in some order which is non-deterministic and not specified by the language semantics. We guarantee forward progress of the execution that is all tile blocks will be guaranteed to eventually be scheduled for execution. It is possible that all tile blocks run completely in parallel, completely serially, or anything in between.

### 6.8.3. Tile Block Execution

Execution of a single tile block is isolated from other tile blocks. Tile blocks can only observe effects of other blocks via global memory which can be used to implement forms of cooperation or communication.

Function bodies are a series of static-single-assignment (SSA) statements which assign the result of each operations to a unique variable. Each variable is mapped to a register in the abstract machine’s register file. A function body is executes statements sequentially, in order. The compiler is free to reorder statements as long as there is no effect on the program visible effects or violated program semantics. For more detailed example programs and explanations of their execution see Programming Model or Appendix.

The one unique semantic of Tile IR is the partitioning of memory operations into program ordered and token ordered operations. All memory operations produce their result values immediately but the order in which order these operations effect memory is more subtle.

For program ordered operations, the order between any pair memory operations acting on the same address is defined by the operation’s position in program. Intuitively the effect of all prior memory operations on the same address will be visible to all subsequent memory operations on the same address.

In contrast, the order between any pair of token ordered operations is undefined, and has no relation to program order. The order of a pair of any two token ordered operations (\(A\) and \(B\)) is only defined if established by a direct or transitive relationship between \(A\)’s output token and \(B\)’s input token.

This choice importantly allows a producer of Tile IR to induce different memory ordering semantics by inserting the appropriate memory ordering tokens. For example starting with a single fresh token depending on by the first operation with the result token of the first operation being depended upon by the second operation and so on threading the tokens through each operation in program order. Token threading like this establishes a ordering of the memory operations which is consistent with the same program with each token ordered operations being replaced by a program ordered memory operation.

For a detailed discussion of the memory model and memory operations see section-memory_model.

### 6.8.4. Termination

A tile block will terminate when the tile block’s function body reaches the final statement. Tile kernels must terminate with a return operation `cuda_tile.return` which signals the end of the execution.

https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#execution-semantics