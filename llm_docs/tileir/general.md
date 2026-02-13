# 8. Operations — Tile IR

## 8.1. Meta Types

Operations have arguments which are Tile IR values with Tile IR types but many operations have immediate or static arguments which correspond to attributes in the MLIR dialect. These meta types are not representable in the Tile IR type system but are used to construct Tile IR programs and only present at compile time. Operations in the specification are described abstractly in both the Tile IR IR and bytecode independent of the MLIR or bytecode encoding. For each of these types we provide a definition of them below and link to them from each operation.

**Note** The convention is that the meta types are capitalized and Tile IR types are snake cased. The convention is that the meta types are capitalized and the native Tile IR types are camel cased are snake cased.

### 8.1.1. Symbol

Symbol a symbol in the program, begins with @ and uniquely identifies a symbol in the program.

### 8.1.2. Flag

Flag a boolean value that can be used to control the behavior of an operation.

### 8.1.3. Token

Token represents a memory ordering token that can be used to control the ordering of memory operations.

### 8.1.4. Variadic

Variadic represents an argument which can accept a statically sized, but variable, number of arguments.

### 8.1.5. Any

Any represents a value of any valid Tile IR type.

### 8.1.6. Name

Name represents a name in the program, begins with # and uniquely identifies a name in the program.

### 8.1.7. Type

Type represents a Tile IR type and are attached as attributes to operations which define IR items.

### 8.1.8. Array

Array represents a statically sized array of values that can be passed to attributes.

### 8.1.9. String

String represents a string value that can be passed to attributes.

### 8.1.10. bool

bool represents a boolean value that can be passed to attributes.

### 8.1.11. DenseConstant

DenseConstant represents a dense constant value that can be passed to attributes.

### 8.1.12. view_type

view_type represents a type which implements the view interface, currently this is only implemented by partition_view but will have new implementers in future releases.

## 8.2. Operation Design Considerations

The design of Tile IR has a set of design considerations that apply to all operations in the dialect this section introduces some of the common design considerations that apply to all operations, or to classes of operations generically.

### 8.2.1. Explicit Broadcast

There are no implicit broadcast performed by operations in the Tile IR dialect all operations that require operands of the same shape must be explicitly broadcasted. For example to use the cuda_tile.offset operation to add an offset tile to a pointer, the pointer and offset must be reshaped or broadcasted to have the same shape using the cuda_tile.reshape or cuda_tile.broadcast operations.

### 8.2.2. Distinct Floating-Point and Integer Operations

Numeric ooerations are split across integer and floating-point types due to differences in flags such as rounding modes, NaN handling, and fast math. For example, the cuda_tile.addf operation supports a rounding attribute, but the addi operation does not.

### 8.2.3. Explicit Overflow Annotations

Some operations such as cuda_tile.addi support an explicit overflow annotation that expresses the expected overflow behavior of the operation. These attributes serve as assumptions that an implementation may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution. We recommend that generators of Tile IR programs utilize these annotations to help the implementation reason about the overflow behavior of the operation, enabling extra optimization opportunities.