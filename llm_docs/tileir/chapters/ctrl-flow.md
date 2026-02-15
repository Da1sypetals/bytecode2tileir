
## 8.5. Control Flow

**Tile IR** contains a standard set of control flow operations that enable conditionals, and loops.

The operations are designed in the style of the MLIR Control Flow dialect.

A notable difference is that we allow the nesting of control flow operations for example a cuda_tile.if may appear inside a cuda_tile.loop or cuda_tile.for.

The main control structures are:

- cuda_tile.if which implements conditional branching.
- cuda_tile.loop which implements a loop with arbitrary exit conditions.
- cuda_tile.for which implements a range-based loop with a fixed number of iterations.

These operations and their supporting operations are described in the following section.

### 8.5.1. cuda_tile.assert

*Terminate kernel execution with an error message if condition is false-y*

```
cuda_tile.assert %condition %message
```

#### Parameters

- **condition** (tile<i1>) - The condition tile to check.
- **message** (String) - The error message to display if assertion fails.

#### Results

No results.

#### Description

The `assert` operation takes as `condition` a tile of `i1` values. For each value that is `0`, it prints the given error message, along with the index of the value within the tile.

If at least one value is `0`, an error is signalled to the host side. The kernel, including the tile block that failed the assertion, may keep running.

Assertions are for debugging purposes. They can affect performance and it is therefore recommended to remove them in production code.

#### Constraints

No constraints.

#### Examples

```mlir
assert %arg0, "assertion failed" : tile<i1>
```

See cuda_tile.assert_0 for the full example listing.

### 8.5.2. cuda_tile.break

*Break from loop*

```
cuda_tile.break %operands
```

#### Parameters

- **operands** (Variadic<Any>) - The operands to yield to the parent loop upon termination.

#### Results

No results.

#### Description

The `break` operation is a terminator operation of a cuda_tile.loop.

It may yield any number of `$operands` to the parent loop upon termination. The number of values yielded and the execution semantics of how they are yielded are determined by the parent loop.

The `break` operation always returns control to the innermost enclosing loop operation, even when it is nested within other control constructs such as `if` or additional loops.

#### Constraints

- The operation must terminate its parent basic block.

#### Examples

```mlir
// Break from the body of a loop.
loop {
    break
}

// Break from an if nested within the loop.
loop  {
    %condition = constant dense<1> : tile<i1>
    if %condition  {
        break
    }
    // ...
}

%initValue0 = constant dense<0.0> : tile<f32>
// Break from an if nested within the loop, while yielding values.
%results = loop iter_values(%var0 = %initValue0): tile<f32> -> tile<f32> {
    %condition = constant dense<1> : tile<i1>
    if %condition  {
        // ...
        yield
    } else {
        // %if.loopValue0 = ...
        %loopValue0 = constant dense<1.0> : tile<f32>
        break %loopValue0 : tile<f32>
    }
    %loopValue1 = constant dense<1.0> : tile<f32>
    continue %loopValue1 : tile<f32>
}
```

See cuda_tile.break_0 for the full example listing.

### 8.5.3. cuda_tile.continue

*Continue to next loop iteration*

```
cuda_tile.continue %operands
```

#### Parameters

- **operands** (Variadic<Any>) - The values to yield to the parent loop.

#### Results

No results.

#### Description

The `continue` operation represents a block terminator that returns control to a loop operation, such as cuda_tile.for and cuda_tile.loop. The operation may yield any number of `$operands` to the parent loop upon termination.

The requirements and semantics of the `continue` operation are defined by the parent loop operation, see the loop operation's description for particular semantics.

The `continue` operation always returns control to the innermost enclosing loop operation, even when it is nested within other control constructs such as `if` or additional loops.

#### Constraints

- The operation must terminate its parent basic block.

#### Examples

```mlir
  %lowerBound = constant dense<0> : tile<i32>
  %upperBound = constant dense<10> : tile<i32>
  %step = constant dense<1> : tile<i32>
  %condition = constant dense<1> : tile<i1>
  // Continue from the body of a loop.
  for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
      continue
  }

  // Continue from an if nested within the loop.
  for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
      if %condition  {
          continue
      }
      // ...
  }

// Continue from an if nested within the loop, while yielding values.
%initVar0 = constant dense<0.0> : tile<f32>
%results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
          iter_values(%var0 = %initVar0) -> (tile<f32>)
  {
      if %condition {
          // ...
          yield
      } else {
          %loopValue0 = constant dense<1.0> : tile<f32>
          continue %loopValue0 : tile<f32>
      }
      %loopValue1 = constant dense<1.0> : tile<f32>
      continue %loopValue1 : tile<f32>
  }
```

See cuda_tile.continue_0 for the full example listing.

### 8.5.4. cuda_tile.for

*For loop over integer range*

```
cuda_tile.for %lowerBound %upperBound %step %initValues
```

#### Parameters

- **lowerBound** (tile<any>) - The lower bound of the loop.
- **upperBound** (tile<any>) - The upper bound of the loop.
- **step** (tile<any>) - The step of the loop.
- **initValues** (Variadic<Any>) - The initial values for the loop carried variables.

#### Results

- **resultValues** (Variadic<Any>) - The values of the loop-carried variables after loop termination.

#### Description

The `for` operation is a structured range-based sequential loop.

The loop operation consists of (1) a range formed by `lowerBound`, `upperBound`, and `step`, (2) a set of loop-carried values which are initialized by `initValues` and updated by each iteration of the loop, and (3) a region which represents the loop body.

The iteration space is defined by the interval `[lowerBound, upperBound)` with each value separated by `step`.

`lowerBound`, `upperBound`, and `step` must be of the same type. `lowerBound` and `upperBound` specify a half-open (or exclusive) range: the range includes the `lowerBound` but does not include the `upperBound`. `step` must be positive but the bounds may be negative or zero.

The first iteration of the loop receives the induction variable initialized to the value of `lowerBound` and the loop-carried values initialized to the values of `initValues`.

The loop body is executed for each value in the range, receiving an integer induction variable incremented by `step` on each iteration and the loop-carried values which correspond to the loop-carried values yielded by the previous loop iteration.

The loop terminates when the induction variable is greater than or equal to `upperBound`. By default, signed comparison is used between the upperBound and the induction variable. To use unsigned comparison instead, specify the optional `unsigned` unit attribute.

The body of the loop must be terminated by a cuda_tile.continue that yields the next iteration's value for each loop carried variable.

The for operation produces one return value for each loop carried variable. The type of the i-th return value is that of the i-th loop carried variable and its value is the final value of the i-th loop carried variable.

> **Warning**
> - Loop carried variables can not be a tensor_view or view type.
> - `for` operations cannot terminate early and must end in a cuda_tile.continue.

#### Constraints

- The operation must define scope when stack allocations are freed automatically.
- `lowerBound`, `upperBound` and `step` must have the same shape and element type (tile<any>).
- `initValues` and `resultValues` must have the same shape and element type (Variadic<Any>).
- The operation must provide custom parsing and printing methods.
- The operation only has an effect if and only if it the region's operation have an effect.
- Each provided region must contain exactly one block.

#### Examples

```mlir
%lowerBound = constant dense<0> : tile<i32>
%upperBound = constant dense<10> : tile<i32>
%step = constant dense<1> : tile<i32>

// A simple loop iterating over an i32 range.
for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
    continue
}

%initVal0 = constant dense<0.0> : tile<f32>
// A similar loop to the above, but with a loop carried value, val0.
%results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
                    iter_values(%val00 = %initVal0) -> (tile<f32>) {
  %loopVal0 = constant dense<1.0> : tile<f32>
  continue %loopVal0 : tile<f32>
}
```

See cuda_tile.for_0 for the full example listing.

### 8.5.5. cuda_tile.if

*Conditional execution*

```
cuda_tile.if %condition
```

#### Parameters

- **condition** (tile<i1>) - The condition of the if operation.

#### Results

- **results** (Variadic<Any>) - The results of the if operation.

#### Description

The `if` operation represents an if-then-else construct.

The `if` operation consists of (1) a control operand which is a `tile<i1>` value, (2) a true branch `thenRegion` and (3) an optional false branch `elseRegion`.

The `if` operation may produce results by yielding values in each branch using cuda_tile.yield.

If yielding value(s) the types of yielded values must match and the result result type of the `if` operation will be the same as the yielded values.

If yielding values the else branch is required and must also yield a value.

The values returned will be dependent on which branch is taken.

> **Warning**
> The `if` operation has a set of additional restrictions today:
> - Results of `if` must not be a tensor_view or view type.

#### Constraints

- All regions must have zero arguments.
- The operation must provide custom parsing and printing methods.
- The operation only has an effect if and only if it the region's operation have an effect.
- Each provided region must contain exactly one block.

#### Examples

```mlir
%condition = constant dense<1> : tile<i1>

// A simple if operation that conditionally executes a region.
if %condition  {
  // ...
}

// An if operation with an "else" branch.
if %condition  {
  // ...
} else {
  // ...
}

// An if operation that returns mixed types (f32,i32)
%x, %y = if %condition -> (tile<f32>, tile<i32>) {
  %x_then = constant dense<1.0> : tile<f32>
  %y_then = constant dense<2> : tile<i32>
  yield %x_then, %y_then : tile<f32>, tile<i32>
} else {
  %x_then = constant dense<1.0> : tile<f32>
  %y_then = constant dense<42> : tile<i32>
  yield %x_then, %y_then : tile<f32>, tile<i32>
}
```

See cuda_tile.if_0 for the full example listing.

### 8.5.6. cuda_tile.loop

*Loop until a break operation*

```
cuda_tile.loop %initValues
```

#### Parameters

- **initValues** (Variadic<Any>) - The initial values of the loop.

#### Results

- **resultValues** (Variadic<Any>) - The result values of the loop.

#### Description

The `loop` operation represents an, unstructured, infinite loop that executes until a cuda_tile.break is reached.

The loop consists of a (1) a set of loop-carried values which are initialized by `initValues` and updated by each iteration of the loop, and (2) a region which represents the loop body.

The loop will execute the body of the loop until a cuda_tile.break is dynamically executed.

Each control path of the loop must be terminated by:

- a cuda_tile.continue that yields the next iteration's value for each loop carried variable.
- a cuda_tile.break that terminates the loop and yields the final loop carried values.

As long as each loop iteration is terminated by one of these operations they may be combined with other control flow operations to express different control flow patterns.

The loop operation produces one return value for each loop carried variable. The type of the i-th return value is that of the i-th loop carried variable and its value is the final value of the i-th loop carried variable.

> **Warning**
> Loop operations have a set of additional restrictions today:
> - Early returns from inside loops are not supported, a code generator must first terminate the loop and then return if they wish to end the function execution entirely.
> - Loop carried variables can not be a tensor_view or view type.

#### Constraints

- The operation must define scope when stack allocations are freed automatically.
- `initValues` and `resultValues` must have the same shape and element type (Variadic<Any>).
- The operation must provide custom parsing and printing methods.
- The operation only has an effect if and only if it the region's operation have an effect.
- Each provided region must contain exactly one block.

#### Examples

```mlir
%initValue0 = constant dense<0.0> : tile<f32>
%results = loop iter_values(%var0 = %initValue0): tile<f32> -> tile<f32> {
    %condition = constant dense<1> : tile<i1>
    if %condition  {
        // ...
        yield
    } else {
        // %if.loopValue0 = ...
        %loopValue0 = constant dense<1.0> : tile<f32>
        break %loopValue0 : tile<f32>
    }
    %loopValue1 = constant dense<1.0> : tile<f32>
    continue %loopValue1 : tile<f32>
}
```

See cuda_tile.loop_0 for the full example listing.

#### Constraints

- The operation must define scope when stack allocations are freed automatically.
- The operation must provide custom parsing and printing methods.
- The operation only has an effect if and only if it the region's operation have an effect.
- Each provided region must contain exactly one block.

#### Examples

```mlir
// A simple "while-do" loop.
loop {
    %cond = constant dense<1> : tile<i1>
    if %cond {
        continue
    }
    break
}
```

See cuda_tile.loop_0 for the full example listing.

```mlir
// A simple "do-while" loop.
loop {
    //... body of the loop.

    %cond = constant dense<1> : tile<i1>
    if %cond {
        continue
    }
    break
}
```

See cuda_tile.loop_1 for the full example listing.

```mlir
%initValue0 = constant dense<0.0> : tile<f32>
// A loop that yields carried-iteration values, returning the final values.
%results = loop iter_values(%value0 = %initValue0) : tile<f32> -> tile<f32> {
    %cond = constant dense<1> : tile<i1>
    if %cond {
        %loopValue0 = constant dense<0.0> : tile<f32>
        continue %loopValue0 : tile<f32>
    }
    break %value0 : tile<f32>
}
```

See cuda_tile.loop_2 for the full example listing.

```mlir
%initValue0 = constant dense<0> : tile<i32>
// A loop that uses loop-carried values and returns a different type.
%results = loop iter_values(%value0 = %initValue0) : tile<i32> -> tile<f32> {
    %cond = constant dense<1> : tile<i1>

    if %cond {
        %newLoopValue = constant dense<0> : tile<i32>
        continue %newLoopValue : tile<i32>
    }

    %finalReturnValue = constant dense<0.0> : tile<f32>
    break %finalReturnValue : tile<f32>
}
```

See cuda_tile.loop_3 for the full example listing.

### 8.5.7. cuda_tile.return

*Return value(s) from function*

```
cuda_tile.return %operands
```

#### Parameters

- **operands** (Variadic<Any>) - The values to return.

#### Results

No results.

#### Description

The `return` operation returns control to the caller of a function.

> **Warning**
> Today the `return` operation has restricted semantics:
> - cuda_tile.entry operations do not produce return value(s) and thus `return` may be used to terminate the execution of the kernel by invoking the operation with no operands.
> - `return` can not be directly used inside of loop bodies to terminate the the execution of the kernel.

#### Constraints

- The operation must terminate its parent basic block.

#### Examples

```mlir
entry @foo() {
  %0 = constant dense<0> : tile<i32>
  %1 = constant dense<0.0> : tile<f16>
  // ...
  return
}
```

See cuda_tile.return_0 for the full example listing.

### 8.5.8. cuda_tile.yield

*Yield a value from the block*

```
cuda_tile.yield %operands
```

#### Parameters

- **operands** (Variadic<Any>) - The operands to yield to the parent operation.

#### Results

No results.

#### Description

The `yield` operation terminates a block that must yield control back to the parent operation such as `if`, `scan`, `reduce`.

The operation may yield any number of `$operands` to the parent upon termination. The number of values yielded and the execution semantics of how they are yielded are determined by the parent operation.

> **Note**
> Unlike standard MLIR control flow dialects `yield` is not used for loop control flow, see cuda_tile.break and cuda_tile.continue for loop control flow.

#### Constraints

- The operation is conditionally speculatablebased on the specific operands and attributes.
- The operation may be speculatively executed without side effects.
- The operation is pure and does not perform any memory side effects.
- The operation must terminate its parent basic block.

#### Examples

```mlir
%condition = constant dense<true> : tile<i1>
// Yield from the body of an if conditional.
if %condition  {
    yield
}

// Yield values from within an if conditional.
%x, %y = if %condition -> (tile<f32>, tile<f32>) {
    %x_then = constant dense<0.0> : tile<f32>
    %y_then = constant dense<1.0> : tile<f32>
    yield %x_then, %y_then : tile<f32>, tile<f32>
} else {
    %x_else = constant dense<2.0> : tile<f32>
    %y_else = constant dense<3.0> : tile<f32>
    yield %x_else, %y_else : tile<f32>, tile<f32>
}
```