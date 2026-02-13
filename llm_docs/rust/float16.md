# Rust f16: Use Builtin Nightly Type Instead of `half` Crate

## Summary

When working with 16-bit floating-point values in Rust **nightly**, prefer the **builtin `f16` primitive type** over the `half` crate. The native type provides better performance, full standard library integration, and zero external dependencies.

## When This Applies

- **Rust Nightly toolchain** is being used
- Target requires 16-bit IEEE 754 binary16 floats

## Feature Gate

Add to `lib.rs` or `main.rs`:

```rust
#![feature(f16)]
```

## Literal Syntax

```rust
// Suffix notation (preferred)
let x = 1.0f16;
let y = 3.14159f16;
let z = -0.5f16;

// Type annotation
let a: f16 = 1.0;
let b: f16 = 2.5;

// Underscore separator
let precise = 1.234_567f16;
```

## Constants

```rust
use std::f16::consts;

// Mathematical constants
let pi = consts::PI;          // 3.14159...
let e = consts::E;            // 2.71828...
let sqrt2 = consts::SQRT_2;

// Type limits
let max = f16::MAX;           // 6.5504e+4
let min = f16::MIN;           // -6.5504e+4
let epsilon = f16::EPSILON;   // 9.7656e-4

// Special values
let nan = f16::NAN;
let inf = f16::INFINITY;
let neg_inf = f16::NEG_INFINITY;
```

## Common Operations

```rust
#![feature(f16)]

fn main() {
    let x = 2.0f16;

    // Arithmetic
    let sum = x + 1.0f16;
    let product = x * 3.0f16;

    // Math functions
    let root = x.sqrt();
    let sine = x.sin();
    let power = x.powi(2);
    let fma = x.mul_add(2.0f16, 1.0f16);  // x * 2 + 1

    // Rounding
    let floored = x.floor();
    let ceiled = x.ceil();
    let rounded = x.round();

    // Classification
    let is_finite = x.is_finite();
    let is_nan = x.is_nan();

    // Comparison
    let clamped = x.clamp(0.0f16, 1.0f16);
    let bigger = x.max(1.5f16);
}
```

## Type Conversions

```rust
#![feature(f16)]

fn main() {
    let x = 1.5f16;

    // Lossless widening (From trait)
    let as_f32: f32 = x.into();
    let as_f64: f64 = x.into();

    // Narrowing (use `as` cast)
    let from_f32 = 1.5f32 as f16;
    let from_f64 = 1.5f64 as f16;

    // Integer conversions
    let from_u8: f16 = 42u8.into();
    let from_i8: f16 = (-5i8).into();
    let to_int = x as i32;

    // Bit manipulation
    let bits: u16 = x.to_bits();
    let from_bits = f16::from_bits(0x3C00);  // 1.0f16

    // Byte arrays
    let bytes = x.to_le_bytes();  // [u8; 2]
    let restored = f16::from_le_bytes(bytes);
}
```

## Equivalence from `half` Crate

| `half` Crate | Builtin `f16` |
|--------------|---------------|
| `half::f16::from_f32(x)` | `x as f16` |
| `half::f16::from_f64(x)` | `x as f16` |
| `x.to_f32()` | `x.into()` or `x as f32` |
| `x.to_f64()` | `x.into()` or `x as f64` |
| `half::f16::from_bits(b)` | `f16::from_bits(b)` |
| `x.to_bits()` | `x.to_bits()` |
| `half::f16::INFINITY` | `f16::INFINITY` |
| `half::f16::NAN` | `f16::NAN` |
| `half::f16::MIN_POSITIVE` | `f16::MIN_POSITIVE` |
| `half::consts::PI` | `std::f16::consts::PI` |

This is just for demonstration. DO NOT use `half` crate. use the operations for builtin `f16` type.

## Quick Reference

```rust
#![feature(f16)]
use std::f16::consts::PI;

fn main() {
    // Literals
    let x = 1.0f16;

    // Constants
    let pi = PI;
    let max = f16::MAX;

    // Math
    let y = x.sqrt().sin().abs();

    // Convert
    let wide: f32 = x.into();
    let narrow = wide as f16;

    // Classify
    assert!(x.is_finite());
    assert!(!x.is_nan());
}
```
