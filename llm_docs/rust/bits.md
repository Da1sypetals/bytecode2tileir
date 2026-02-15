# f32 Bit Manipulation Functions

## `pub const fn to_bits(self) -> u32`

**Raw transmutation to u32.**

This is currently identical to `transmute::<f32, u32>(self)` on all platforms.

See `from_bits` for some discussion of the portability of this operation (there are almost no issues).

**Note** that this function is distinct from `as` casting, which attempts to preserve the numeric value, and not the bitwise value.

### Examples

```rust
assert_ne!((1f32).to_bits(), 1f32 as u32); // to_bits() is not casting!
assert_eq!((12.5f32).to_bits(), 0x41480000);
```

*Stable since: 1.20.0 (const: 1.83.0) · Source*

---

## `pub const fn from_bits(v: u32) -> f32`

**Raw transmutation from u32.**

This is currently identical to `transmute::<u32, f32>(v)` on all platforms. It turns out this is incredibly portable, for two reasons:

- Floats and Ints have the same endianness on all supported platforms.
- IEEE 754 very precisely specifies the bit layout of floats.

### Portability Caveat

However there is one caveat: prior to the 2008 version of IEEE 754, how to interpret the NaN signaling bit wasn't actually specified. Most platforms (notably x86 and ARM) picked the interpretation that was ultimately standardized in 2008, but some didn't (notably MIPS). As a result, all signaling NaNs on MIPS are quiet NaNs on x86, and vice-versa.

Rather than trying to preserve signaling-ness cross-platform, this implementation favors preserving the exact bits. This means that any payloads encoded in NaNs will be preserved even if the result of this method is sent over the network from an x86 machine to a MIPS one.

**No portability concern when:**
- The results of this method are only manipulated by the same architecture that produced them
- The input isn't NaN
- You don't care about signalingness (very likely)

**Note** that this function is distinct from `as` casting, which attempts to preserve the numeric value, and not the bitwise value.

### Examples

```rust
let v = f32::from_bits(0x41480000);
assert_eq!(v, 12.5);
```