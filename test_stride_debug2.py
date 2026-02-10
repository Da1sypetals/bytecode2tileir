#!/usr/bin/env python3
# Analyze mismatch

shape = [256, 256, 128]
strides = [32868, 138, 2]

# Case 1
expected = 526408
got = 524175
tensor_idx = [16, 16, 72]

# What linear index produces tensor_idx [16, 16, 72]?
linear_expected = 16 * 256 * 128 + 16 * 128 + 72
print(f"tensor_idx {tensor_idx} -> linear_idx = {linear_expected}")
print(f"Expected: {expected}, matches: {expected == linear_expected}")

# What tensor_idx produces got value?
linear_got = got
i = linear_got // (256 * 128)
remainder = linear_got % (256 * 128)
j = remainder // 128
k = remainder % 128
print(f"\nGot value {got} comes from tensor_idx [{i}, {j}, {k}]")

# Byte offset for tensor_idx [16, 16, 72]
byte_offset_expected = 16 * strides[0] * 4 + 16 * strides[1] * 4 + 72 * strides[2] * 4
print(f"\nByte offset for [16,16,72] = {byte_offset_expected}")

# Byte offset for got indices
byte_offset_got = i * strides[0] * 4 + j * strides[1] * 4 + k * strides[2] * 4
print(f"Byte offset for [{i},{j},{k}] = {byte_offset_got}")

print(f"\nDifference in byte offset: {byte_offset_got - byte_offset_expected}")
