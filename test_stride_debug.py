#!/usr/bin/env python3
# Debug strided access issue

shape = [256, 256, 128]
strides = [32768 + 100, 128 + 10, 2]  # [32868, 138, 2]

# Test case: what's the byte offset for element at [2, 12, 8]?
# Grid pos [0, 0, 0], tile idx [2, 12, 8] means tensor idx [2, 12, 8]

tensor_i, tensor_j, tensor_k = 2, 12, 8

# Expected linear index (row-major)
expected_linear = tensor_i * shape[1] * shape[2] + tensor_j * shape[2] + tensor_k
print(f"Expected linear index: {expected_linear}")

# Byte offset using strides (in elements, multiply by 4 for bytes)
byte_offset = tensor_i * strides[0] * 4 + tensor_j * strides[1] * 4 + tensor_k * strides[2] * 4
print(f"Byte offset: {byte_offset}")

# What linear index was written at this location by fill_tensor_arange_i32?
# The fill function writes to: indices[0]*strides[0]*4 + indices[1]*strides[1]*4 + indices[2]*strides[2]*4
# At offset 'byte_offset', we wrote the linear_idx corresponding to indices [2, 12, 8]
# which is exactly 'expected_linear'

print(f"\nSo at byte offset {byte_offset}, the value should be {expected_linear}")

# Now let's compute what the difference of 59 means
# Got 6725597, expected 6725656, diff = -59
# This means library read from wrong offset that had value 6725597
# What indices produce linear_idx = 6725597?
wrong_val = 6725597
# Decompose: i * 256 * 128 + j * 128 + k = 6725597
i = wrong_val // (256 * 128)
remainder = wrong_val % (256 * 128)
j = remainder // 128
k = remainder % 128
print(f"\nWrong value 6725597 corresponds to indices: [{i}, {j}, {k}]")
print(f"But we wanted indices: [2, 12, 8]")

# Difference in indices
print(f"Difference: i_diff={i-2}, j_diff={j-12}, k_diff={k-8}")
