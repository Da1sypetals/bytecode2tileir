Tests must follow the following guidelines:
- This test is for deep learning use. All tests inputs must use realistic workload shapes, e.g. Sequence length be 4096, matrix dimenision >= 1024 and should be different for M, N, K, embed dim = 64/96/128 etc.
- Tile size must be typical type sizes on GPU, be power of 2 and range from 8 to 64.
- For matrix multiplication, Tensor size must be at least 8 times corresponding tile size, and at least 256. Also you should include tests for irregular shapes, which means tensor size cannot be evenly divided by corresponding tile size (use odd numbers). Using exceedingly small tensor sizes will be harshly punished.
- For flash attention, tiling is as such:
    - Q (Query) is split into blocks of size Br (typically 128 or 64)
    - K and V (Key/Value) are split into blocks of size Bc (typically 128 or 64)


