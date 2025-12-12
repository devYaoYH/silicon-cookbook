# Post 4: Tensor Tetris – Fused Conv-MaxPool on Trainium

> **TL;DR:** Optimized CNN convolution + max pooling kernel on AWS Trainium through output channel tiling, PSUM accumulation strategy, and eliminating redundant memory copies. Achieved <1.2ms for fp16 inference.

## The Challenge

Implement a fused convolution-maxpool layer that:
- Convolves `(B, C_in, H, W)` input with `(C_out, C_in, K_h, K_w)` kernels
- Applies max pooling over `(pool_h, pool_w)` windows
- Runs efficiently on Trainium's systolic array architecture

**Constraint:** PSUM buffer only holds 128 × 512 fp32 values—we must tile carefully.

## Algorithm Overview

### The Tiling Strategy

```python
for out_c_tile in range(0, C_out, 128):           # Tile output channels
    for in_c_tile in range(0, C_in, 128):         # Tile input channels
        for out_h_block in range(0, H_out, block_h):  # Tile output height
            for k_h in range(kernel_height):
                for k_w in range(kernel_width):
                    # Matrix multiply: accumulate in PSUM
                    matmul(inputs[:, :, k_h, k_w], kernels[out_c_tile, in_c_tile, k_h, k_w])
```

**Key decision:** Tile across output channels first, then accumulate an entire output row in PSUM before writing back.

### Why This Tiling?

**PSUM dimensions:** `(128 output_channels, 512 free_dim)`  
**Output row size:** `(128 channels, H_out × W_out positions)`

For a 32×16 image with 3×3 kernel:
- `H_out = 30, W_out = 14`
- Single row = `30 × 14 = 420 < 512` ✓ Fits in PSUM!

This allows us to accumulate partial sums across all `(kernel_height × kernel_width)` filter positions without spilling to HBM.

## Optimization Journey

### Baseline Implementation

```python
for out_c in range(C_out // 128):
    for in_c in range(C_in // 128):
        for fh in range(kernel_height):
            for fw in range(kernel_width):
                X_tile = nl.load(inputs[:, in_c, fh:fh+out_h, fw:fw+out_w])
                W_tile = nl.load(kernels[out_c, in_c, fh, fw])
                
                # Transpose weights for matmul alignment
                W_tile_psum = nisa.nc_transpose(W_tile)
                
                # Accumulate
                partial_sum = nisa.nc_matmul(W_tile_psum, X_tile)
                
        # Copy PSUM → SBUF → HBM
        nl.store(output[out_c, :, :], partial_sum)
```

**Performance (fp16, large image):** 1,600 μs  
**Problem:** Excessive memory copies

### Optimization 1: Remove Redundant tensor_copy

Original code:

```python
X_tile = nisa.tensor_copy(inputs[:, in_c, fh:fh+out_h, fw:fw+out_w])
W_tile = nisa.tensor_copy(kernels[out_c, in_c, fh, fw])
W_tile_psum = nisa.nc_transpose(W_tile)
W_tile = nisa.tensor_copy(W_tile_psum)  # ← Redundant!
```

**Discovery:** `nc_transpose` supports direct slicing! No need for intermediate copies:

```python
# Before: 4 operations
X_tile = tensor_copy(inputs[...])
W_tile = tensor_copy(kernels[...])
W_tile_psum = nc_transpose(W_tile)
W_tile = tensor_copy(W_tile_psum)

# After: 2 operations
W_tile_psum = nisa.nc_transpose(kernels[out_c, in_c, fh, fw])
# X_tile directly uses slice in nc_matmul
```

**Impact:** 1,600 μs → 1,203 μs (33% speedup)  
**Savings:** ~400 μs per kernel from eliminated copies

### Optimization 2: Bias Broadcasting

Original bias addition:

```python
# Explicitly broadcast bias to match PSUM shape
bias_tile = nl.load(bias[out_c_start:out_c_end])
bias_broadcast = nisa.tensor_copy(
    bias_tile, 
    shape=(128, H_out * W_out)  # ← Expensive DMA!
)
out_tile = nisa.tensor_tensor(psum_result, bias_broadcast, op=nl.add)
```

**Discovery:** `tensor_tensor` auto-broadcasts! 

```python
# Just load 1D bias - broadcasting happens automatically
bias_tile = nl.load(bias[out_c_start:out_c_end])
out_tile = nisa.tensor_tensor(psum_result, bias_tile, op=nl.add)
```

**Impact:** Eliminated 200 μs startup overhead

### Optimization 3: Output Height Tiling (Extra Credit)

For small images (32×16), a single output row only uses 420/512 = 82% of PSUM capacity. We can fit multiple rows:

```python
# Calculate how many output rows fit in PSUM
max_free_dim = 512
row_size = out_w  # 14 for 32×16 image
rows_per_tile = min(H_out, max_free_dim // row_size)  # 30 rows × 14 = 420

# Now accumulate multiple rows simultaneously
for h_block in range(0, H_out, rows_per_tile):
    # Single matmul covers all rows in block
    psum_shape = (128, rows_per_tile, out_w)
```

**Impact:** Small image tests (32×16) dropped from 65 μs → 45 μs (fp32), 46 μs → 38 μs (fp16)

## Profiling Insights

### Memory-Bound Kernel

Looking at the trace (with logical cores enabled):

```
Float32 MFU: 41.88%
Float16 MFU: 61.65%
```

**Bottleneck:** DMA transfers dominate. Timeline shows:

1. Green segments (DMA): Loading input tiles from HBM → SBUF
2. Yellow segments (Compute): `tensor_tensor` operations
3. **Idle periods:** DMA engines waiting for SBUF space

The kernel is memory-bound because:
- **Arithmetic intensity too low:** Each weight used only once (no reuse across multiple inputs)
- **Small tile sizes:** 128×128 matmuls don't saturate Tensor Engine

### Why Not More Pipelining?

Unlike vector addition (Part 3), we can't pipeline aggressively here because:
- PSUM must accumulate across all kernel positions before writing
- Overlapping output channels requires more SBUF space than available
- Compiler can't auto-pipeline with complex dependency chains

## Final Performance

**Without profiling overhead:**

| Test Case | Data Type | Image Size | Execution Time | Score |
|-----------|-----------|------------|----------------|-------|
| Large     | fp32      | 1024×1024  | 3,530 μs       | ✓     |
| Large     | fp16      | 1024×1024  | **1,203 μs**   | ✓     |
| Small     | fp32      | 32×16      | 58 μs          | ✓ EC  |
| Small     | fp16      | 32×16      | **38 μs**      | ✓ EC  |

**Key metrics:**
- fp16 achieved 120% of reference performance
- fp32 comfortably within performance targets
- Extra credit tests (<100 μs, <80 μs) passed

## Memory Hierarchy Management

```
┌─────────────────────────────────────┐
│ HBM (32 GB)                         │  ← Input/weights/output live here
└────────────┬────────────────────────┘
             │ DMA (explicit nl.load)
┌────────────▼────────────────────────┐
│ SBUF (24 MB)                        │  ← Staging area for tiles
│  • Input tiles:  128 × H × W        │
│  • Weight tiles: 128 × 128 × K × K  │
└────────────┬────────────────────────┘
             │ Feed to Tensor Engine
┌────────────▼────────────────────────┐
│ Tensor Engine (Systolic Array)     │  ← Matmul happens here
└────────────┬────────────────────────┘
             │ Accumulate partial sums
┌────────────▼────────────────────────┐
│ PSUM (512 KB)                       │  ← Keep hot data here!
│  Shape: (128, 512) fp32             │  ← Single output channel tile
│  Holds: Entire output row(s)        │
└────────────┬────────────────────────┘
             │ Copy completed tile
┌────────────▼────────────────────────┐
│ SBUF → HBM (nl.store)               │
└─────────────────────────────────────┘
```

**Critical insight:** PSUM is precious—design tiling to maximize accumulation before eviction.

## Key Lessons

1. **Zero-copy operations exist:** `nc_transpose` and `tensor_tensor` handle slicing/broadcasting natively
2. **PSUM is the accumulator:** Keep partial sums hot to avoid round-trips
3. **Small tiles hurt:** 128×128 matmuls underutilize systolic arrays (need 1024×1024+ for peak)
4. **Profile with logical cores:** Shows per-NeuronCore utilization patterns
5. **Memory-bound workloads:** Low arithmetic intensity (1 FLOP per 4 bytes) limits throughput

## Limitations

- **Weight reuse:** Kernels aren't reused across different inputs (batch size = 1)
- **Small matmuls:** DSAs excel at large matrix operations (BERT, GPT), not small CNNs
- **Manual tiling:** Hand-tuning required for different image sizes

## What's Next?

Part 5 returns to NVIDIA GPUs to implement FlashAttention—a kernel optimization that uses SRAM tiling to avoid quadratic memory complexity in transformer attention.

---

**Tools Used:** AWS Neuron SDK, NKI, neuron-profile  
**Hardware:** AWS Trn1 (Trainium NeuronCores)  
**Lines of Code:** ~250 (kernel) + ~150 (test harness)
