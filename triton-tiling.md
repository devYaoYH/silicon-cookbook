# Post 5: FlashAttention – SRAM Tiling for Quadratic Complexity

> **TL;DR:** Implemented FlashAttention in OpenAI Triton, achieving 26.9ms execution (faster than PyTorch's 28ms baseline) through online softmax, optimized tiling, and grid swizzling to fix L2 cache thrashing.

## The Problem: Attention's Memory Wall

Standard attention computation:

```python
# Naive implementation - O(N²) memory
scores = Q @ K.T              # (B, H, N, N) - huge!
attention = softmax(scores)   # Materialize full matrix
output = attention @ V        # Final result
```

For sequence length N=4096, a single attention matrix is:
- **Memory:** 4096² × 4 bytes (fp32) = 67 MB per head
- **For 32 heads:** 2.1 GB just for intermediate scores

**FlashAttention insight:** Never materialize the full N×N matrix—tile and fuse operations in SRAM.

## The Algorithm: Online Softmax

Key mathematical trick from the [FlashAttention paper](https://arxiv.org/abs/2205.14135):

Instead of computing normalized output `O = softmax(QK^T) @ V` in two passes, maintain **running statistics**:

```
For each Q tile (size M):
    For each K/V tile (size N):
        1. Compute scores: S_i = Q @ K^T
        2. Update running max: m_i = max(m_{i-1}, max(S_i))
        3. Adjust previous sums: d_i = d_{i-1} * exp(m_{i-1} - m_i) + sum(exp(S_i - m_i))
        4. Update output: O_i = O_{i-1} * exp(m_{i-1} - m_i) / d_i + exp(S_i - m_i) @ V / d_i
```

**Critical optimization (NotebookLM-assisted discovery):** Track the **unnormalized numerator** instead:

```python
# Instead of: O_i = (O_{i-1} * d_{i-1} + P_i @ V) / d_i  (division in loop!)
# Do: Acc_i = O_i * d_i  (defer division until end)

# Recurrence becomes:
Acc_i = Acc_{i-1} * exp(m_{i-1} - m_i) + exp(S_i - m_i) @ V

# Final output:
O = Acc_N / d_N  (single division at end)
```

This eliminates costly divisions from the inner loop.

## Implementation in Triton

### Kernel Structure

```python
@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm,
    BLOCK_SIZE_M: tl.constexpr,  # Q tile size
    BLOCK_SIZE_N: tl.constexpr,  # K/V tile size
    BLOCK_SIZE_D: tl.constexpr,  # Head dimension
):
    # Load Q tile once (shape: BLOCK_SIZE_M × D)
    q_tile = tl.load(q_ptrs).to(tl.float16)
    
    # Initialize accumulator (unnormalized numerator)
    acc_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    prev_m = tl.full((BLOCK_SIZE_M,), -float('inf'), dtype=tl.float32)
    prev_d = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over K/V tiles
    for k_idx in range(0, seq_len, BLOCK_SIZE_N):
        k_tile = tl.load(k_ptrs).to(tl.float16)
        v_tile = tl.load(v_ptrs).to(tl.float16)
        
        # Compute attention scores (FP16 matmul → FP32 accumulation)
        scores = tl.dot(q_tile, k_tile) * scale  # (M, N)
        
        # Online softmax statistics
        m_local = tl.max(scores, axis=1)  # (M,)
        m_new = tl.maximum(m_local, prev_m)
        
        # Compute safe exponentials
        exp_adjustment = tl.exp(prev_m - m_new)
        exp_scores = tl.exp(scores - m_new[:, None])  # (M, N)
        
        # Update running sum
        d_new = prev_d * exp_adjustment + tl.sum(exp_scores, axis=1)
        
        # Update accumulator (key optimization!)
        acc_tile = acc_tile * exp_adjustment[:, None] + tl.dot(exp_scores, v_tile)
        
        prev_m, prev_d = m_new, d_new
    
    # Final normalization (single division)
    output = (acc_tile / prev_d[:, None]).to(tl.float16)
    tl.store(out_ptrs, output)
```

### Memory Layout

Each thread block keeps in SRAM:
- `q_tile`: M × D fp16 (32 KB for M=128, D=128)
- `k_tile`, `v_tile`: N × D fp16 (32 KB each)
- `scores`: M × N fp32 (64 KB)
- `acc_tile`: M × D fp32 (64 KB)
- **Total:** ~225 KB (saturates H100's 228 KB L1/SRAM)

## Optimization Journey

### Baseline: M=N=64

```
Execution Time:    134.75 ms
Compute Throughput: 20.53%
DRAM Read:         129.45 GB
L2 Cache Hit Rate:  16.92%  ← Terrible!
```

**Problem:** Excessive DRAM traffic and cache misses.

### Optimization 1: Autotuning Block Sizes

Used Triton's `@triton.autotune` to grid-search parameters:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, 
                     num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, 
                     num_stages=4, num_warps=8),
        # ... more configs
    ],
    key=['seq_len', 'head_dim']
)
```

**Best config:** M=N=128, 2 pipeline stages, 8 warps

```
Execution Time:    88.38 ms (1.5× faster)
DRAM Read:         64.50 GB (2× reduction)
L2 Cache Hit Rate:  0.80%  ← Still terrible!
```

### Optimization 2: Grid Swizzling (The Breakthrough)

**Problem identified by Nsight Compute:** L2 cache thrashing. The default grid launches blocks in order:

```
Block 0: (batch=0, head=0, query_block=0)
Block 1: (batch=0, head=0, query_block=1)
Block 2: (batch=0, head=0, query_block=2)
...
Block 256: (batch=1, head=0, query_block=0)
```

**Issue:** By the time `batch=1` loads K/V tiles, `batch=0`'s K/V data has been evicted from L2.

**Solution:** Reorder execution to prioritize batch → head → spatial locality:

```python
# Flatten 3D grid to 1D
pid = tl.program_id(0)

# Original mapping
# batch_idx = pid // (num_heads * num_query_blocks)
# head_idx = (pid % (num_heads * num_query_blocks)) // num_query_blocks
# query_idx = pid % num_query_blocks

# Swizzled mapping (prioritize batch/head grouping)
num_blocks_per_batch = num_heads * num_query_blocks
batch_idx = pid // num_blocks_per_batch
head_idx = (pid % num_blocks_per_batch) // num_query_blocks
query_idx = pid % num_query_blocks
```

Execution order becomes:

```
Block 0: (batch=0, head=0, query=0)  ← Load K/V for head 0
Block 1: (batch=0, head=0, query=1)  ← Reuse K/V from L2!
Block 2: (batch=0, head=0, query=2)  ← Still in cache
...
Block 32: (batch=0, head=1, query=0) ← Load K/V for head 1
```

**Impact:**

```
Execution Time:    33.86 ms (2.6× faster!)
Compute Throughput: 36.42%
DRAM Read:         1.50 GB (43× reduction!)
L2 Cache Hit Rate:  91.90% (0.8% → 92%)
```

### Without Profiling Overhead

```
Final kernel time: 26.9 ms
PyTorch baseline:  28.0 ms
Speedup:          1.04× ✓
```

## Profiling Deep Dive

**Before swizzling:**
- Every query block loads same K/V tiles independently
- 64 query blocks × 2 GB K/V = 128 GB traffic
- L2 (40 MB) can't hold working set

**After swizzling:**
- K/V loaded once per head, reused across all queries
- 32 heads × 64 MB K/V = 2 GB traffic
- L2 hit rate 92% → DRAM bandwidth saved for output writes

## Key Lessons

1. **Mathematical optimization matters:** Deferring normalization saved inner-loop divisions
2. **SRAM capacity is the limit:** 128×128 tiles saturate 228 KB L1 on H100
3. **Cache locality beats raw compute:** 43× DRAM reduction from better scheduling
4. **Grid layout affects performance:** Default Triton dispatch isn't always optimal
5. **LLM-assisted debugging:** NotebookLM helped discover the numerator simplification; Claude Code identified cache thrashing

## Triton-Specific Insights

**Strengths:**
- High-level Python API with implicit CUDA generation
- `tl.dot` auto-promotes fp16 → fp32 accumulation
- `@autotune` makes hyperparameter search easy

**Gotchas:**
- Default grid launch order isn't cache-aware
- Must manually manage SRAM tile sizes
- Profiling changes execution characteristics

## What's Next?

This kernel runs on H100 GPUs at datacenter scale. The next post could explore:
- Multi-query attention (MQA) optimizations
- Quantized attention (fp8/int8)
- Distributed attention across GPU clusters

---

**Tools Used:** OpenAI Triton, Nsight Compute, PyTorch (baseline), NotebookLM  
**Hardware:** NVIDIA H100 (80 GB, 228 KB L1/SM, 50 MB L2)  
**Performance:** 26.9ms (vs 28ms PyTorch), 36% compute throughput, 92% L2 hit rate
