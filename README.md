# Silicon Cookbook

A collection of low-level systems programming adventures across different hardware accelerators. These posts explore optimization techniques, memory hierarchies, and architecture-specific patterns for CUDA, Trainium, and Triton.

## Blog Posts

### 1. [Parallelizing the Sequential (Circle Renderer)](cuda-circle-renderer.md)
**Platform:** NVIDIA CUDA (T4 GPU)

Achieved 10-15x speedup on a graphics rendering workload by converting random memory access patterns into coalesced reads. Explores tiling strategies, sorting optimizations, shared memory caching, and warp divergence elimination.

**Key topics:** Memory coalescing, Thrust sorting, shared memory, kernel specialization

---

### 2. [Tensor Tetris – Fused Conv-MaxPool on Trainium](trainium-systolic-array.md)
**Platform:** AWS Trainium

Optimized CNN convolution + max pooling kernel through output channel tiling and PSUM accumulation strategy. Deep dive into systolic array programming, memory hierarchy management (SBUF/PSUM), and eliminating redundant copies.

**Key topics:** Systolic arrays, PSUM buffer management, DMA optimization, zero-copy operations

---

### 3. [FlashAttention – SRAM Tiling for Quadratic Complexity](triton-tiling.md)
**Platform:** NVIDIA H100 (OpenAI Triton)

Implemented FlashAttention achieving 26.9ms execution (faster than PyTorch baseline) through online softmax, optimized tiling, and grid swizzling. Reduced DRAM traffic by 43x through cache-aware scheduling.

**Key topics:** Online softmax, SRAM tiling, L2 cache optimization, grid swizzling, autotuning

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](LICENSE.md).
