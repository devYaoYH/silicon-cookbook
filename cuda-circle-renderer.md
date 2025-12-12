# Post 2: Parallelizing the Sequential (Circle Renderer)

> **TL;DR:** Achieved 10-15x speedup on a graphics rendering workload by converting random memory access patterns into coalesced reads through tiling, sorting, and careful kernel design.

## The Problem

Render thousands of semi-transparent circles with proper z-ordering (painter's algorithm). The naive approach:

```c
// Serial algorithm - correct but slow
for each circle in order:
    for each pixel:
        if pixel inside circle:
            alpha_blend(pixel, circle.color)
```

Parallelizing over circles creates race conditions. Parallelizing over pixels breaks z-ordering. We needed a third way.

## The Solution: Map-Sort-Reduce

### Algorithm Overview

```
1. TILE image into 32×32 blocks (1024 threads per block)
2. MAP: Each thread takes a circle → writes circle ID to all tiles it overlaps
3. SORT: Use thrust::sort to order circle IDs per tile
4. REDUCE: Each thread takes a pixel → iterate through sorted circles
```

**Key insight:** Converting random pixel access into tile-granularity access allowed batching circle IDs into contiguous memory.

### Implementation Challenges

#### Challenge 1: Dynamic Circle-to-Tile Mapping

We don't know upfront how many circles intersect each tile. Solution:

```c
// Pass 1: Count circles per tile
__global__ void count_circles_per_tile(...);

// CPU: Exclusive scan to get offsets
cudaMemcpy(counts, ...);
exclusive_scan(counts, offsets);  // GPU-based scan

// Pass 2: Write circle IDs to allocated segments
__global__ void write_circle_ids(...);
```

#### Challenge 2: Sorting Per-Tile Segments

Initial approach: Launch `thrust::sort` 1024 times (once per tile).  
**Cost:** 33.9ms for 10K circles (sorting overhead dominated runtime)

**Optimization:** Single `thrust::sort` call with custom comparator:

```cpp
struct TileCircleComparator {
    __host__ __device__
    bool operator()(thrust::tuple<int, int> lhs,
                   thrust::tuple<int, int> rhs) {
        int lhs_tile = thrust::get<0>(lhs);
        int rhs_tile = thrust::get<0>(rhs);
        if (lhs_tile != rhs_tile)
            return lhs_tile < rhs_tile;  // Sort by tile first
        
        // Within tile, maintain circle order
        return thrust::get<1>(lhs) < thrust::get<1>(rhs);
    }
};
```

**Impact:** Sorting time dropped from 33.9ms → 2.3ms (**15x speedup**)

## Optimization Journey

### Speedup #1: Fast Path for Few Circles

For scenes with <16 circles, the tiling overhead wasn't worth it. Added a simple per-pixel kernel:

```c
__global__ void render_few_circles(int* circles, int n_circles, ...) {
    int px = blockIdx.x * 32 + threadIdx.x;
    int py = blockIdx.y * 32 + threadIdx.y;
    
    float4 color = make_float4(1,1,1,1);  // Background
    for (int i = 0; i < n_circles; i++) {
        color = shadePixel(circles[i], px, py, color);
    }
    imgPtr[py * width + px] = color;
}
```

**Result:** RGB test (3 circles) went from 13.7ms → 0.19ms

### Speedup #2: Shared Memory Caching

For each tile, threads repeatedly read the same circle data (position, radius, color). Moved hot data into shared memory:

```c
__shared__ float3 circle_positions[BATCH_SIZE];
__shared__ float circle_radii[BATCH_SIZE];
__shared__ float3 circle_colors[BATCH_SIZE];

// Cooperative load by all threads
int tid = threadIdx.y * 32 + threadIdx.x;
if (tid < batch_size) {
    int cid = tile_circles[start + tid];
    circle_positions[tid] = positions[cid];
    circle_radii[tid] = radii[cid];
    circle_colors[tid] = colors[cid];
}
__syncthreads();

// Now all threads read from fast shared memory
for (int i = 0; i < batch_size; i++) {
    if (inside_circle(px, py, circle_positions[i], circle_radii[i])) {
        blend(color, circle_colors[i]);
    }
}
```

**Impact:** `biglittle` test (unbalanced workload) improved from 66.7ms → 32.2ms

### Speedup #3: Thread-Local Accumulator

Original code read/wrote `imgPtr` for every circle. Optimized to accumulate in registers:

```c
// Read once
float4 pixel_color = imgPtr[pixel_idx];

// Accumulate in register (no memory access)
for (int i = 0; i < n_circles; i++) {
    pixel_color = blend(pixel_color, circles[i]);
}

// Write once
imgPtr[pixel_idx] = pixel_color;
```

### Speedup #4: Kernel Specialization (Warp Divergence)

The `shadePixel` function had branching for snow scenes:

```c
if (scene_type == SNOW) {
    // Special blending logic
} else {
    // Normal alpha blend
}
```

This caused warp divergence when tiles straddled snow/non-snow regions. Solution: Dispatch different kernels:

```c
if (sceneName == SNOW_SINGLE || sceneName == SNOWSINGLE) {
    render_tiles_snow<<<...>>>();
} else {
    render_tiles_standard<<<...>>>();
}
```

**Impact:** Pattern test improved from 34.5ms → 0.57ms

## Final Results

| Scene     | Circles | Ref Time | Optimized | Speedup | Score |
|-----------|---------|----------|-----------|---------|-------|
| rgb       | 3       | 0.27ms   | 0.20ms    | 1.4×    | 9/9   |
| rand10k   | 10K     | 3.06ms   | 1.52ms    | 2.0×    | 9/9   |
| rand100k  | 100K    | 29.4ms   | 9.22ms    | 3.2×    | 9/9   |
| pattern   | ~1K     | 0.41ms   | 0.56ms    | 0.7×    | 8/9   |
| snowsingle| ~10K    | 19.5ms   | 1.03ms    | **18.9×** | 9/9 |
| biglittle | 10K     | 15.1ms   | 10.3ms    | 1.5×    | 9/9   |
| rand1M    | 1M      | 223.6ms  | 15.8ms    | **14.1×** | 9/9 |
| micro2M   | 2M      | 427.2ms  | 13.0ms    | **32.9×** | 9/9 |

**Overall Score:** 71/72 points

## Key Lessons

1. **Coalesced memory access >> random access:** Tiling converted scattered circle lookups into sequential array scans
2. **Sorting cost matters:** Single sort with custom comparator beat many small sorts
3. **Shared memory is fast:** 100× faster than global memory for hot data
4. **Eliminate branching:** Separate kernels avoid warp divergence
5. **Minimize global memory round-trips:** Register accumulation reduced memory traffic

## Limitations

- **Constant overhead:** Tiling/sorting adds ~1ms baseline cost (hurt small scenes)
- **Pattern test:** Low circle density meant many threads did little work
- **Sequential bottleneck:** Pixels with 1000+ overlapping circles still process serially

## What's Next?

The next post moves from graphics (NVIDIA CUDA) to machine learning accelerators (Amazon Trainium), where the programming model shifts from "threads" to "pipelines."

---

**Tools Used:** CUDA, Thrust (GPU sorting library), Nsight Compute  
**Hardware:** NVIDIA T4 (40 SMs, 1024 threads/block max)  
**Lines of Code:** ~800 (including all optimization variants)
