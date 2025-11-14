#ifndef GPTQ_MARLIN_CUH
#define GPTQ_MARLIN_CUH
#include <cuda.h>

namespace gptq_marlin {
// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

static constexpr int pipe_stages = 4; // 4 pipeline stages fit into shared memory

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

void gptq_marlin_mm_bf16(void *c, const void *a, const void *b, const void *scale,
                         int m, int n, int k,
                         void *workspace, int num_bits,
                         int num_groups, int group_size,
                         int device_id, cudaStream_t stream);
void gptq_marlin_mm_fp16(void *c, const void *a, const void *b, const void *scale,
                         int m, int n, int k,
                         void *workspace, int num_bits,
                         int num_groups, int group_size,
                         int device_id, cudaStream_t stream);

} // namespace gptq_marlin

#endif
