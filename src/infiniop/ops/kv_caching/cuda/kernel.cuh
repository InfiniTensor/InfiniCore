#ifndef __KV_CACHING_KERNEL_CUH__
#define __KV_CACHING_KERNEL_CUH__

template <typename Tdata>
__device__ void kvCachingKernel(
    Tdata *__restrict__ k_cache,
    Tdata *__restrict__ v_cache,
    const Tdata *__restrict__ k,
    const Tdata *__restrict__ v,
    const int64_t *__restrict__ past_kv_lengths,
    int batch_size,
    int num_kv_heads,
    int max_seq_len,
    int seq_len,
    int hidden_dim,
    ptrdiff_t cache_strides_0,
    ptrdiff_t cache_strides_1,
    ptrdiff_t cache_strides_2,
    ptrdiff_t cache_strides_3) {
    // 总元素数 = B * H * seq_len * D
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_kv_heads * seq_len * hidden_dim;

    const int grid_size = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total; idx += grid_size) {
        // 反解 index

        int d = idx % hidden_dim;
        idx /= hidden_dim;

        int s = idx % seq_len;
        idx /= seq_len;

        int h = idx % num_kv_heads;
        int b = idx / num_kv_heads;

        int past_len = static_cast<int32_t>(past_kv_lengths[b]);
        // 写入位置
        int cache_s = past_len + s;
        int cache_offset = d * (int)cache_strides_3 + cache_s * (int)cache_strides_2 + h * (int)cache_strides_1 + b * (int)cache_strides_0;

        int src_offset = ((b * num_kv_heads + h) * seq_len + s) * hidden_dim + d;

        k_cache[cache_offset] = k[src_offset];
        v_cache[cache_offset] = v[src_offset];
    }
}

#endif // __KV_CACHING_KERNEL_CUH__
