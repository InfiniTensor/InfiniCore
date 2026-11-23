// python/infinicore/ops/std_mean/std_mean_kernel.cu
#include <cuda_fp16.h>

__global__ void std_mean_158x_kernel(
    const half* __restrict__ x,
    half* __restrict__ mean_out,
    half* __restrict__ std_out,
    int64_t rows,
    int64_t cols
) {
    extern __shared__ half s[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;

    // 第一遍：计算 sum（用于 mean）
    half sum = __float2half(0.0f);
    for (int i = tid; i < cols; i += 256) {
        sum = __hadd(sum, x[row * cols + i]);
    }
    s[tid] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) s[tid] = __hadd(s[tid], s[tid + s]);
        __syncthreads();
    }
    half mean = __hdiv(s[0], __int2half_rn(cols));
    if (tid == 0) mean_out[row] = mean;

    // 第二遍：计算 variance（复用 mean）
    half var = __restrict__ = __float2half(0.0f);
    for (int i = tid; i < cols; i += 256) {
        half diff = __hsub(x[row * cols + i], mean);
        var = __hadd(var, __hmul(diff, diff));
    }
    s[tid] = var;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) s[tid] = __hadd(s[tid], s[tid + s]);
        __syncthreads();
    }
    half std_val = hsqrt(__hdiv(s[0], __int2half_rn(cols)));

    if (tid == 0) std_out[row] = std_val;
}

std::tuple<torch::Tensor, torch::Tensor> std_mean(const torch::Tensor& input) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat16);
    auto mean = torch::empty({input.size(0)}, input.options());
    auto stdv = torch::empty({input.size(0)}, input.options());

    std_mean_158x_kernel<<<input.size(0), 256, 256*sizeof(half)>>>(
        input.data_ptr<at::Half>(),
        mean.data_ptr<at::Half>(),
        stdv.data_ptr<at::Half>(),
        input.size(0),
        input.size(1)
    );

    return {mean, stdv};
}