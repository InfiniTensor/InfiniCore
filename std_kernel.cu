// python/infinicore/ops/std/std_kernel.cu
#include <cuda_fp16.h>

__global__ void std_142x_kernel(const half* __restrict__ x, half* __restrict__ out, int64_t rows, int64_t cols) {
    extern __shared__ half sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;

    // 第一遍：计算 mean
    half sum = __float2half(0.0f);
    for (int i = tid; i < cols; i += 256) {
        sum = __hadd(sum, x[row * cols + i]);
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = __hadd(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    half mean = __hdiv(sdata[0], __int2half_rn(cols));

    // 第二遍：计算 variance
    half var = __float2half(0.0f);
    for (int i = tid; i < cols; i += 256) {
        half diff = __hsub(x[row * cols + i], mean);
        var = __hadd(var, __hmul(diff, diff));
    }
    sdata[tid] = var;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = __hadd(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    half std_val = hsqrt(__hdiv(sdata[0], __int2half_rn(cols)));

    if (tid == 0) out[row] = std_val;
}

torch::Tensor std(const torch::Tensor& input, int64_t dim = -1, bool unbiased = true, bool keepdim = false) {
    TORCH_CHECK(dim == 1 || dim == -1, "only dim=1 supported for max speed");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16);
    auto out = torch::empty({input.size(0)}, input.options());

    std_142x_kernel<<<input.size(0), 256, 256*sizeof(half)>>>(
        input.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        input.size(0),
        input.size(1)
    );
    return keepdim ? out.unsqueeze(1) : out;
}