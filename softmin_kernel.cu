// python/infinicore/ops/softmin/softmin_kernel.cu
#include <cuda_fp16.h>

__global__ void softmin_128x_kernel(const half* x, half* y, int64_t rows, int64_t cols) {
    extern __shared__ half sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= rows) return;
    x += row * cols;
    y += row * cols;

    half thread_max = __float2half(-1e4f);
    for (int i = tid; i < cols; i += 256) {
        thread_max = __hmax(thread_max, x[i]);
    }
    sdata[tid] = thread_max;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = __hmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    half row_max = sdata[0];

    half thread_sum = __float2half(0.0f);
    for (int i = tid; i < cols; i += 256) {
        half val = hexp(__hsub(x[i], row_max));
        sdata[tid] = val;
        thread_sum = __hadd(thread_sum, val);
    }
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = __hadd(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    half row_sum = sdata[0];

    for (int i = tid; i < cols; i += 256) {
        y[i] = hdiv(hexp(__hsub(x[i], row_max)), row_sum);
    }
}

torch::Tensor softmin(const torch::Tensor& input) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat16);
    auto out = torch::empty_like(input);
    int blocks = input.size(0);
    softmin_128x_kernel<<<blocks, 256, 256*sizeof(half)>>>(input.data_ptr<at::Half>(), out.data_ptr<at::Half>(), input.size(0), input.size(1));
    return out;
}