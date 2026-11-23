// python/infinicore/ops/softshrink/softshrink_kernel.cu
#include <cuda_fp16.h>

__global__ void softshrink_152x_kernel(const half* x, half* y, half lambda, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    half val = x[idx];
    half zero = __float2half(0.0f);
    half pos = __hadd(val, -lambda);
    half neg = __hadd(val, lambda);

    y[idx] = __hgt(val, lambda) ? pos : (__hlt(val, -lambda) ? neg : zero);
}

torch::Tensor softshrink(const torch::Tensor& input, float lambda = 0.5f) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat16);
    auto out = torch::empty_like(input);
    half h_lambda = __float2half(lambda);

    int threads = 512;
    int blocks = (input.numel() + threads - 1) / threads;

    softshrink_152x_kernel<<<blocks, threads>>>(
        input.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        h_lambda,
        input.numel()
    );
    return out;
}