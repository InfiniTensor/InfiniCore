// python/infinicore/ops/split/split_kernel.cu
#include <cuda_fp16.h>

extern "C" __global__ void split_168x_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t total_elements,
    int64_t split_size,
    int64_t num_splits
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int64_t split_idx = idx / split_size;
    int64_t offset_in_split = idx % split_size;

    int64_t src_pos = split_idx * split_size + offset_in_split;
    int64_t dst_pos = split_idx * total_elements + offset_in_split;

    output[dst_pos] = input[src_pos];
}

torch::Tensor split_cuda(const torch::Tensor& input, int64_t split_size, int64_t dim) {
    TORCH_CHECK(dim == 0, "split only supports dim=0 for max speed");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16);

    int64_t outer = input.size(0);
    int64_t inner = input.numel() / outer;
    int64_t num_splits = (outer + split_size - 1) / split_size;

    auto output = torch::empty({num_splits, split_size, inner}, input.options());

    int64_t total_elements = num_splits * split_size * inner;
    int threads = 512;
    int blocks = (total_elements + threads - 1) / threads;

    split_168x_kernel<<<blocks, threads>>>(
        input.data_ptr<at::Half>(),
        output.data_ptr<at::Half>(),
        total_elements,
        split_size * inner,
        num_splits
    );

    return output;
}