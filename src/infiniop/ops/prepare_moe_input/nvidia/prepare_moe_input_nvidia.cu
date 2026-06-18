#include "prepare_moe_input_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

namespace op::prepare_moe_input::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t expert_offsets_desc,
    infiniopTensorDescriptor_t blockscale_offsets_desc,
    infiniopTensorDescriptor_t problem_sizes1_desc,
    infiniopTensorDescriptor_t problem_sizes2_desc,
    infiniopTensorDescriptor_t input_permutation_desc,
    infiniopTensorDescriptor_t output_permutation_desc,
    infiniopTensorDescriptor_t topk_ids_desc,
    size_t num_experts,
    size_t n,
    size_t k) {
    auto result = PrepareMoeInputInfo::create(
        expert_offsets_desc,
        blockscale_offsets_desc,
        problem_sizes1_desc,
        problem_sizes2_desc,
        input_permutation_desc,
        output_permutation_desc,
        topk_ids_desc,
        num_experts,
        n,
        k);
    CHECK_RESULT(result);
    auto info = result.take();
    const size_t workspace_size = info.num_experts * sizeof(int32_t);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        workspace_size,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

constexpr uint32_t THREADS_PER_EXPERT = 512;
constexpr int32_t BLOCKSCALE_ALIGNMENT = 128;

__global__ void compute_problem_sizes_kernel(
    const int32_t *__restrict__ topk_ids,
    int32_t *__restrict__ problem_sizes1,
    int32_t *__restrict__ problem_sizes2,
    int32_t *__restrict__ atomic_buffer,
    size_t topk_length,
    int32_t n,
    int32_t k) {
    const int expert_id = blockIdx.x;

    int32_t occurrences = 0;
    for (size_t i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
        occurrences += (topk_ids[i] == expert_id);
    }
    atomicAdd(&atomic_buffer[expert_id], occurrences);
    __syncthreads();

    if (threadIdx.x == 0) {
        const int32_t final_occurrences = atomic_buffer[expert_id];
        problem_sizes1[expert_id * 3] = final_occurrences;
        problem_sizes1[expert_id * 3 + 1] = 2 * n;
        problem_sizes1[expert_id * 3 + 2] = k;
        problem_sizes2[expert_id * 3] = final_occurrences;
        problem_sizes2[expert_id * 3 + 1] = k;
        problem_sizes2[expert_id * 3 + 2] = n;
    }
}

__global__ void compute_expert_offsets_kernel(
    const int32_t *__restrict__ problem_sizes1,
    int32_t *__restrict__ expert_offsets,
    int32_t *__restrict__ atomic_buffer,
    size_t num_experts) {
    int32_t total_offset = 0;
    expert_offsets[0] = 0;
    for (size_t expert = 0; expert < num_experts; ++expert) {
        atomic_buffer[expert] = total_offset;
        total_offset += problem_sizes1[expert * 3];
        expert_offsets[expert + 1] = total_offset;
    }
}

__global__ void compute_expert_blockscale_offsets_kernel(
    const int32_t *__restrict__ problem_sizes1,
    int32_t *__restrict__ expert_offsets,
    int32_t *__restrict__ blockscale_offsets,
    int32_t *__restrict__ atomic_buffer,
    size_t num_experts) {
    int32_t total_offset = 0;
    int32_t total_rounded_offset = 0;
    expert_offsets[0] = 0;
    blockscale_offsets[0] = 0;
    for (size_t expert = 0; expert < num_experts; ++expert) {
        atomic_buffer[expert] = total_offset;
        const int32_t num_tokens = problem_sizes1[expert * 3];
        const int32_t rounded_num_tokens = ((num_tokens + BLOCKSCALE_ALIGNMENT - 1) / BLOCKSCALE_ALIGNMENT) * BLOCKSCALE_ALIGNMENT;
        total_offset += num_tokens;
        total_rounded_offset += rounded_num_tokens;
        expert_offsets[expert + 1] = total_offset;
        blockscale_offsets[expert + 1] = total_rounded_offset;
    }
}

__global__ void compute_arg_sorts_kernel(
    const int32_t *__restrict__ topk_ids,
    int32_t *__restrict__ input_permutation,
    int32_t *__restrict__ output_permutation,
    int32_t *__restrict__ atomic_buffer,
    size_t topk_length,
    size_t topk) {
    const int expert_id = blockIdx.x;

    for (size_t i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
        if (topk_ids[i] == expert_id) {
            const int32_t start = atomicAdd(&atomic_buffer[expert_id], 1);
            input_permutation[start] = static_cast<int32_t>(i / topk);
            output_permutation[i] = start;
        }
    }
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *expert_offsets,
    void *blockscale_offsets,
    void *problem_sizes1,
    void *problem_sizes2,
    void *input_permutation,
    void *output_permutation,
    const void *topk_ids,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_info.has_blockscale_offsets && blockscale_offsets == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    auto *atomic_buffer = static_cast<int32_t *>(workspace);
    cudaMemsetAsync(atomic_buffer, 0, _workspace_size, cuda_stream);

    const uint32_t threads = static_cast<uint32_t>(std::min<size_t>(THREADS_PER_EXPERT, _info.topk_length));
    const uint32_t safe_threads = std::max<uint32_t>(threads, 1);
    const uint32_t blocks = static_cast<uint32_t>(_info.num_experts);

    compute_problem_sizes_kernel<<<blocks, safe_threads, 0, cuda_stream>>>(
        static_cast<const int32_t *>(topk_ids),
        static_cast<int32_t *>(problem_sizes1),
        static_cast<int32_t *>(problem_sizes2),
        atomic_buffer,
        _info.topk_length,
        static_cast<int32_t>(_info.n),
        static_cast<int32_t>(_info.k));

    if (_info.has_blockscale_offsets) {
        compute_expert_blockscale_offsets_kernel<<<1, 1, 0, cuda_stream>>>(
            static_cast<const int32_t *>(problem_sizes1),
            static_cast<int32_t *>(expert_offsets),
            static_cast<int32_t *>(blockscale_offsets),
            atomic_buffer,
            _info.num_experts);
    } else {
        compute_expert_offsets_kernel<<<1, 1, 0, cuda_stream>>>(
            static_cast<const int32_t *>(problem_sizes1),
            static_cast<int32_t *>(expert_offsets),
            atomic_buffer,
            _info.num_experts);
    }

    compute_arg_sorts_kernel<<<blocks, safe_threads, 0, cuda_stream>>>(
        static_cast<const int32_t *>(topk_ids),
        static_cast<int32_t *>(input_permutation),
        static_cast<int32_t *>(output_permutation),
        atomic_buffer,
        _info.topk_length,
        _info.topk);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::prepare_moe_input::nvidia
