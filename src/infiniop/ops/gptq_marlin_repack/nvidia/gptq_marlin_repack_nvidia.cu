#if defined(ENABLE_NVIDIA_API)
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "gptq_marlin_repack_nvidia.cuh"
#include <cassert>

template <int const num_threads, int const num_bits, bool const has_perm,
          bool is_a_8bit>
INFINIOP_CUDA_KERNEL gptqMarlinRepackKernel(
    const uint32_t *__restrict__ b_q_weight_ptr,
    const uint32_t *__restrict__ perm_ptr, uint32_t *__restrict__ out_ptr,
    int size_k, int size_n) {

    marlin::gptq_marlin_repack_kernel<num_threads, num_bits, has_perm, is_a_8bit>(
        b_q_weight_ptr, perm_ptr, out_ptr,
        size_k, size_n);
}

#define CALL_IF(NUM_BITS, HAS_PERM, IS_A_8BIT)                                         \
    else if (num_bits == NUM_BITS && has_perm == HAS_PERM && is_a_8bit == IS_A_8BIT) { \
        cudaFuncSetAttribute(                                                          \
            gptqMarlinRepackKernel<marlin::repack_threads, NUM_BITS,                   \
                                   HAS_PERM, IS_A_8BIT>,                               \
            cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);              \
        gptqMarlinRepackKernel<marlin::repack_threads, NUM_BITS,                       \
                               HAS_PERM, IS_A_8BIT>                                    \
            <<<blocks, marlin::repack_threads, max_shared_mem, stream>>>(              \
                b_q_weight_ptr, perm_ptr, out_ptr, size_k, size_n);                    \
    }

infiniStatus_t gptqMarlinRepack(uint32_t *out_ptr, const uint32_t *b_q_weight_ptr, const uint32_t *perm_ptr,
                                int64_t size_k, int64_t size_n, int64_t num_bits,
                                bool is_a_8bit, bool has_perm, cudaStream_t stream) {

    // Verify compatibility with marlin tile of 16x64
    if (size_k % marlin::tile_k_size != 0) {
        std::cout << "size_k = " << size_k << " is not divisible by tile_k_size = " << marlin::tile_k_size << std::endl;
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (size_n % marlin::tile_n_size != 0) {
        std::cout << "size_n = " << size_n << " is not divisible by tile_n_size = " << marlin::tile_n_size << std::endl;
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (num_bits != 4 && num_bits != 8) {
        std::cout << "num_bits must be 4 or 8. Got = " << num_bits << std::endl;
        return INFINI_STATUS_BAD_PARAM;
    }

    // Get dev info
    int device_id = 0;

    int blocks;
    cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, device_id);

    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    assert(max_shared_mem > 0 && "max_shared_mem must be greater than 0");

    if (false) {
    }
    CALL_IF(4, false, false)
    CALL_IF(4, true, false)
    CALL_IF(8, false, false)
    CALL_IF(8, true, false)

    CALL_IF(4, false, true)
    CALL_IF(8, false, true)
    else {
        fprintf(stderr, "Unsupported repack config: num_bits = %ld, has_perm = %s, is_a_8bit = %s\n",
                num_bits,
                has_perm ? "true" : "false",
                is_a_8bit ? "true" : "false");
        assert(false);
    }

    return INFINI_STATUS_SUCCESS;
}

namespace op::gptq_marlin_repack::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t perm_desc,
    int64_t num_bits,
    bool is_a_8bit) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto result = GptqMarlinRepackInfo::create(output_desc, input_desc, perm_desc, num_bits, is_a_8bit);

    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        result.take(),
        workspace_size,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t
Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output,
    const void *input,
    const void *perm,
    void *stream_) const {

    cudaStream_t stream = (cudaStream_t)stream_;

    int64_t size_k = static_cast<int64_t>(_info.size_k);
    int64_t size_n = static_cast<int64_t>(_info.size_n);
    int64_t num_bits = _info.num_bits;
    bool is_a_8bit = _info.is_a_8bit;
    bool has_perm = _info.has_perm;

    gptqMarlinRepack((uint32_t *)output, (const uint32_t *)input, (const uint32_t *)perm,
                     size_k, size_n, num_bits,
                     is_a_8bit, has_perm, stream);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gptq_marlin_repack::nvidia
#endif // ENABLE_NVIDIA_API
