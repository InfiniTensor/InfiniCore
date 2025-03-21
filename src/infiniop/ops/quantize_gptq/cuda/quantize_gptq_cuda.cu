#include "../../../devices/cuda/cuda_common.cuh"
#include "gptq_marlin.cuh"
#include "quantize_gptq_cuda.cuh"
#include <cassert>
#ifdef NDEBUG
#define SAFE_ASSERT(x) ((void)(x))
#else
#define SAFE_ASSERT(x) assert(x)
#endif
namespace op::quantize_gptq::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t packed_weights_desc,
    infiniopTensorDescriptor_t b_scale_desc,
    infiniopTensorDescriptor_t zero_desc) {
    auto result = MatmulGptqInfo::createMatmulGptqInfo(c_desc, a_desc, packed_weights_desc, b_scale_desc, zero_desc);
    CHECK_RESULT(result);
    MatmulGptqInfo info = result.take();
    int max_par = gptq_marlin::max_par;
    size_t min_workspace_size = info.n / gptq_marlin::min_thread_n * max_par * sizeof(int) + info.m * info.k * infiniSizeOf(info.atype);

    *desc_ptr = new Descriptor(info, new Opaque{reinterpret_cast<device::cuda::Handle *>(handle)->internal()}, min_workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::quant(
    void *workspace,
    size_t workspace_size,
    void *packed_weights,
    void *b_scale,
    void *zero,
    const void *a,
    const void *b,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    void *packed_weights,
    void *b_scale,
    void *zero,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    int m = int(_info.m);
    int n = int(_info.n);
    int k = int(_info.k);
    int bits = 4;
    int group_size = int(_info.group_size);
    int num_groups = int(_info.num_groups);
    bool is_weight_transposed = _info.is_weight_transposed;
    if (_info.atype == INFINI_DTYPE_F16 && !is_weight_transposed) {
        gptq_marlin::gptq_marlin_mm_fp16(c, a, packed_weights, b_scale,
                                         m, n, k,
                                         workspace, bits,
                                         num_groups, group_size,
                                         this->device_id, (cudaStream_t)stream);

    } else if (_info.atype == INFINI_DTYPE_BF16 && !is_weight_transposed) {
        gptq_marlin::gptq_marlin_mm_bf16(c, a, packed_weights, b_scale,
                                         m, n, k,
                                         workspace, bits,
                                         num_groups, group_size,
                                         this->device_id, (cudaStream_t)stream);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::quantize_gptq::cuda
