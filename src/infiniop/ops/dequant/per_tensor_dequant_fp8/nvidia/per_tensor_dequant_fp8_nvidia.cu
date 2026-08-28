#include "../../../../devices/nvidia/nvidia_common.cuh"
#include "per_tensor_dequant_fp8_nvidia.cuh"

#include "../../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../cuda/kernel.cuh"

template <typename Tout>
INFINIOP_CUDA_KERNEL perTensorDequantFp8Sym(
    Tout *x, const unsigned char *x_packed, const float *x_scale,
    size_t batch_size, size_t channel, size_t hidden_dim, size_t width,
    ptrdiff_t strides_0, ptrdiff_t strides_1, ptrdiff_t strides_2, ptrdiff_t strides_3,
    ptrdiff_t p_strides_0, ptrdiff_t p_strides_1, ptrdiff_t p_strides_2, ptrdiff_t p_strides_3,
    int num_elements) {
    perTensorDequantFp8SymKernel<Tout>(x, x_packed, x_scale,
                                       batch_size, channel, hidden_dim, width,
                                       strides_0, strides_1, strides_2, strides_3,
                                       p_strides_0, p_strides_1, p_strides_2, p_strides_3,
                                       num_elements);
}

namespace op::per_tensor_dequant_fp8::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t x_packed_desc,
    infiniopTensorDescriptor_t x_scale_desc) {
    auto info = PerTensorDequantFp8Info::createPerTensorDequantFp8Info(x_desc, x_packed_desc, x_scale_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t per_tensor_dequant_fp8Kernel(const PerTensorDequantFp8Info &info, Tdata *x, const unsigned char *x_packed, const float *x_scale, cudaStream_t stream) {
    int num_elements = (int)info.num_elements;
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t batch_size = info.batch_size;
    size_t channel = info.channel;
    size_t hidden_dim = info.hidden_dim;
    size_t width = info.width;

    ptrdiff_t strides_0 = info.strides_0;
    ptrdiff_t strides_1 = info.strides_1;
    ptrdiff_t strides_2 = info.strides_2;
    ptrdiff_t strides_3 = info.strides_3;

    ptrdiff_t p_strides_0 = info.p_strides_0;
    ptrdiff_t p_strides_1 = info.p_strides_1;
    ptrdiff_t p_strides_2 = info.p_strides_2;
    ptrdiff_t p_strides_3 = info.p_strides_3;

    perTensorDequantFp8Sym<Tdata>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(x, x_packed, x_scale,
                                                batch_size, channel, hidden_dim, width,
                                                strides_0, strides_1, strides_2, strides_3,
                                                p_strides_0, p_strides_1, p_strides_2, p_strides_3,
                                                num_elements);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *x, const void *x_packed, const void *x_scale,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
#define DEQUANT(BLOCK_SIZE, TDATA) \
    per_tensor_dequant_fp8Kernel<BLOCK_SIZE, TDATA>(_info, (TDATA *)x, (const unsigned char *)x_packed, (const float *)x_scale, stream)
#define DEQUANT_WITH_BLOCK_SIZE(BLOCK_SIZE)        \
    {                                              \
        if (_info.dtype == INFINI_DTYPE_F16)       \
            return DEQUANT(BLOCK_SIZE, half);      \
        else if (_info.dtype == INFINI_DTYPE_F32)  \
            return DEQUANT(BLOCK_SIZE, float);     \
        else if (_info.dtype == INFINI_DTYPE_BF16) \
            return DEQUANT(BLOCK_SIZE, __nv_bfloat16); \
        else                                       \
            return INFINI_STATUS_BAD_TENSOR_DTYPE; \
    }
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        DEQUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        DEQUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        DEQUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::per_tensor_dequant_fp8::nvidia
