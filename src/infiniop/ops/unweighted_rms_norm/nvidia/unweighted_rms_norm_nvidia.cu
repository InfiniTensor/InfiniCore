#include "../../../devices/nvidia/nvidia_common.cuh"
#include "unweighted_rms_norm_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata>
INFINIOP_CUDA_KERNEL unweightedRMSNormKernel(
    Tdata *__restrict__ y,
    const Tdata *__restrict__ x,
    size_t dim,
    float epsilon) {
    unweightedRMSNormBlock<BLOCK_SIZE, Tcompute>(y, x, dim, epsilon);
}

namespace op::unweighted_rms_norm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    float epsilon) {
    auto result = UnweightedRMSNormInfo::create(y_desc, x_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    uint32_t outer_size,
    size_t dim,
    void *y,
    infiniDtype_t atype,
    const void *x,
    float epsilon,
    cudaStream_t cuda_stream) {

#define LAUNCH_KERNEL(Tdata, Tcompute)                                         \
    unweightedRMSNormKernel<BLOCK_SIZE, Tcompute, Tdata><<<outer_size, BLOCK_SIZE, 0, cuda_stream>>>( \
        reinterpret_cast<Tdata *>(y),                                          \
        reinterpret_cast<const Tdata *>(x),                                    \
        dim,                                                                   \
        epsilon)

    if (atype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(half, float);
    } else if (atype == INFINI_DTYPE_BF16) {
        LAUNCH_KERNEL(__nv_bfloat16, float);
    } else if (atype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(float, float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_KERNEL

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    uint32_t outer_size = static_cast<uint32_t>(_info.outer_size);
    size_t dim = _info.last_dim;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(outer_size, dim, y, _info.atype, x, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_2048>(outer_size, dim, y, _info.atype, x, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(outer_size, dim, y, _info.atype, x, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(outer_size, dim, y, _info.atype, x, _info.epsilon, cuda_stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::unweighted_rms_norm::nvidia
