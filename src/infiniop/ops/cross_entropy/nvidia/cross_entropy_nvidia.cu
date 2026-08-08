#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "cross_entropy_nvidia.cuh"

template <unsigned int BLOCK_SIZE,
          typename Tout,
          typename Tdata,
          typename Tidx,
          typename Tcompute = float>
INFINIOP_CUDA_KERNEL crossEntropy(
    Tout *y, const Tdata *x, const void *target,
    size_t outer_size, size_t vocab_size, ptrdiff_t x_stride) {

    crossEntropyKernel<BLOCK_SIZE, Tout, Tdata, Tidx, Tcompute>(
        y, x, target, outer_size, vocab_size, x_stride);
}

namespace op::cross_entropy::nvidia {

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
    infiniopTensorDescriptor_t target_desc) {

    auto y_dtype = y_desc->dtype();
    auto x_dtype = x_desc->dtype();
    auto t_dtype = target_desc->dtype();

    CHECK_DTYPE(x_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_DTYPE(y_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_DTYPE(t_dtype, INFINI_DTYPE_I32, INFINI_DTYPE_I64);

    if (y_dtype != x_dtype && y_dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CrossEntropyInfo info{};
    info.dtype = x_dtype;
    info.output_dtype = y_dtype;
    info.target_dtype = t_dtype;

    info.vocab_size = x_desc->shape().back();
    info.outer_size = target_desc->numel();
    info.x_stride = static_cast<ptrdiff_t>(info.vocab_size);

    auto internal = reinterpret_cast<device::nvidia::Handle *>(handle)->internal();

    *desc_ptr = new Descriptor(
        new Opaque{internal},
        info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tidx>
infiniStatus_t launchTypedKernel(void *y, const void *x, const void *target,
                                const CrossEntropyInfo &info, cudaStream_t stream) {
    dim3 grid(static_cast<uint32_t>(info.outer_size), 1, 1);
    if (info.output_dtype == INFINI_DTYPE_F32) {
        crossEntropy<BLOCK_SIZE, float, Tdata, Tidx>
            <<<grid, BLOCK_SIZE, 0, stream>>>(
                (float *)y,
                (const Tdata *)x,
                target,
                info.outer_size,
                info.vocab_size,
                info.x_stride);
    } else if (info.output_dtype == info.dtype) {
        crossEntropy<BLOCK_SIZE, Tdata, Tdata, Tidx>
            <<<grid, BLOCK_SIZE, 0, stream>>>(
                (Tdata *)y,
                (const Tdata *)x,
                target,
                info.outer_size,
                info.vocab_size,
                info.x_stride);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(void *y, const void *x, const void *target,
                            const CrossEntropyInfo &info, cudaStream_t stream) {
    if (info.target_dtype == INFINI_DTYPE_I64) {
        if (info.dtype == INFINI_DTYPE_F16) {
            return launchTypedKernel<BLOCK_SIZE, half, int64_t>(
                y, x, target, info, stream);
        }
        if (info.dtype == INFINI_DTYPE_BF16) {
            return launchTypedKernel<BLOCK_SIZE, __nv_bfloat16, int64_t>(
                y, x, target, info, stream);
        }
        if (info.dtype == INFINI_DTYPE_F32) {
            return launchTypedKernel<BLOCK_SIZE, float, int64_t>(
                y, x, target, info, stream);
        }
    } else if (info.target_dtype == INFINI_DTYPE_I32) {
        if (info.dtype == INFINI_DTYPE_F16) {
            return launchTypedKernel<BLOCK_SIZE, half, int32_t>(
                y, x, target, info, stream);
        }
        if (info.dtype == INFINI_DTYPE_BF16) {
            return launchTypedKernel<BLOCK_SIZE, __nv_bfloat16, int32_t>(
                y, x, target, info, stream);
        }
        if (info.dtype == INFINI_DTYPE_F32) {
            return launchTypedKernel<BLOCK_SIZE, float, int32_t>(
                y, x, target, info, stream);
        }
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     const void *target,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;

    int max_threads = _opaque->internal->maxThreadsPerBlock();

    if (max_threads >= 1024) {
        CHECK_STATUS(launchKernel<1024>(y, x, target, _info, stream));
    } else if (max_threads >= 512) {
        CHECK_STATUS(launchKernel<512>(y, x, target, _info, stream));
    } else {
        CHECK_STATUS(launchKernel<256>(y, x, target, _info, stream));
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::cross_entropy::nvidia
