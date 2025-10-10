#include "../../../devices/nvidia/nvidia_common.cuh"
#include "logsoftmax_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../cuda/kernel.cuh"

namespace op::logsoftmax::nvidia {

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
    infiniopTensorDescriptor_t x_desc) {
    auto info = LogSoftmaxInfo::create(y_desc, x_desc);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(void *y, const void *x, infiniDtype_t x_dtype, infiniDtype_t y_dtype,
                            size_t batch_size, size_t probs_size, size_t ndim, size_t seq_len,
                            ptrdiff_t y_stride_b, ptrdiff_t y_stride_p,
                            ptrdiff_t x_stride_b, ptrdiff_t x_stride_p,
                            ptrdiff_t y_stride_0, ptrdiff_t y_stride_1,
                            ptrdiff_t x_stride_0, ptrdiff_t x_stride_1,
                            cudaStream_t stream) {
    dim3 grid(uint32_t(batch_size), 1, 1);

    // Handle mixed precision cases
    if (x_dtype == INFINI_DTYPE_F16 && y_dtype == INFINI_DTYPE_F32) {
        logSoftmax<BLOCK_SIZE, float, half, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((float *)y, (const half *)x,
                                              batch_size, probs_size, ndim, seq_len,
                                              y_stride_b, y_stride_p,
                                              x_stride_b, x_stride_p,
                                              y_stride_0, y_stride_1,
                                              x_stride_0, x_stride_1);
    } else if (x_dtype == INFINI_DTYPE_F32 && y_dtype == INFINI_DTYPE_F16) {
        logSoftmax<BLOCK_SIZE, half, float, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((half *)y, (const float *)x,
                                              batch_size, probs_size, ndim, seq_len,
                                              y_stride_b, y_stride_p,
                                              x_stride_b, x_stride_p,
                                              y_stride_0, y_stride_1,
                                              x_stride_0, x_stride_1);
    } else if (x_dtype == INFINI_DTYPE_BF16 && y_dtype == INFINI_DTYPE_F32) {
        logSoftmax<BLOCK_SIZE, float, __nv_bfloat16, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((float *)y, (const __nv_bfloat16 *)x,
                                              batch_size, probs_size, ndim, seq_len,
                                              y_stride_b, y_stride_p,
                                              x_stride_b, x_stride_p,
                                              y_stride_0, y_stride_1,
                                              x_stride_0, x_stride_1);
    } else if (x_dtype == INFINI_DTYPE_F32 && y_dtype == INFINI_DTYPE_BF16) {
        logSoftmax<BLOCK_SIZE, __nv_bfloat16, float, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((__nv_bfloat16 *)y, (const float *)x,
                                              batch_size, probs_size, ndim, seq_len,
                                              y_stride_b, y_stride_p,
                                              x_stride_b, x_stride_p,
                                              y_stride_0, y_stride_1,
                                              x_stride_0, x_stride_1);
    } else if (x_dtype == INFINI_DTYPE_F16 && y_dtype == INFINI_DTYPE_F16) {
        logSoftmax<BLOCK_SIZE, half, half, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((half *)y, (const half *)x,
                                              batch_size, probs_size, ndim, seq_len,
                                              y_stride_b, y_stride_p,
                                              x_stride_b, x_stride_p,
                                              y_stride_0, y_stride_1,
                                              x_stride_0, x_stride_1);
    } else if (x_dtype == INFINI_DTYPE_BF16 && y_dtype == INFINI_DTYPE_BF16) {
        logSoftmax<BLOCK_SIZE, __nv_bfloat16, __nv_bfloat16, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((__nv_bfloat16 *)y, (const __nv_bfloat16 *)x,
                                              batch_size, probs_size, ndim, seq_len,
                                              y_stride_b, y_stride_p,
                                              x_stride_b, x_stride_p,
                                              y_stride_0, y_stride_1,
                                              x_stride_0, x_stride_1);
    } else if (x_dtype == INFINI_DTYPE_F32 && y_dtype == INFINI_DTYPE_F32) {
        logSoftmax<BLOCK_SIZE, float, float, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((float *)y, (const float *)x,
                                              batch_size, probs_size, ndim, seq_len,
                                              y_stride_b, y_stride_p,
                                              x_stride_b, x_stride_p,
                                              y_stride_0, y_stride_1,
                                              x_stride_0, x_stride_1);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            y, x, _info.x_dtype, _info.y_dtype, _info.batch_size, _info.probs_size, _info.ndim, _info.seq_len,
            _info.y_stride_b, _info.y_stride_p, _info.x_stride_b, _info.x_stride_p,
            _info.y_stride_0, _info.y_stride_1, _info.x_stride_0, _info.x_stride_1, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            y, x, _info.x_dtype, _info.y_dtype, _info.batch_size, _info.probs_size, _info.ndim, _info.seq_len,
            _info.y_stride_b, _info.y_stride_p, _info.x_stride_b, _info.x_stride_p,
            _info.y_stride_0, _info.y_stride_1, _info.x_stride_0, _info.x_stride_1, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            y, x, _info.x_dtype, _info.y_dtype, _info.batch_size, _info.probs_size, _info.ndim, _info.seq_len,
            _info.y_stride_b, _info.y_stride_p, _info.x_stride_b, _info.x_stride_p,
            _info.y_stride_0, _info.y_stride_1, _info.x_stride_0, _info.x_stride_1, stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::logsoftmax::nvidia
