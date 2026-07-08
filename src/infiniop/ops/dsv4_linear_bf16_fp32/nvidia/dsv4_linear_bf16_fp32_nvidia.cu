#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_linear_bf16_fp32_nvidia.cuh"

namespace {

template <typename T>
__global__ void castToFloatKernel(const T *x, float *x_fp32, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        x_fp32[idx] = static_cast<float>(x[idx]);
    }
}

template <typename T>
infiniStatus_t castToFloat(const void *x, void *workspace, size_t total, cudaStream_t stream) {
    constexpr int block = 256;
    size_t blocks = (total + block - 1) / block;
    int grid = static_cast<int>(blocks < 4096 ? blocks : 4096);
    castToFloatKernel<T><<<grid, block, 0, stream>>>(static_cast<const T *>(x), static_cast<float *>(workspace), total);
    return INFINI_STATUS_SUCCESS;
}

} // namespace

namespace op::dsv4_linear_bf16_fp32::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    Info info;
    CHECK_STATUS(createInfo(&info, y_desc, x_desc, w_desc));
    size_t workspace_size = info.m * info.k * sizeof(float);
    *desc_ptr = new Descriptor(info, workspace_size, new Opaque{handle->internal()}, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size, void *y, const void *x, const void *w, void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    size_t x_numel = _info.m * _info.k;
    switch (_info.x_dtype) {
    case INFINI_DTYPE_F16:
        CHECK_STATUS(castToFloat<half>(x, workspace, x_numel, cuda_stream));
        break;
    case INFINI_DTYPE_BF16:
        CHECK_STATUS(castToFloat<__nv_bfloat16>(x, workspace, x_numel, cuda_stream));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
    cudaDataType compute_type = CUDA_R_32F;
#else
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#endif

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int m = static_cast<int>(_info.m);
    const int n = static_cast<int>(_info.n);
    const int k = static_cast<int>(_info.k);
    const float *x_fp32 = static_cast<const float *>(workspace);

    CHECK_STATUS(_opaque->internal->useCublas(
        cuda_stream,
        [&](cublasHandle_t handle) {
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha,
                w,
                CUDA_R_32F,
                k,
                x_fp32,
                CUDA_R_32F,
                k,
                &beta,
                y,
                CUDA_R_32F,
                n,
                compute_type,
                CUBLAS_GEMM_DEFAULT));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_linear_bf16_fp32::nvidia
