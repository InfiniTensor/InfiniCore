#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_mhc_pre_nvidia.cuh"

namespace {
constexpr int BLOCK = 128;

template <typename T>
__device__ float toFloat(T v) {
    return static_cast<float>(v);
}

template <typename T>
__device__ T fromFloat(float v) {
    return static_cast<T>(v);
}

template <typename T, int BS>
__global__ void mhcPreKernel(const T *__restrict__ input,
                             const float *__restrict__ scale,
                             const float *__restrict__ base,
                             T *__restrict__ output,
                             int rows,
                             int cols,
                             float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int tid = threadIdx.x;
    int off = row * cols;
    float s = scale[0];
    for (int j = tid; j < cols; j += BS) {
        float v = toFloat(input[off + j]);
        float result = 1.0f / (1.0f + expf(-(v * s + base[j]))) + eps;
        output[off + j] = fromFloat<T>(result);
    }
}

template <>
__device__ half fromFloat<half>(float v) {
    return __float2half(v);
}

template <>
__device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
infiniStatus_t launch(const op::dsv4_mhc_pre::Info &info, void *output, const void *input, const void *scale, const void *base, cudaStream_t stream) {
    mhcPreKernel<T, BLOCK><<<static_cast<unsigned int>(info.rows), BLOCK, 0, stream>>>(
        static_cast<const T *>(input),
        static_cast<const float *>(scale),
        static_cast<const float *>(base),
        static_cast<T *>(output),
        static_cast<int>(info.rows),
        static_cast<int>(info.cols),
        info.eps);
    return INFINI_STATUS_SUCCESS;
}
} // namespace

namespace op::dsv4_mhc_pre::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t base_desc, float eps) {
    Info info;
    CHECK_STATUS(createInfo(&info, output_desc, input_desc, scale_desc, base_desc, eps));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *, size_t, void *output, const void *input, const void *scale, const void *base, void *stream) const {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launch<half>(_info, output, input, scale, base, s);
    case INFINI_DTYPE_BF16:
        return launch<__nv_bfloat16>(_info, output, input, scale, base, s);
    case INFINI_DTYPE_F32:
        return launch<float>(_info, output, input, scale, base, s);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::dsv4_mhc_pre::nvidia
