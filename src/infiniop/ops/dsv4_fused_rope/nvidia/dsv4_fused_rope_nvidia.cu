#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_fused_rope_nvidia.cuh"
namespace {
template <typename T>
__device__ float toFloat(T v) { return static_cast<float>(v); }
template <typename T>
__device__ T fromFloat(float v) { return static_cast<T>(v); }
template <typename T>
__global__ void kernel(T *q, T *k, const float *fr, const float *fi, size_t seq_len, size_t q_heads, size_t k_heads, size_t rope_dim, bool has_k) {
    size_t s = blockIdx.x;
    size_t h = blockIdx.y;
    size_t half = rope_dim / 2;
    if (s >= seq_len || h >= q_heads) {
        return;
    }
    for (size_t i = threadIdx.x; i < half; i += blockDim.x) {
        float real = fr[s * half + i];
        float imag = fi[s * half + i];
        size_t even = s * q_heads * rope_dim + h * rope_dim + 2 * i;
        size_t odd = even + 1;
        float x0 = toFloat(q[even]);
        float x1 = toFloat(q[odd]);
        q[even] = fromFloat<T>(x0 * real - x1 * imag);
        q[odd] = fromFloat<T>(x0 * imag + x1 * real);
    }
    if (has_k && h < k_heads) {
        for (size_t i = threadIdx.x; i < half; i += blockDim.x) {
            float real = fr[s * half + i];
            float imag = fi[s * half + i];
            size_t even = s * k_heads * rope_dim + h * rope_dim + 2 * i;
            size_t odd = even + 1;
            float x0 = toFloat(k[even]);
            float x1 = toFloat(k[odd]);
            k[even] = fromFloat<T>(x0 * real - x1 * imag);
            k[odd] = fromFloat<T>(x0 * imag + x1 * real);
        }
    }
}
template <typename T>
infiniStatus_t launch(const op::dsv4_fused_rope::Info &info, void *q, void *k, const void *fr, const void *fi, cudaStream_t stream) {
    dim3 grid(info.seq_len, info.q_heads);
    int threads = static_cast<int>((info.rope_dim / 2) < 256 ? (info.rope_dim / 2) : 256);
    kernel<T><<<grid, threads, 0, stream>>>(static_cast<T *>(q), static_cast<T *>(k), static_cast<const float *>(fr), static_cast<const float *>(fi), info.seq_len, info.q_heads, info.k_heads, info.rope_dim, info.has_k);
    return INFINI_STATUS_SUCCESS;
}
} // namespace
namespace op::dsv4_fused_rope::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t k_desc, infiniopTensorDescriptor_t freq_real_desc, infiniopTensorDescriptor_t freq_imag_desc, int has_k) {
    Info info;
    CHECK_STATUS(createInfo(&info, q_desc, k_desc, freq_real_desc, freq_imag_desc, has_k));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, void *q, void *k, const void *fr, const void *fi, void *stream) const {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launch<half>(_info, q, k, fr, fi, s);
    case INFINI_DTYPE_BF16:
        return launch<__nv_bfloat16>(_info, q, k, fr, fi, s);
    case INFINI_DTYPE_F32:
        return launch<float>(_info, q, k, fr, fi, s);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
} // namespace op::dsv4_fused_rope::nvidia
