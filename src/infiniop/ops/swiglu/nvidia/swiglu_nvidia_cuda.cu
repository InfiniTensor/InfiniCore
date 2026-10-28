#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel_cuda.cuh"
#include "swiglu_nvidia_cuda.cuh"

template <typename T, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL SwiGLUCuda(
    T *c,
    const T *a,
    const T *b,
    int length,
    size_t batch, size_t seq_len, size_t hidden_dim,
    ptrdiff_t c_strides_0, ptrdiff_t c_strides_1, ptrdiff_t c_strides_2,
    ptrdiff_t a_strides_0, ptrdiff_t a_strides_1, ptrdiff_t a_strides_2,
    ptrdiff_t b_strides_0, ptrdiff_t b_strides_1, ptrdiff_t b_strides_2) {
    SwiGLUCudaKernel<T, BLOCK_SIZE>(c, a, b, length, batch, seq_len, hidden_dim,
                                    c_strides_0, c_strides_1, c_strides_2,
                                    a_strides_0, a_strides_1, a_strides_2,
                                    b_strides_0, b_strides_1, b_strides_2);
}

#if defined(ENABLE_ILUVATAR_API)
struct alignas(16) SwiGLUBFloat8 {
    __nv_bfloat16 values[8];
};
INFINIOP_CUDA_KERNEL swigluBFloatVector(
    __nv_bfloat16 *c, const __nv_bfloat16 *a, const __nv_bfloat16 *b,
    size_t dim, size_t seq_len, size_t tiles,
    ptrdiff_t c0, ptrdiff_t c1, ptrdiff_t a0, ptrdiff_t a1, ptrdiff_t b0, ptrdiff_t b1) {
    const size_t row = blockIdx.x / tiles;
    const size_t col = (blockIdx.x % tiles * blockDim.x + threadIdx.x) * 8;
    if (col >= dim) {
        return;
    }
    const auto ai = row / seq_len * a0 + row % seq_len * a1 + col;
    const auto bi = row / seq_len * b0 + row % seq_len * b1 + col;
    const auto ci = row / seq_len * c0 + row % seq_len * c1 + col;
    SwiGLUBFloat8 up, gate, out;
    *reinterpret_cast<uint4 *>(&up) = *reinterpret_cast<const uint4 *>(a + ai);
    *reinterpret_cast<uint4 *>(&gate) = *reinterpret_cast<const uint4 *>(b + bi);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float g = __bfloat162float(gate.values[i]);
        // Preserve the existing BF16 sigmoid rounding and multiplication order.
        const float sig = __bfloat162float(sigmoid(gate.values[i]));
        out.values[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(g, sig), __bfloat162float(up.values[i])));
    }
    *reinterpret_cast<uint4 *>(c + ci) = *reinterpret_cast<uint4 *>(&out);
}
#endif

namespace op::swiglu_cuda::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

    auto info = SwiGLUCudaInfo::createSwiGLUCudaInfo(c_desc, a_desc, b_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename T>
infiniStatus_t calculate_swiglu_cuda(
    const SwiGLUCudaInfo &info,
    T *c,
    const T *a,
    const T *b,
    cudaStream_t stream,
    void *workspace) {

    int length = (int)info.length;
    size_t batch = info.batch;
    size_t seq_len = info.seq_len;
    size_t hidden_dim = info.hidden_dim;
    ptrdiff_t c_strides_0 = info.c_strides_0;
    ptrdiff_t c_strides_1 = info.c_strides_1;
    ptrdiff_t c_strides_2 = info.c_strides_2;
    ptrdiff_t a_strides_0 = info.a_strides_0;
    ptrdiff_t a_strides_1 = info.a_strides_1;
    ptrdiff_t a_strides_2 = info.a_strides_2;
    ptrdiff_t b_strides_0 = info.b_strides_0;
    ptrdiff_t b_strides_1 = info.b_strides_1;
    ptrdiff_t b_strides_2 = info.b_strides_2;

    int num_blocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    SwiGLUCuda<T, BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(c, a, b, length, batch, seq_len, hidden_dim,
                                                c_strides_0, c_strides_1, c_strides_2,
                                                a_strides_0, a_strides_1, a_strides_2,
                                                b_strides_0, b_strides_1, b_strides_2);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream_) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    cudaStream_t stream = (cudaStream_t)stream_;

#if defined(ENABLE_ILUVATAR_API)
    const auto aligned = [](const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
    if (device_type == INFINI_DEVICE_ILUVATAR && _info.dtype == INFINI_DTYPE_BF16
        && _info.length >= 16384 && _info.hidden_dim >= 256 && _info.hidden_dim % 8 == 0
        && _info.a_strides_2 == 1 && _info.b_strides_2 == 1 && _info.c_strides_2 == 1
        && _info.a_strides_0 % 8 == 0 && _info.a_strides_1 % 8 == 0
        && _info.b_strides_0 % 8 == 0 && _info.b_strides_1 % 8 == 0
        && _info.c_strides_0 % 8 == 0 && _info.c_strides_1 % 8 == 0
        && aligned(a) && aligned(b) && aligned(c)) {
        const size_t tiles = (_info.hidden_dim + 2047) / 2048;
        swigluBFloatVector<<<_info.batch * _info.seq_len * tiles, 256, 0, stream>>>(
            static_cast<__nv_bfloat16 *>(c), static_cast<const __nv_bfloat16 *>(a), static_cast<const __nv_bfloat16 *>(b),
            _info.hidden_dim, _info.seq_len, tiles,
            _info.c_strides_0, _info.c_strides_1, _info.a_strides_0, _info.a_strides_1, _info.b_strides_0, _info.b_strides_1);
        return INFINI_STATUS_SUCCESS;
    }
#endif

#define CALCULATE_SWIGLU_CUDA(BLOCK_SIZE, TDATA) \
    calculate_swiglu_cuda<BLOCK_SIZE, TDATA>(_info, (TDATA *)c, (const TDATA *)a, (const TDATA *)b, stream, workspace)
#define CALCULATE_SWIGLU_CUDA_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                                \
        if (_info.dtype == INFINI_DTYPE_F16)                         \
            return CALCULATE_SWIGLU_CUDA(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)                    \
            return CALCULATE_SWIGLU_CUDA(BLOCK_SIZE, float);         \
        else if (_info.dtype == INFINI_DTYPE_BF16)                   \
            return CALCULATE_SWIGLU_CUDA(BLOCK_SIZE, __nv_bfloat16); \
        else                                                         \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                   \
    }

    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CALCULATE_SWIGLU_CUDA_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CALCULATE_SWIGLU_CUDA_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CALCULATE_SWIGLU_CUDA_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_2048)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CALCULATE_SWIGLU_CUDA_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::swiglu_cuda::nvidia
