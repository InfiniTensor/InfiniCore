#include "../../../devices/nvidia/nvidia_common.cuh"
#include "ernie45_rope_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

template <typename T>
__device__ inline float load_float(const T *ptr, size_t idx) {
    return static_cast<float>(ptr[idx]);
}

template <>
__device__ inline float load_float<half>(const half *ptr, size_t idx) {
    return __half2float(ptr[idx]);
}

template <>
__device__ inline float load_float<cuda_bfloat16>(const cuda_bfloat16 *ptr, size_t idx) {
    return __bfloat162float(ptr[idx]);
}

template <typename T>
__device__ inline void store_float(T *ptr, size_t idx, float value) {
    ptr[idx] = static_cast<T>(value);
}

template <>
__device__ inline void store_float<half>(half *ptr, size_t idx, float value) {
    ptr[idx] = __float2half(value);
}

template <>
__device__ inline void store_float<cuda_bfloat16>(cuda_bfloat16 *ptr, size_t idx, float value) {
    ptr[idx] = __float2bfloat16(value);
}

template <typename Tindex>
__device__ inline int64_t read_pos(const Tindex *positions, size_t offset) {
    return static_cast<int64_t>(positions[offset]);
}

template <typename Tdata, typename Tindex>
INFINIOP_CUDA_KERNEL ernie45_mrope_kernel(
    Tdata *q,
    Tdata *k,
    const Tindex *__restrict__ positions,
    size_t seqlen,
    size_t q_heads,
    size_t k_heads,
    size_t head_dim,
    ptrdiff_t q_stride_seq,
    ptrdiff_t q_stride_head,
    ptrdiff_t k_stride_seq,
    ptrdiff_t k_stride_head,
    ptrdiff_t pos_stride_seq,
    ptrdiff_t pos_stride_axis,
    bool pos_axis_first,
    double rope_theta,
    size_t section_h,
    size_t section_w) {
    const size_t token = blockIdx.x;
    const size_t head = blockIdx.y;
    const bool is_q = blockIdx.z == 0;
    const size_t nheads = is_q ? q_heads : k_heads;
    if (token >= seqlen || head >= nheads) {
        return;
    }

    const size_t half_dim = head_dim / 2;
    const size_t section_hw = section_h + section_w;
    const size_t pos_base = token * pos_stride_seq;
    const int64_t tpos = read_pos(positions, pos_base + 0 * pos_stride_axis);
    const int64_t hpos = read_pos(positions, pos_base + 1 * pos_stride_axis);
    const int64_t wpos = read_pos(positions, pos_base + 2 * pos_stride_axis);

    Tdata *base = is_q ? q + token * q_stride_seq + head * q_stride_head
                       : k + token * k_stride_seq + head * k_stride_head;

    for (size_t pair = threadIdx.x; pair < half_dim; pair += blockDim.x) {
        const int64_t pos = pair < section_hw ? ((pair & 1) == 0 ? hpos : wpos) : tpos;
        const float inv_freq = powf(static_cast<float>(rope_theta),
                                    -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
        float sinv;
        float cosv;
        __sincosf(static_cast<float>(pos) * inv_freq, &sinv, &cosv);

        const size_t even = 2 * pair;
        const size_t odd = even + 1;
        const float x0 = load_float(base, even);
        const float x1 = load_float(base, odd);
        store_float(base, even, x0 * cosv - x1 * sinv);
        store_float(base, odd, x1 * cosv + x0 * sinv);
    }
}

template <typename Tdata, typename Tindex>
INFINIOP_CUDA_KERNEL ernie45_vision_rope_kernel(
    Tdata *q,
    Tdata *k,
    const Tindex *__restrict__ positions,
    size_t seqlen,
    size_t q_heads,
    size_t k_heads,
    size_t head_dim,
    ptrdiff_t q_stride_seq,
    ptrdiff_t q_stride_head,
    ptrdiff_t k_stride_seq,
    ptrdiff_t k_stride_head,
    ptrdiff_t pos_stride_seq,
    ptrdiff_t pos_stride_axis,
    double rope_theta) {
    const size_t token = blockIdx.x;
    const size_t head = blockIdx.y;
    const bool is_q = blockIdx.z == 0;
    const size_t nheads = is_q ? q_heads : k_heads;
    if (token >= seqlen || head >= nheads) {
        return;
    }

    const size_t half_dim = head_dim / 2;
    const size_t quarter_dim = head_dim / 4;
    const int64_t hpos = read_pos(positions, token * pos_stride_seq + 0 * pos_stride_axis);
    const int64_t wpos = read_pos(positions, token * pos_stride_seq + 1 * pos_stride_axis);

    Tdata *base = is_q ? q + token * q_stride_seq + head * q_stride_head
                       : k + token * k_stride_seq + head * k_stride_head;

    for (size_t i = threadIdx.x; i < half_dim; i += blockDim.x) {
        const int64_t pos = i < quarter_dim ? hpos : wpos;
        const size_t freq_idx = i < quarter_dim ? i : i - quarter_dim;
        const float inv_freq = powf(static_cast<float>(rope_theta),
                                    -2.0f * static_cast<float>(freq_idx) / static_cast<float>(half_dim));
        float sinv;
        float cosv;
        __sincosf(static_cast<float>(pos) * inv_freq, &sinv, &cosv);

        const size_t lo = i;
        const size_t hi = i + half_dim;
        const float x0 = load_float(base, lo);
        const float x1 = load_float(base, hi);
        store_float(base, lo, x0 * cosv - x1 * sinv);
        store_float(base, hi, x1 * cosv + x0 * sinv);
    }
}

} // namespace

namespace op::ernie45_rope::nvidia {

struct MropeDescriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

MropeDescriptor::~MropeDescriptor() {
    delete _opaque;
}

infiniStatus_t MropeDescriptor::create(infiniopHandle_t handle_,
                                       MropeDescriptor **desc_ptr,
                                       infiniopTensorDescriptor_t q_desc,
                                       infiniopTensorDescriptor_t k_desc,
                                       infiniopTensorDescriptor_t pos_desc,
                                       double theta,
                                       size_t section_h,
                                       size_t section_w,
                                       size_t section_t) {
    auto result = QKInfo::create(q_desc, k_desc, pos_desc, theta, section_h, section_w, section_t);
    CHECK_RESULT(result);
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    *desc_ptr = new MropeDescriptor(
        new Opaque{handle->internal()},
        result.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tindex>
infiniStatus_t launch_mrope(const QKInfo &info, Tdata *q, Tdata *k, const Tindex *positions, cudaStream_t stream) {
    dim3 grid(static_cast<unsigned int>(info.seqlen),
              static_cast<unsigned int>(std::max(info.q_heads, info.k_heads)),
              2);
    ernie45_mrope_kernel<<<grid, 128, 0, stream>>>(
        q, k, positions,
        info.seqlen, info.q_heads, info.k_heads, info.head_dim,
        info.q_stride_seq, info.q_stride_head,
        info.k_stride_seq, info.k_stride_head,
        info.pos_stride_seq, info.pos_stride_axis, info.pos_axis_first,
        info.rope_theta, info.section_h, info.section_w);
    return INFINI_STATUS_SUCCESS;
}

#define LAUNCH_MROPE(TDATA, TINDEX) launch_mrope(_info, (TDATA *)q, (TDATA *)k, (const TINDEX *)positions, (cudaStream_t)stream)
#define MROPE_POS_TYPE(TDATA)                  \
    switch (_info.pos_type) {                  \
    case INFINI_DTYPE_U8:                      \
        return LAUNCH_MROPE(TDATA, uint8_t);   \
    case INFINI_DTYPE_U16:                     \
        return LAUNCH_MROPE(TDATA, uint16_t);  \
    case INFINI_DTYPE_U32:                     \
        return LAUNCH_MROPE(TDATA, uint32_t);  \
    case INFINI_DTYPE_U64:                     \
        return LAUNCH_MROPE(TDATA, uint64_t);  \
    case INFINI_DTYPE_I8:                      \
        return LAUNCH_MROPE(TDATA, int8_t);    \
    case INFINI_DTYPE_I16:                     \
        return LAUNCH_MROPE(TDATA, int16_t);   \
    case INFINI_DTYPE_I32:                     \
        return LAUNCH_MROPE(TDATA, int32_t);   \
    case INFINI_DTYPE_I64:                     \
        return LAUNCH_MROPE(TDATA, int64_t);   \
    default:                                   \
        return INFINI_STATUS_BAD_TENSOR_DTYPE; \
    }

infiniStatus_t MropeDescriptor::calculate(void *workspace,
                                          size_t workspace_size,
                                          void *q,
                                          void *k,
                                          const void *positions,
                                          void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        MROPE_POS_TYPE(half);
    case INFINI_DTYPE_BF16:
        MROPE_POS_TYPE(cuda_bfloat16);
    case INFINI_DTYPE_F32:
        MROPE_POS_TYPE(float);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef MROPE_POS_TYPE
#undef LAUNCH_MROPE

struct VisionRopeDescriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

VisionRopeDescriptor::~VisionRopeDescriptor() {
    delete _opaque;
}

infiniStatus_t VisionRopeDescriptor::create(infiniopHandle_t handle_,
                                            VisionRopeDescriptor **desc_ptr,
                                            infiniopTensorDescriptor_t q_desc,
                                            infiniopTensorDescriptor_t k_desc,
                                            infiniopTensorDescriptor_t pos_desc,
                                            double theta) {
    auto result = VisionInfo::create(q_desc, k_desc, pos_desc, theta);
    CHECK_RESULT(result);
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    *desc_ptr = new VisionRopeDescriptor(
        new Opaque{handle->internal()},
        result.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tindex>
infiniStatus_t launch_vision_rope(const VisionInfo &info, Tdata *q, Tdata *k, const Tindex *positions, cudaStream_t stream) {
    dim3 grid(static_cast<unsigned int>(info.seqlen),
              static_cast<unsigned int>(std::max(info.q_heads, info.k_heads)),
              2);
    ernie45_vision_rope_kernel<<<grid, 128, 0, stream>>>(
        q, k, positions,
        info.seqlen, info.q_heads, info.k_heads, info.head_dim,
        info.q_stride_seq, info.q_stride_head,
        info.k_stride_seq, info.k_stride_head,
        info.pos_stride_seq, info.pos_stride_axis,
        info.rope_theta);
    return INFINI_STATUS_SUCCESS;
}

#define LAUNCH_VISION(TDATA, TINDEX) launch_vision_rope(_info, (TDATA *)q, (TDATA *)k, (const TINDEX *)positions, (cudaStream_t)stream)
#define VISION_POS_TYPE(TDATA)                 \
    switch (_info.pos_type) {                  \
    case INFINI_DTYPE_U8:                      \
        return LAUNCH_VISION(TDATA, uint8_t);  \
    case INFINI_DTYPE_U16:                     \
        return LAUNCH_VISION(TDATA, uint16_t); \
    case INFINI_DTYPE_U32:                     \
        return LAUNCH_VISION(TDATA, uint32_t); \
    case INFINI_DTYPE_U64:                     \
        return LAUNCH_VISION(TDATA, uint64_t); \
    case INFINI_DTYPE_I8:                      \
        return LAUNCH_VISION(TDATA, int8_t);   \
    case INFINI_DTYPE_I16:                     \
        return LAUNCH_VISION(TDATA, int16_t);  \
    case INFINI_DTYPE_I32:                     \
        return LAUNCH_VISION(TDATA, int32_t);  \
    case INFINI_DTYPE_I64:                     \
        return LAUNCH_VISION(TDATA, int64_t);  \
    default:                                   \
        return INFINI_STATUS_BAD_TENSOR_DTYPE; \
    }

infiniStatus_t VisionRopeDescriptor::calculate(void *workspace,
                                               size_t workspace_size,
                                               void *q,
                                               void *k,
                                               const void *positions,
                                               void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        VISION_POS_TYPE(half);
    case INFINI_DTYPE_BF16:
        VISION_POS_TYPE(cuda_bfloat16);
    case INFINI_DTYPE_F32:
        VISION_POS_TYPE(float);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef VISION_POS_TYPE
#undef LAUNCH_VISION

} // namespace op::ernie45_rope::nvidia
