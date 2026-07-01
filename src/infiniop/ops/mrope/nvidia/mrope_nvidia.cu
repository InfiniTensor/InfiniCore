#include "mrope_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "../cuda/kernel.cuh"

namespace {

template <typename Tdata, typename Tpos>
INFINIOP_CUDA_KERNEL mropeKernel(
    Tdata *q_out,
    Tdata *k_out,
    const Tdata *q,
    const Tdata *k,
    const Tdata *cos,
    const Tdata *sin,
    const Tpos *positions,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_size,
    size_t rotary_dim,
    size_t half_rotary_dim,
    ptrdiff_t q_out_stride_token,
    ptrdiff_t q_out_stride_head,
    ptrdiff_t k_out_stride_token,
    ptrdiff_t k_out_stride_head,
    ptrdiff_t q_stride_token,
    ptrdiff_t q_stride_head,
    ptrdiff_t k_stride_token,
    ptrdiff_t k_stride_head,
    ptrdiff_t cos_stride_position,
    ptrdiff_t sin_stride_position,
    ptrdiff_t positions_stride_axis,
    ptrdiff_t positions_stride_token,
    size_t max_position_embeddings,
    size_t section_t,
    size_t section_h,
    size_t section_w,
    bool positions_has_axes,
    bool interleaved) {
    op::mrope::cuda::mropeBlock<Tdata, float, Tpos>(
        q_out, k_out, q, k, cos, sin, positions, num_q_heads, num_kv_heads, head_size, rotary_dim, half_rotary_dim,
        q_out_stride_token, q_out_stride_head, k_out_stride_token, k_out_stride_head,
        q_stride_token, q_stride_head, k_stride_token, k_stride_head,
        cos_stride_position, sin_stride_position,
        positions_stride_axis, positions_stride_token, max_position_embeddings,
        section_t, section_h, section_w, positions_has_axes, interleaved);
}

template <typename Tdata, typename Tpos>
infiniStatus_t launchMRoPE(
    const MRoPEInfo &info,
    int block_size,
    Tdata *q_out,
    Tdata *k_out,
    const Tdata *q,
    const Tdata *k,
    const Tdata *cos,
    const Tdata *sin,
    const Tpos *positions,
    cudaStream_t stream) {
    dim3 grid_dim(uint32_t(info.num_tokens), uint32_t(std::max(info.num_q_heads, info.num_kv_heads)));
    int nthreads = std::max(int(info.half_rotary_dim), std::min(block_size, 256));
    mropeKernel<Tdata, Tpos><<<grid_dim, nthreads, 0, stream>>>(
        q_out, k_out, q, k, cos, sin, positions,
        info.num_q_heads, info.num_kv_heads, info.head_size, info.rotary_dim, info.half_rotary_dim,
        info.q_out_stride_token, info.q_out_stride_head, info.k_out_stride_token, info.k_out_stride_head,
        info.q_stride_token, info.q_stride_head, info.k_stride_token, info.k_stride_head,
        info.cos_stride_position, info.sin_stride_position,
        info.positions_stride_axis, info.positions_stride_token, info.max_position_embeddings,
        info.section_t, info.section_h, info.section_w, info.positions_has_axes, info.interleaved);
    return INFINI_STATUS_SUCCESS;
}

} // namespace

namespace op::mrope::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t q_out_desc,
    infiniopTensorDescriptor_t k_out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t cos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t positions_desc,
    int head_size,
    int rotary_dim,
    int section_t,
    int section_h,
    int section_w,
    bool interleaved) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto info = MRoPEInfo::create(q_out_desc, k_out_desc, q_desc, k_desc, cos_desc, sin_desc, positions_desc,
                                  head_size, rotary_dim, section_t, section_h, section_w, interleaved);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        info.take(),
        0,
        new Opaque{handle->internal()},
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *q_out,
    void *k_out,
    const void *q,
    const void *k,
    const void *cos,
    const void *sin,
    const void *positions,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;
    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        if (_info.position_type == INFINI_DTYPE_I32) {
            return launchMRoPE<half, int32_t>(_info, _opaque->internal->maxThreadsPerBlock(),
                                              static_cast<half *>(q_out), static_cast<half *>(k_out),
                                              static_cast<const half *>(q), static_cast<const half *>(k),
                                              static_cast<const half *>(cos), static_cast<const half *>(sin),
                                              static_cast<const int32_t *>(positions), static_cast<cudaStream_t>(stream));
        }
        return launchMRoPE<half, int64_t>(_info, _opaque->internal->maxThreadsPerBlock(),
                                          static_cast<half *>(q_out), static_cast<half *>(k_out),
                                          static_cast<const half *>(q), static_cast<const half *>(k),
                                          static_cast<const half *>(cos), static_cast<const half *>(sin),
                                          static_cast<const int64_t *>(positions), static_cast<cudaStream_t>(stream));
    case INFINI_DTYPE_BF16:
        if (_info.position_type == INFINI_DTYPE_I32) {
            return launchMRoPE<cuda_bfloat16, int32_t>(_info, _opaque->internal->maxThreadsPerBlock(),
                                                       static_cast<cuda_bfloat16 *>(q_out), static_cast<cuda_bfloat16 *>(k_out),
                                                       static_cast<const cuda_bfloat16 *>(q), static_cast<const cuda_bfloat16 *>(k),
                                                       static_cast<const cuda_bfloat16 *>(cos), static_cast<const cuda_bfloat16 *>(sin),
                                                       static_cast<const int32_t *>(positions), static_cast<cudaStream_t>(stream));
        }
        return launchMRoPE<cuda_bfloat16, int64_t>(_info, _opaque->internal->maxThreadsPerBlock(),
                                                   static_cast<cuda_bfloat16 *>(q_out), static_cast<cuda_bfloat16 *>(k_out),
                                                   static_cast<const cuda_bfloat16 *>(q), static_cast<const cuda_bfloat16 *>(k),
                                                   static_cast<const cuda_bfloat16 *>(cos), static_cast<const cuda_bfloat16 *>(sin),
                                                   static_cast<const int64_t *>(positions), static_cast<cudaStream_t>(stream));
    case INFINI_DTYPE_F32:
        if (_info.position_type == INFINI_DTYPE_I32) {
            return launchMRoPE<float, int32_t>(_info, _opaque->internal->maxThreadsPerBlock(),
                                               static_cast<float *>(q_out), static_cast<float *>(k_out),
                                               static_cast<const float *>(q), static_cast<const float *>(k),
                                               static_cast<const float *>(cos), static_cast<const float *>(sin),
                                               static_cast<const int32_t *>(positions), static_cast<cudaStream_t>(stream));
        }
        return launchMRoPE<float, int64_t>(_info, _opaque->internal->maxThreadsPerBlock(),
                                           static_cast<float *>(q_out), static_cast<float *>(k_out),
                                           static_cast<const float *>(q), static_cast<const float *>(k),
                                           static_cast<const float *>(cos), static_cast<const float *>(sin),
                                           static_cast<const int64_t *>(positions), static_cast<cudaStream_t>(stream));
    case INFINI_DTYPE_F64:
        if (_info.position_type == INFINI_DTYPE_I32) {
            return launchMRoPE<double, int32_t>(_info, _opaque->internal->maxThreadsPerBlock(),
                                                static_cast<double *>(q_out), static_cast<double *>(k_out),
                                                static_cast<const double *>(q), static_cast<const double *>(k),
                                                static_cast<const double *>(cos), static_cast<const double *>(sin),
                                                static_cast<const int32_t *>(positions), static_cast<cudaStream_t>(stream));
        }
        return launchMRoPE<double, int64_t>(_info, _opaque->internal->maxThreadsPerBlock(),
                                            static_cast<double *>(q_out), static_cast<double *>(k_out),
                                            static_cast<const double *>(q), static_cast<const double *>(k),
                                            static_cast<const double *>(cos), static_cast<const double *>(sin),
                                            static_cast<const int64_t *>(positions), static_cast<cudaStream_t>(stream));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::mrope::nvidia
