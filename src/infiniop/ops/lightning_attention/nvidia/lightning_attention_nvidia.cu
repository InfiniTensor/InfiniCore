#include "../../../devices/nvidia/nvidia_common.cuh"
#include "lightning_attention_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace op::lightning_attention::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

// One block per (batch, head); one thread per state/output column. The block is
// launched with exactly `D` threads (`D <= maxThreadsPerBlock()`, validated in
// `Descriptor::calculate`), so every thread reaches each `__syncthreads()`.
//
// The recurrence is staged into the destination row of the state pool: the
// initial row is copied to the final row first and the accumulation happens in
// place on the final row. This keeps the initial row untouched, matching the
// CPU implementation when `initial_state_indices != final_state_indices`.
//
// NOTE: fp32 only for now; fp16/bf16 kernels are a follow-up.
INFINIOP_CUDA_KERNEL lightningAttentionKernel(
    const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
    float *__restrict__ out, float *__restrict__ state_pool,
    const float *__restrict__ slope,
    const int32_t *__restrict__ init_idx, const int32_t *__restrict__ final_idx,
    size_t T, size_t D,
    ptrdiff_t q_sb, ptrdiff_t q_st, ptrdiff_t q_sh, ptrdiff_t q_sd,
    ptrdiff_t k_sb, ptrdiff_t k_st, ptrdiff_t k_sh, ptrdiff_t k_sd,
    ptrdiff_t v_sb, ptrdiff_t v_st, ptrdiff_t v_sh, ptrdiff_t v_sd,
    ptrdiff_t o_sb, ptrdiff_t o_st, ptrdiff_t o_sh, ptrdiff_t o_sd,
    ptrdiff_t s_s0, ptrdiff_t s_s1, ptrdiff_t s_s2, ptrdiff_t s_s3,
    size_t slope_stride) {
    const size_t b = blockIdx.y;
    const size_t h = blockIdx.x;
    const size_t tid = threadIdx.x;

    extern __shared__ float smem[];
    float *s_k = smem;
    float *s_q = smem + D;

    const size_t init_row = static_cast<size_t>(init_idx[b]);
    const size_t final_row = static_cast<size_t>(final_idx[b]);
    const float ratio = __expf(-slope[h * slope_stride]);

    float *S = state_pool + final_row * s_s0 + h * s_s1;
    const float *S_init = state_pool + init_row * s_s0 + h * s_s1;
    if (init_row != final_row) {
        for (size_t idx = tid; idx < D * D; idx += D) {
            const size_t i = idx / D;
            const size_t j = idx % D;
            S[i * s_s2 + j * s_s3] = S_init[i * s_s2 + j * s_s3];
        }
    }
    __syncthreads();

    for (size_t t = 0; t < T; ++t) {
        s_k[tid] = k[b * k_sb + t * k_st + h * k_sh + tid * k_sd];
        s_q[tid] = q[b * q_sb + t * q_st + h * q_sh + tid * q_sd];
        __syncthreads();

        // S[i][j] = ratio * S[i][j] + k[i] * v[j]   (thread j owns column j)
        const float v_j = v[b * v_sb + t * v_st + h * v_sh + tid * v_sd];
        for (size_t i = 0; i < D; ++i) {
            float *s_ij = S + i * s_s2 + tid * s_s3;
            *s_ij = ratio * (*s_ij) + s_k[i] * v_j;
        }

        // o[j] = sum_i q[i] * S[i][j]
        float acc = 0.0f;
        for (size_t i = 0; i < D; ++i) {
            acc += s_q[i] * S[i * s_s2 + tid * s_s3];
        }
        out[b * o_sb + t * o_st + h * o_sh + tid * o_sd] = acc;

        __syncthreads(); // All threads must finish reading s_k/s_q before reloading.
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t slope_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc) {
    auto info = LightningAttentionInfo::create(out_desc, initial_state_desc,
                                               q_desc, k_desc, v_desc, slope_desc,
                                               initial_state_indices_desc,
                                               final_state_indices_desc);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, void *initial_state,
    const void *q, const void *k, const void *v,
    const void *slope,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream_) const {
    (void)workspace;
    (void)workspace_size;

    if (_info.data_dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (_info.index_dtype != INFINI_DTYPE_I32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (_info.D > static_cast<size_t>(_opaque->internal->maxThreadsPerBlock())) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (_info.B > static_cast<size_t>(65535)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
    const auto &info = _info;
    dim3 grid(static_cast<unsigned int>(info.H), static_cast<unsigned int>(info.B));
    const size_t smem_bytes = 2 * info.D * sizeof(float);
    lightningAttentionKernel<<<grid, static_cast<unsigned int>(info.D), smem_bytes, stream>>>(
        static_cast<const float *>(q), static_cast<const float *>(k), static_cast<const float *>(v),
        static_cast<float *>(out), static_cast<float *>(initial_state),
        static_cast<const float *>(slope),
        static_cast<const int32_t *>(initial_state_indices),
        static_cast<const int32_t *>(final_state_indices),
        info.T, info.D,
        info.q_strides[0], info.q_strides[1], info.q_strides[2], info.q_strides[3],
        info.k_strides[0], info.k_strides[1], info.k_strides[2], info.k_strides[3],
        info.v_strides[0], info.v_strides[1], info.v_strides[2], info.v_strides[3],
        info.out_strides[0], info.out_strides[1], info.out_strides[2], info.out_strides[3],
        info.initial_state_strides[0], info.initial_state_strides[1],
        info.initial_state_strides[2], info.initial_state_strides[3],
        static_cast<size_t>(info.slope_strides[0]));

    if (cudaGetLastError() != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::lightning_attention::nvidia

